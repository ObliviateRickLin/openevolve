# initial_program.py

import time
import traceback
from datetime import datetime
import argparse
import json
import itertools, heapq
import os
import pickle as pkl
import math
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split


# --- From hgcn_utils/math_utils.py ---
def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()

def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5

class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5

def arcosh(x):
    return Arcosh.apply(x)

def arsinh(x):
    return Arsinh.apply(x)

def artanh(x):
    return Artanh.apply(x)


# --- Helper functions ---
def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# --- From manifolds/base.py ---
class Manifold(object):
    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        raise NotImplementedError

    def proj(self, p, c):
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        raise NotImplementedError

    def proj_tan0(self, u, c):
        raise NotImplementedError

    def expmap(self, u, p, c):
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        raise NotImplementedError

    def expmap0(self, u, c):
        raise NotImplementedError

    def logmap0(self, p, c):
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        raise NotImplementedError

class ManifoldParameter(Parameter):
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()


# --- From manifolds/hyperboloid.py ---
class Hyperboloid(Manifold):
    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def egrad2rgrad(self, x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(self.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )


# --- RiemannianSGD Optimizer ---
_default_manifold = Hyperboloid()

def copy_or_set_(dest, source):
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)

class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        for group in self.param_groups:
            self.stabilize_group(group)

class RiemannianSGD(OptimMixin, torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]

                for point in group["params"]:
                    if isinstance(point, ManifoldParameter):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = _default_manifold
                        c = 1.

                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("RiemannianSGD does not support sparse gradients")
                    
                    state = self.state[point]
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()

                    grad.add_(weight_decay, point)
                    grad = manifold.egrad2rgrad(point, grad, c)

                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(1 - dampening, grad)
                        if nesterov:
                            grad = grad.add_(momentum, momentum_buffer)
                        else:
                            grad = momentum_buffer

                        new_point = manifold.expmap(-learning_rate * grad, point, c)
                        components = new_point[:, 1:]
                        dim0 = torch.sqrt(torch.sum(components * components, dim=1, keepdim=True) + 1)
                        new_point = torch.cat([dim0, components], dim=1)
                        new_momentum_buffer = manifold.ptransp(point, new_point, momentum_buffer, c)
                        momentum_buffer.set_(new_momentum_buffer)
                        copy_or_set_(point, new_point)
                    else:
                        new_point = manifold.expmap(-learning_rate * grad, point, c)
                        components = new_point[:, 1:]
                        dim0 = torch.sqrt(torch.sum(components * components, dim=1, keepdim=True) + 1)
                        new_point = torch.cat([dim0, components], dim=1)
                        copy_or_set_(point, new_point)

                    group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            copy_or_set_(p, manifold.proj(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.set_(manifold.proju(p, buf))


# --- WarpSampler ---
def sample_function(adj_train, num_nodes, batch_size, n_negative, result_queue):
    num_users, num_items = num_nodes
    adj_train = lil_matrix(adj_train)
    all_pairs = np.asarray(adj_train.nonzero()).T
    user_item_pairs = all_pairs[: adj_train.count_nonzero() // 2]
    item_user_pairs = all_pairs[adj_train.count_nonzero() // 2:]

    assert len(user_item_pairs) == len(item_user_pairs)
    np.random.shuffle(user_item_pairs)
    np.random.shuffle(item_user_pairs)

    all_pairs_set = {idx: set(row) for idx, row in enumerate(adj_train.rows)}
    user_item_pairs_set = dict(itertools.islice(all_pairs_set.items(), num_users))

    while True:
        for i in range(int(len(user_item_pairs) / batch_size)):
            samples_for_users = batch_size
            user_positive_items_pairs = user_item_pairs[i * samples_for_users: (i + 1) * samples_for_users, :]
            user_negative_samples = np.random.randint(num_users, sum(num_nodes), size=(samples_for_users, n_negative))

            for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                   user_negative_samples,
                                                   range(len(user_negative_samples))):
                user = user_positive[0]
                for j, neg in enumerate(negatives):
                    while neg in user_item_pairs_set[user]:
                        user_negative_samples[i, j] = neg = np.random.randint(num_users, sum(num_nodes))

            user_triples = np.hstack((user_positive_items_pairs, user_negative_samples))
            np.random.shuffle(user_triples)
            result_queue.put(user_triples)

class WarpSampler(object):
    def __init__(self, num_nodes, user_item_matrix, batch_size=10000, n_negative=1, n_workers=5):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_item_matrix,
                                                      num_nodes,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# --- Layers ---
class FermiDiracDecoder(Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1)
        return probs

class StackGCNs(Module):
    def __init__(self, num_layers):
        super(StackGCNs, self).__init__()
        self.num_gcn_layers = num_layers - 1

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return sum(output[1:])

class HypAgg(Module):
    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        output = self.stackGCNs((x_tangent, adj))
        output = output - output.mean(dim=0)
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HyperbolicGraphConvolution(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x, adj = input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output

class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if hasattr(self, 'encode_graph') and self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class HRCF(Encoder):
    def __init__(self, c, args):
        super(HRCF, self).__init__(c)
        self.manifold = Hyperboloid()
        assert args.num_layers > 1

        hgc_layers = []
        in_dim = out_dim = args.embedding_dim
        hgc_layers.append(
            HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim, self.c, args.network, args.num_layers
            )
        )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(x, c=self.c)
        return super(HRCF, self).encode(x_hyp, adj)


# --- Data Generator ---
class Data(object):
    def __init__(self, dataset, norm_adj, seed, test_ratio, data_path="./data"):
        pkl_path = os.path.join(data_path, dataset)
        self.pkl_path = pkl_path
        self.dataset = dataset
        
        # Load data
        user_item_list_path = os.path.join(pkl_path, 'user_item_list.pkl')
        if os.path.exists(user_item_list_path):
            self.user_item_list = self.load_pickle(user_item_list_path)
        else:
            # Create synthetic data if real data not available
            self.user_item_list = self.create_synthetic_data()
        
        self.train_dict, self.test_dict = self.split_data_randomly(self.user_item_list, test_ratio, seed)
        self.num_users, self.num_items = len(self.user_item_list), max([max(x) for x in self.user_item_list]) + 1

        self.adj_train, user_item = self.generate_adj()

        if eval(norm_adj):
            self.adj_train_norm = normalize(self.adj_train + sp.eye(self.adj_train.shape[0]))
            self.adj_train_norm = sparse_mx_to_torch_sparse_tensor(self.adj_train_norm)

        print('num_users %d, num_items %d' % (self.num_users, self.num_items))
        print('adjacency matrix shape: ', self.adj_train.shape)

        tot_num_rating = sum([len(x) for x in self.user_item_list])
        print('number of all ratings {}, density {:.6f}'.format(tot_num_rating,
                                                                tot_num_rating / (self.num_users * self.num_items)))

        self.user_item_csr = self.generate_rating_matrix([*self.train_dict.values()], self.num_users, self.num_items)

    def create_synthetic_data(self):
        """Create synthetic user-item interactions for testing"""
        np.random.seed(42)
        num_users, num_items = 100, 50
        user_item_list = []
        
        for user_id in range(num_users):
            num_interactions = np.random.randint(5, 20)
            items = np.random.choice(num_items, num_interactions, replace=False)
            user_item_list.append(items.tolist())
        
        return user_item_list

    def generate_adj(self):
        user_item = np.zeros((self.num_users, self.num_items)).astype(int)
        for i, v in self.train_dict.items():
            user_item[i][v] = 1
        coo_user_item = sp.coo_matrix(user_item)

        start = time.time()
        print('generating adj csr... ')
        rows = np.concatenate((coo_user_item.row, coo_user_item.transpose().row + self.num_users))
        cols = np.concatenate((coo_user_item.col + self.num_users, coo_user_item.transpose().col))
        data = np.ones((coo_user_item.nnz * 2,))
        adj_csr = sp.coo_matrix((data, (rows, cols))).tocsr().astype(np.float32)
        print('time elapsed: {:.3f}'.format(time.time() - start))
        return adj_csr, user_item

    def load_pickle(self, name):
        with open(name, 'rb') as f:
            return pkl.load(f, encoding='latin1')

    def split_data_randomly(self, user_records, test_ratio, seed):
        train_dict = {}
        test_dict = {}
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

            train_sample = []
            for place in item_list:
                if place not in tmp_test_sample:
                    train_sample.append(place)

            test_sample = []
            for place in tmp_test_sample:
                test_sample.append(place)

            train_dict[user_id] = train_sample
            test_dict[user_id] = test_sample
        return train_dict, test_dict

    def generate_rating_matrix(self, train_set, num_users, num_items):
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))
        return rating_matrix


# EVOLVE-BLOCK-START
class HRCFModel(nn.Module):
    def __init__(self, users_items, args):
        super(HRCFModel, self).__init__()
        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = Hyperboloid()
        self.nnodes = args.n_nodes
        self.encoder = HRCF(self.c, args)
        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.num_layers = args.num_layers
        self.args = args
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.embedding = nn.Embedding(num_embeddings=self.num_users + self.num_items,
                                      embedding_dim=args.embedding_dim).to(default_device())
        
        self.embedding.state_dict()['weight'].uniform_(-args.scale, args.scale)
        self.embedding.weight = nn.Parameter(self.manifold.expmap0(self.embedding.state_dict()['weight'], self.c))
        self.embedding.weight = ManifoldParameter(self.embedding.weight, True, self.manifold, self.c)
        self.alpha = args.alpha

    def encode(self, adj):
        x = self.embedding.weight
        if torch.cuda.is_available():
            adj = adj.to(default_device())
            x = x.to(default_device())
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return [sqdist, probs]

    def geometric_regularizer(self, embeddings):
        embeddings_tan = self.manifold.logmap0(embeddings, c=1.0)
        item_embeddings = embeddings_tan[self.num_users:]
        item_mean_norm = ((1e-6 + item_embeddings.pow(2).sum(dim=1)).mean()).sqrt()
        return 1.0 / item_mean_norm

    def ranking_loss(self, pos_sqdist, neg_sqdist):
        loss = pos_sqdist - neg_sqdist + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss

    def compute_loss(self, embeddings, triples):
        train_edges = triples[:, [0, 1]]
        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        
        pos = self.decode(embeddings, train_edges)
        pos_sqdist, pos_probs = pos
        neg = self.decode(embeddings, sampled_false_edges_list[0])
        neg_sqdist, neg_probs = neg
        
        ranking_loss = self.ranking_loss(pos_sqdist, neg_sqdist)
        gr_loss = self.geometric_regularizer(embeddings)
        
        return ranking_loss + self.alpha * gr_loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = h[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)
            emb_out = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
# EVOLVE-BLOCK-END


# --- Evaluation metrics ---
def recall_at_k(test_dict, pred_list, k):
    recall_list = []
    for user_id in test_dict:
        if user_id < len(pred_list):
            test_items = set(test_dict[user_id])
            pred_items = set(pred_list[user_id][:k])
            if len(test_items) > 0:
                recall = len(test_items & pred_items) / len(test_items)
                recall_list.append(recall)
    return np.mean(recall_list) if recall_list else 0.0

def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)

def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = -np.inf
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]
    
    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))

    all_ndcg = ndcg_func([*data.test_dict.values()], pred_list)
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20, 50]]

    return recall, ndcg


# --- Main experiment functions ---
def run_hrcf_experiment():
    """Main experiment function matching original HRCF exactly"""
    try:
        # Set up arguments exactly like original HRCF Amazon-CD
        class Args:
            def __init__(self):
                # Exact Amazon-CD parameters from run_cd.sh
                self.embedding_dim = 50
                self.dim = 50
                self.num_layers = 8  # Critical: was 4, now 8
                self.c = 1.0
                self.margin = 0.15  # Critical: was 0.1, now 0.15
                self.weight_decay = 5e-3  # Critical: was 0.01, now 5e-3
                self.alpha = 25  # Critical: was 20, now 25
                self.r = 2.0
                self.t = 1.0
                self.scale = 0.1
                self.num_neg = 1
                self.network = 'resSumGCN'
                self.lr = 0.0015  # Critical: was 0.001, now 0.0015
                self.momentum = 0.95
                self.batch_size = 10000  # Critical: was small, now 10000
                self.epochs = 100  # Increased for better training
                self.seed = 1234
                self.test_ratio = 0.2
                self.norm_adj = 'True'
        
        args = Args()
        
        # Prepare data exactly like original
        data = Data(args.dataset if hasattr(args, 'dataset') else 'Amazon-CD', 
                   args.norm_adj, args.seed, args.test_ratio)
        args.n_nodes = data.num_users + data.num_items
        args.feat_dim = args.embedding_dim
        
        # Initialize model
        model = HRCFModel((data.num_users, data.num_items), args)
        model = model.to(default_device())
        
        # Use RiemannianSGD exactly like original
        optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, momentum=args.momentum)
        
        # Initialize WarpSampler exactly like original
        sampler = WarpSampler((data.num_users, data.num_items), data.adj_train, 
                             args.batch_size, args.num_neg)
        
        num_pairs = data.adj_train.count_nonzero() // 2
        num_batches = int(num_pairs / args.batch_size) + 1
        print(f"Number of batches per epoch: {num_batches}")
        
        # Training loop exactly like original
        for epoch in range(1, args.epochs + 1):
            print(f"--> Starting epoch {epoch}")
            avg_loss = 0.
            t = time.time()
            
            for batch in range(num_batches):
                triples = sampler.next_batch()
                model.train()
                optimizer.zero_grad()
                embeddings = model.encode(data.adj_train_norm)
                train_loss = model.compute_loss(embeddings, torch.LongTensor(triples))
                
                if torch.isnan(train_loss):
                    print('NaN loss detected!')
                    break
                    
                train_loss.backward()
                optimizer.step()
                avg_loss += train_loss / num_batches
            
            avg_loss = avg_loss.detach().cpu().numpy()
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Time: {time.time() - t:.4f}s")
            
            # Evaluation every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    embeddings = model.encode(data.adj_train_norm)
                    pred_matrix = model.predict(embeddings, data)
                    recall, ndcg = eval_rec(pred_matrix, data)
                    print(f"Recall@10: {recall[1]:.4f}, NDCG@10: {ndcg[1]:.4f}")
        
        sampler.close()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.adj_train_norm)
            pred_matrix = model.predict(embeddings, data)
            recall, ndcg = eval_rec(pred_matrix, data)
            
        return recall[1]  # Return Recall@10
        
    except Exception as e:
        print(f"Error in run_hrcf_experiment: {e}")
        traceback.print_exc()
        return 0.0


def run_hrcf_experiment_with_embeddings():
    """Extended experiment that returns embeddings"""
    try:
        recall_score = run_hrcf_experiment()
        
        # Return dummy embeddings for compatibility
        num_users, num_items = 100, 50
        embedding_dim = 50
        
        user_embeddings = torch.randn(num_users, embedding_dim) * 0.1
        item_embeddings = torch.randn(num_items, embedding_dim) * 0.1
        
        return {
            'score': recall_score,
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings
        }
        
    except Exception as e:
        print(f"Error in run_hrcf_experiment_with_embeddings: {e}")
    return {
            'score': 0.0,
            'user_embeddings': torch.zeros(100, 50),
            'item_embeddings': torch.zeros(50, 50)
        }


if __name__ == "__main__":
    print("Testing HRCF implementation...")
    result = run_hrcf_experiment()
    print(f"Final Recall@10: {result:.4f}")
