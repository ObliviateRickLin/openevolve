from multiprocessing import Process, Queue

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import lil_matrix
import itertools


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
    """
    A generator that generates tuples: user-positive-item pairs, negative-items
    Simplified version without multiprocessing for compatibility
    """

    def __init__(self, num_nodes, user_item_matrix, batch_size=10000, n_negative=1):
        self.num_users, self.num_items = num_nodes
        self.batch_size = batch_size
        self.n_negative = n_negative
        
        # Convert to lil_matrix for efficient row access
        self.adj_train = lil_matrix(user_item_matrix)
        
        # Get all positive user-item pairs
        all_pairs = np.asarray(self.adj_train.nonzero()).T
        self.user_item_pairs = all_pairs[: self.adj_train.count_nonzero() // 2]
        
        # Create user-item sets for negative sampling
        self.user_item_sets = {idx: set(row) for idx, row in enumerate(self.adj_train.rows)}
        
        # Shuffle pairs
        np.random.shuffle(self.user_item_pairs)
        self.current_index = 0

    def next_batch(self):
        """Generate next batch of triples (user, pos_item, neg_item)"""
        if self.current_index + self.batch_size >= len(self.user_item_pairs):
            # Reshuffle and restart
            np.random.shuffle(self.user_item_pairs)
            self.current_index = 0
        
        # Get batch of positive pairs
        batch_pairs = self.user_item_pairs[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Generate negative samples
        batch_size_actual = len(batch_pairs)
        triples = []
        
        for user, pos_item in batch_pairs:
            # Sample negative items
            neg_items = []
            for _ in range(self.n_negative):
                neg_item = np.random.randint(self.num_users, self.num_users + self.num_items)
                # Ensure negative item is not in user's positive items
                while neg_item in self.user_item_sets.get(user, set()):
                    neg_item = np.random.randint(self.num_users, self.num_users + self.num_items)
                neg_items.append(neg_item)
            
            # Create triple: [user, pos_item, neg_item1, neg_item2, ...]
            triple = [user, pos_item] + neg_items
            triples.append(triple)
        
        return np.array(triples)

    def close(self):
        """Compatibility method"""
        pass


def normalize(mx):
    """Row-normalize sparse matrix (from original HRCF)"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor (from original HRCF)"""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
