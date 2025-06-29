"""
Evaluator for HRCF recommendation system
Evaluates the evolved algorithms and returns performance metrics
"""

import importlib.util
import numpy as np
import time
import traceback
import tempfile
import os
from typing import Dict, Any
import sys
from types import ModuleType
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import train_test_split
import argparse
import pickle as pkl

# Import EvaluationResult for artifacts support
from openevolve.evaluation_result import EvaluationResult


# ---- Evaluation Metrics (integrated from eval_metrics.py) ----

def recall_at_k_per_user(actual, pred, k):
    """Compute recall@k for a single user.

    Args:
        actual (Iterable[int]): Ground-truth item IDs the user interacted with.
        pred (Iterable[int]):  Predicted item IDs sorted by relevance.
        k (int):               Cut-off rank.
    Returns
        float: Recall@k value for this user.
    """
    act_set = set(actual)
    if len(act_set) == 0:
        return 0.0
    return len(act_set & set(pred[:k])) / float(len(act_set))


def recall_at_k(actual, predicted, topk):
    """Macro-averaged recall@k over all users.

    Args:
        actual (Dict[int, List[int]]):    user_id -> list of ground-truth item IDs.
        predicted (Dict[int, List[int]] or np.ndarray): user_id -> ranked item IDs or matrix.
        topk (int):                       Cut-off rank k.
    Returns
        float: Averaged recall@k across users.
    """
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    
    for i, v in actual.items():
        act_set = set(v)
        if len(act_set) == 0:
            continue
        
        # Handle both dict and array formats for predicted
        if isinstance(predicted, dict):
            pred_items = predicted[i][:topk]
        else:
            pred_items = predicted[i][:topk]
        
        # Convert to Python int to avoid numpy type issues
        # Handle nested lists that might come from predictions
        flat_pred_items = []
        for item in pred_items:
            if isinstance(item, (list, np.ndarray)):
                flat_pred_items.extend(item)
            else:
                flat_pred_items.append(item)
        pred_set = set(int(item) for item in flat_pred_items)
            
        sum_recall += len(act_set & pred_set) / float(len(act_set))
        true_users += 1
    
    assert num_users == true_users, f"Mismatch: {num_users} users expected, {true_users} processed"
    return sum_recall / true_users


def precision_at_k(actual, predicted, topk):
    """Macro-averaged precision@k over all users."""
    sum_precision = 0.0
    num_users = len(actual)
    true_users = 0
    
    for i, v in actual.items():
        act_set = set(v)
        if len(act_set) == 0:
            continue
            
        if isinstance(predicted, dict):
            pred_list = predicted[i][:topk]
        else:
            pred_list = predicted[i][:topk]
            
        # Convert to Python int to avoid numpy type issues
        # Handle nested lists that might come from predictions
        flat_pred_list = []
        for item in pred_list:
            if isinstance(item, (list, np.ndarray)):
                flat_pred_list.extend(item)
            else:
                flat_pred_list.append(item)
        pred_set = set(int(item) for item in flat_pred_list)
        if len(pred_list) > 0:
            sum_precision += len(act_set & pred_set) / float(len(pred_list))
        true_users += 1
    
    return sum_precision / true_users if true_users > 0 else 0.0


def ndcg_at_k(actual, predicted, topk):
    """Normalized Discounted Cumulative Gain at k."""
    sum_ndcg = 0.0
    num_users = len(actual)
    true_users = 0
    
    for i, v in actual.items():
        act_set = set(v)
        if len(act_set) == 0:
            continue
            
        if isinstance(predicted, dict):
            pred_list = predicted[i][:topk]
        else:
            pred_list = predicted[i][:topk]
        
        # Convert to Python int to avoid numpy type issues
        # Handle nested lists that might come from predictions
        flat_pred_list = []
        for item in pred_list:
            if isinstance(item, (list, np.ndarray)):
                flat_pred_list.extend(item)
            else:
                flat_pred_list.append(item)
        pred_list = [int(item) for item in flat_pred_list]
        
        # DCG
        dcg = 0.0
        for j, item in enumerate(pred_list):
            if item in act_set:
                dcg += 1.0 / np.log2(j + 2)  # j+2 because log2(1) = 0
        
        # IDCG (ideal DCG)
        idcg = 0.0
        for j in range(min(len(act_set), topk)):
            idcg += 1.0 / np.log2(j + 2)
        
        if idcg > 0:
            sum_ndcg += dcg / idcg
        true_users += 1
    
    return sum_ndcg / true_users if true_users > 0 else 0.0


def hit_rate_at_k(actual, predicted, topk):
    """Hit rate at k - fraction of users with at least one relevant item in top-k."""
    hits = 0
    num_users = len(actual)
    true_users = 0
    
    for i, v in actual.items():
        act_set = set(v)
        if len(act_set) == 0:
            continue
            
        if isinstance(predicted, dict):
            pred_items = predicted[i][:topk]
        else:
            pred_items = predicted[i][:topk]
        
        # Convert to Python int to avoid numpy type issues
        # Handle nested lists that might come from predictions
        flat_pred_items = []
        for item in pred_items:
            if isinstance(item, (list, np.ndarray)):
                flat_pred_items.extend(item)
            else:
                flat_pred_items.append(item)
        pred_set = set(int(item) for item in flat_pred_items)
            
        if len(act_set & pred_set) > 0:
            hits += 1
        true_users += 1
    
    return hits / true_users if true_users > 0 else 0.0


# ---- Real Data Loading and Processing ----

class RealDataset:
    """Real dataset class for HRCF evaluation using actual data files."""
    
    def __init__(self, dataset_name='Amazon-CD', test_ratio=0.2, seed=42, subset_ratio=None):
        self.dataset_name = dataset_name
        self.test_ratio = test_ratio
        self.seed = seed
        self.subset_ratio = subset_ratio  # Use only a subset of data if specified
        
        # Load data from the appropriate directory
        self.data_path = os.path.join(os.path.dirname(__file__), 'data', dataset_name)
        
        if not os.path.exists(self.data_path):
            raise ValueError(f"Dataset path {self.data_path} does not exist")
        
        print(f"Loading {dataset_name} dataset from {self.data_path}")
        if subset_ratio:
            print(f"Using {subset_ratio*100:.1f}% subset of the data for testing")
        
        # Load the data
        self._load_data()
        
        print(f"Dataset loaded: {self.num_users} users, {self.num_items} items")
        print(f"Training interactions: {len(self.train_data)}")
        print(f"Test users: {len(self.test_data)}")
        
        # Create adjacency matrix
        self.adj_matrix = self._create_adjacency_matrix()
    
    def _load_data(self):
        """Load data based on dataset structure."""
        user_item_path = os.path.join(self.data_path, 'user_item_list.pkl')
        
        if not os.path.exists(user_item_path):
            raise ValueError(f"Required file {user_item_path} not found")
        
        # Load user-item interaction list
        with open(user_item_path, 'rb') as f:
            user_item_list = pkl.load(f, encoding='latin1')
        
        # Apply subset if specified
        if self.subset_ratio and self.subset_ratio < 1.0:
            np.random.seed(self.seed)
            num_users_subset = int(len(user_item_list) * self.subset_ratio)
            selected_users = np.random.choice(len(user_item_list), num_users_subset, replace=False)
            selected_users = sorted(selected_users)
            
            # Create subset
            self.user_item_list = [user_item_list[i] for i in selected_users]
            
            # Remap item IDs to be continuous
            all_items = set()
            for items in self.user_item_list:
                all_items.update(items)
            
            item_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(all_items))}
            
            # Apply item mapping
            for i, items in enumerate(self.user_item_list):
                self.user_item_list[i] = [item_mapping[item] for item in items]
            
            print(f"Subset created: {len(self.user_item_list)} users, {len(item_mapping)} items")
        else:
            self.user_item_list = user_item_list
        
        # Check if pre-split train/test files exist (only for full dataset)
        train_path = os.path.join(self.data_path, 'train.pkl')
        test_path = os.path.join(self.data_path, 'test.pkl')
        
        if os.path.exists(train_path) and os.path.exists(test_path) and not self.subset_ratio:
            print("Loading pre-split train/test data...")
            with open(train_path, 'rb') as f:
                self.train_dict = pkl.load(f, encoding='latin1')
            with open(test_path, 'rb') as f:
                self.test_dict = pkl.load(f, encoding='latin1')
        else:
            print("Splitting data randomly...")
            self.train_dict, self.test_dict = self._split_data_randomly()
        
        # Convert to the format expected by evaluator
        self.train_data = []
        for user_id, item_list in self.train_dict.items():
            for item_id in item_list:
                self.train_data.append((user_id, item_id))
        
        self.test_data = self.test_dict
        
        # Calculate dataset statistics
        self.num_users = len(self.user_item_list)
        self.num_items = max([max(items) for items in self.user_item_list if len(items) > 0]) + 1
        
        print(f"Data statistics:")
        print(f"  Total users: {self.num_users}")
        print(f"  Total items: {self.num_items}")
        print(f"  Total interactions: {sum(len(items) for items in self.user_item_list)}")
        print(f"  Density: {sum(len(items) for items in self.user_item_list) / (self.num_users * self.num_items):.6f}")
    
    def _split_data_randomly(self):
        """Split data into train/test sets."""
        train_dict = {}
        test_dict = {}
        
        np.random.seed(self.seed)
        
        for user_id, item_list in enumerate(self.user_item_list):
            if len(item_list) == 0:
                train_dict[user_id] = []
                test_dict[user_id] = []
                continue
                
            # Split items for this user
            tmp_train_sample, tmp_test_sample = train_test_split(
                item_list, test_size=self.test_ratio, random_state=self.seed + user_id
            )
            
            train_dict[user_id] = tmp_train_sample
            test_dict[user_id] = tmp_test_sample
        
        return train_dict, test_dict
    
    def _create_adjacency_matrix(self):
        """Create adjacency matrix from training data."""
        print("Creating adjacency matrix...")
        
        # Check if pre-computed adjacency matrix exists
        adj_path = os.path.join(self.data_path, 'adj_csr.npz')
        if os.path.exists(adj_path):
            print(f"Loading pre-computed adjacency matrix from {adj_path}")
            adj_csr = sp.load_npz(adj_path)
            return adj_csr.tocoo()
        
        # Create user-item bipartite graph
        n_nodes = self.num_users + self.num_items
        adj = lil_matrix((n_nodes, n_nodes))
        
        for user_id, item_list in self.train_dict.items():
            for item_id in item_list:
                # User to item edge
                adj[user_id, self.num_users + item_id] = 1.0
                # Item to user edge (symmetric)
                adj[self.num_users + item_id, user_id] = 1.0
        
        print("Normalizing adjacency matrix...")
        # Convert to CSR for efficient operations
        adj = adj.tocsr()
        
        # Degree normalization
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Save the adjacency matrix for future use
        print(f"Saving adjacency matrix to {adj_path}")
        sp.save_npz(adj_path, normalized_adj.tocsr())
        
        return normalized_adj.tocoo()

    def generate_rating_matrix(self):
        """Generate rating matrix from training data (like original HRCF)"""
        row = []
        col = []
        data = []
        for user_id, item_list in self.train_dict.items():
            for item in item_list:
                row.append(user_id)
                col.append(item)
                data.append(1)
        
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = sp.csr_matrix((data, (row, col)), shape=(self.num_users, self.num_items))
        return rating_matrix


def load_real_dataset(dataset_name='Amazon-CD'):
    """Load real dataset from data directory."""
    try:
        return RealDataset(dataset_name=dataset_name)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        print("Falling back to synthetic data...")
        return create_synthetic_dataset(num_users=1000, num_items=500, num_interactions=10000)


# ---- Legacy functions for backward compatibility ----

class SimpleDataset:
    """Simple dataset class for HRCF evaluation."""
    
    def __init__(self, num_users, num_items, train_data, test_data):
        self.num_users = num_users
        self.num_items = num_items
        self.train_data = train_data  # List of (user, item) pairs
        self.test_data = test_data    # Dict {user_id: [item_ids]}
        
        # Create adjacency matrix
        self.adj_matrix = self._create_adjacency_matrix()
    
    def _create_adjacency_matrix(self):
        """Create adjacency matrix from training data."""
        # Create user-item bipartite graph
        n_nodes = self.num_users + self.num_items
        adj = lil_matrix((n_nodes, n_nodes))
        
        for user, item in self.train_data:
            # User to item edge
            adj[user, self.num_users + item] = 1.0
            # Item to user edge (symmetric)
            adj[self.num_users + item, user] = 1.0
        
        # Add self-loops
        for i in range(n_nodes):
            adj[i, i] = 1.0
        
        # Normalize adjacency matrix
        adj = adj.tocsr()
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return normalized_adj.tocoo()


def create_synthetic_dataset(num_users=1000, num_items=500, num_interactions=10000):
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    
    # Generate random user-item interactions
    users = np.random.randint(0, num_users, num_interactions)
    items = np.random.randint(0, num_items, num_interactions)
    
    # Remove duplicates
    interactions = list(set(zip(users, items)))
    
    # Split into train/test
    train_data, test_data_list = train_test_split(interactions, test_size=0.2, random_state=42)
    
    # Convert test data to dict format
    test_data = {}
    for user, item in test_data_list:
        if user not in test_data:
            test_data[user] = []
        test_data[user].append(item)
    
    return SimpleDataset(num_users, num_items, train_data, test_data)


def load_amazon_book_data():
    """Load Amazon Book dataset if available."""
    return load_real_dataset('Amazon-Book')


# ---- Main Evaluation Function ----

def evaluate(program_path: str, dataset_name: str = 'Amazon-CD') -> dict:
    """
    Evaluate a HRCF program by running it and measuring recommendation performance.
    
    Args:
        program_path: Path to the Python program file to evaluate
        dataset_name: Name of the dataset to use for evaluation
        
    Returns:
        Dictionary with recommendation metrics and other performance indicators
    """
    
    try:
        # Read the program code from file
        with open(program_path, 'r') as f:
            program_code = f.read()
        
        # Create a temporary file with the program code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(program_code)
            temp_file = f.name
        
        # Add the current directory to sys.path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Load the module from the temporary file
        spec = importlib.util.spec_from_file_location("evolved_program", temp_file)
        evolved_program = importlib.util.module_from_spec(spec)
        
        # Execute the module
        spec.loader.exec_module(evolved_program)
        
        # Clean up the temporary file
        os.unlink(temp_file)
        
        # Check if the required components exist
        if not hasattr(evolved_program, 'HRCFModel'):
            return {
                'recall_at_10': 0.0,
                'recall_at_20': 0.0,
                'precision_at_10': 0.0,
                'precision_at_20': 0.0,
                'ndcg_at_10': 0.0,
                'ndcg_at_20': 0.0,
                'hit_rate_at_10': 0.0,
                'hit_rate_at_20': 0.0,
                'execution_time': 0.0,
                'combined_score': 0.0,
                'error': 'HRCFModel class not found in program'
            }
        
        # Run the complete HRCF experiment
        print("Starting HRCF recommendation system evaluation...")
        start_time = time.time()
        
        try:
            # Load dataset
            data = load_real_dataset(dataset_name)
            
            # Create arguments object with dataset-specific parameters (matching original HRCF)
            args = argparse.Namespace(
                embedding_dim=50,        # Original HRCF uses 50
                num_layers=4,           # Original HRCF uses 4 layers
                c=1.0,
                margin=0.1,
                weight_decay=0.005,     # Original HRCF uses 0.005
                alpha=20,               # Original HRCF uses 20
                r=2.0,
                t=1.0,
                scale=0.1,
                num_neg=1,
                network='resSumGCN',
                n_nodes=data.num_users + data.num_items,
                lr=0.001,               # Original HRCF uses 0.001
                momentum=0.95,          # Add momentum for RiemannianSGD
                batch_size=10000,       # Original HRCF uses 10000
                epochs=100,  # Full training epochs for production
                dataset=dataset_name
            )
            
            # Initialize model
            model = evolved_program.HRCFModel((data.num_users, data.num_items), args)
            
            # Convert adjacency matrix to sparse tensor
            adj_coo = data.adj_matrix
            indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
            values = torch.FloatTensor(adj_coo.data)
            shape = adj_coo.shape
            adj_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
            
            # Training loop using RiemannianSGD and WarpSampler (like original HRCF)
            from rgd.rsgd import RiemannianSGD
            from utils.sampler import WarpSampler, normalize, sparse_mx_to_torch_sparse_tensor
            
            optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay, momentum=args.momentum)
            
            # Create WarpSampler for proper negative sampling
            sampler = WarpSampler((data.num_users, data.num_items), data.adj_matrix, 
                                args.batch_size, args.num_neg)
            
            # Properly normalize adjacency matrix (like original HRCF)
            adj_normalized = normalize(data.adj_matrix + sp.eye(data.adj_matrix.shape[0]))
            adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized)
            
            model.train()
            # Training configuration (matching original HRCF)
            max_epochs = args.epochs
            num_pairs = data.adj_matrix.count_nonzero() // 2
            num_batches = int(num_pairs / args.batch_size) + 1
            
            print(f"Training with {max_epochs} epochs, batch size {args.batch_size}")
            print(f"Dataset size: {data.num_users} users, {data.num_items} items")
            print(f"Number of batches per epoch: {num_batches}")
            
            for epoch in range(1, max_epochs + 1):
                avg_loss = 0.
                
                # Process all batches in epoch (like original HRCF)
                for batch in range(num_batches):
                    triples = sampler.next_batch()
                    triples = torch.LongTensor(triples)
                    
                    model.train()
                    optimizer.zero_grad()
                    embeddings = model.encode(adj_tensor)
                    train_loss = model.compute_loss(embeddings, triples)
                    
                    # Check for NaN (like original HRCF)
                    if torch.isnan(train_loss):
                        print("NaN loss detected, stopping training")
                        break
                        
                    train_loss.backward()
                    optimizer.step()
                    avg_loss += train_loss / num_batches
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss.item():.4f}")
                elif epoch % 10 == 0 and epoch < 100:
                    print(f"Epoch {epoch}, Loss: {avg_loss.item():.4f}")
            
            sampler.close()
            
            # Evaluation (matching original HRCF approach)
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(adj_tensor)
                
                # Generate prediction matrix like original HRCF
                print(f"Evaluating {data.num_users} users")
                pred_matrix = model.predict(embeddings, data)
                
                # Exclude training items from predictions (critical!)
                # This is the key difference from our previous implementation
                user_item_csr = data.generate_rating_matrix()
                pred_matrix[user_item_csr.nonzero()] = -np.inf
                
                # Get top-k predictions for each user (like original HRCF)
                topk = 50
                ind = np.argpartition(pred_matrix, -topk)
                ind = ind[:, -topk:]
                arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
                pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]
                
                # Convert to predictions dict format
                predictions = {}
                for user_id in range(data.num_users):
                    if user_id in data.test_data and len(data.test_data[user_id]) > 0:
                        predictions[user_id] = pred_list[user_id].tolist()
            
                execution_time = time.time() - start_time
                
            # Compute evaluation metrics
            recall_10 = recall_at_k(data.test_data, predictions, 10)
            recall_20 = recall_at_k(data.test_data, predictions, 20)
            precision_10 = precision_at_k(data.test_data, predictions, 10)
            precision_20 = precision_at_k(data.test_data, predictions, 20)
            ndcg_10 = ndcg_at_k(data.test_data, predictions, 10)
            ndcg_20 = ndcg_at_k(data.test_data, predictions, 20)
            hit_rate_10 = hit_rate_at_k(data.test_data, predictions, 10)
            hit_rate_20 = hit_rate_at_k(data.test_data, predictions, 20)
            
            # Combined score emphasizing recall and NDCG
            combined_score = (
                0.3 * recall_10 +
                0.2 * recall_20 +
                0.2 * ndcg_10 +
                0.15 * ndcg_20 +
                0.1 * precision_10 +
                0.05 * hit_rate_10
            )
            
            print(f"HRCF Evaluation completed:")
            print(f"  Recall@10: {recall_10:.4f}")
            print(f"  Recall@20: {recall_20:.4f}")
            print(f"  Precision@10: {precision_10:.4f}")
            print(f"  Precision@20: {precision_20:.4f}")
            print(f"  NDCG@10: {ndcg_10:.4f}")
            print(f"  NDCG@20: {ndcg_20:.4f}")
            print(f"  Hit Rate@10: {hit_rate_10:.4f}")
            print(f"  Hit Rate@20: {hit_rate_20:.4f}")
            print(f"  Execution Time: {execution_time:.2f}s")
            print(f"  Combined Score: {combined_score:.4f}")
            
            # Prepare success artifacts
            artifacts = {
                "success_summary": f"Successful HRCF training with {max_epochs} epochs",
                "model_architecture": f"Embedding dim: {args.embedding_dim}, Layers: {args.num_layers}, Network: {args.network}",
                "training_details": f"Batch size: {args.batch_size}, LR: {args.lr}, Weight decay: {args.weight_decay}",
                "dataset_info": f"{data.num_users} users, {data.num_items} items, {data.adj_matrix.count_nonzero()} interactions",
                "performance_analysis": f"Combined score: {combined_score:.4f} (Target: >0.10 for good performance)",
                "metric_breakdown": f"Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}, Precision@10: {precision_10:.4f}",
                "execution_efficiency": f"Training time: {execution_time:.1f}s ({execution_time/max_epochs:.1f}s per epoch)",
                "recommendation_quality": "Excellent" if combined_score > 0.08 else "Good" if combined_score > 0.05 else "Moderate" if combined_score > 0.02 else "Poor"
            }
            
            # Add embedding quality analysis if available
            try:
                embedding_quality = analyze_embedding_quality(embeddings)
                artifacts["embedding_analysis"] = f"Effective rank: {embedding_quality['effective_rank']:.1f}/{embedding_quality['embedding_dimension']} ({embedding_quality['effective_rank_normalized']:.2f})"
            except:
                artifacts["embedding_analysis"] = "Embedding analysis unavailable"
            
            return EvaluationResult(
                metrics={
                    'recall_at_10': float(recall_10),
                    'recall_at_20': float(recall_20),
                    'precision_at_10': float(precision_10),
                    'precision_at_20': float(precision_20),
                    'ndcg_at_10': float(ndcg_10),
                    'ndcg_at_20': float(ndcg_20),
                    'hit_rate_at_10': float(hit_rate_10),
                    'hit_rate_at_20': float(hit_rate_20),
                    'execution_time': float(execution_time),
                    'combined_score': float(combined_score),
                    'error': None
                },
                artifacts=artifacts
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Training/evaluation error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Prepare detailed error artifacts
            artifacts = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "failure_stage": "training_or_evaluation",
                "execution_time": f"{execution_time:.2f}s",
                "full_traceback": traceback.format_exc()
            }
            
            # Add specific error analysis
            if "AttributeError" in str(e):
                if "'float' object has no attribute" in str(e):
                    artifacts["error_analysis"] = "CRITICAL: Model initialization failed - likely tensor/parameter type mismatch. Check model constructor and parameter initialization."
                    artifacts["suggested_fix"] = "Ensure all model parameters are properly initialized as tensors, not scalars. Check HRCFModel.__init__ method."
                else:
                    artifacts["error_analysis"] = "Attribute error suggests missing or incorrectly named model components."
                    artifacts["suggested_fix"] = "Verify all required methods and attributes exist in the evolved model."
            elif "RuntimeError" in str(e) and "device" in str(e):
                artifacts["error_analysis"] = "CRITICAL: Device mismatch - tensors on different devices (CPU vs GPU)."
                artifacts["suggested_fix"] = "Ensure all tensors are moved to the same device using .to(device) or .cuda()."
            elif "CUDA out of memory" in str(e):
                artifacts["error_analysis"] = "GPU memory exhausted during training."
                artifacts["suggested_fix"] = "Reduce batch size, embedding dimensions, or model complexity."
            elif "NaN" in str(e) or "nan" in str(e):
                artifacts["error_analysis"] = "Training diverged - NaN values detected."
                artifacts["suggested_fix"] = "Reduce learning rate, add gradient clipping, or check loss function implementation."
            else:
                artifacts["error_analysis"] = f"Unexpected error during training: {str(e)[:200]}"
                artifacts["suggested_fix"] = "Review the error message and check model implementation."
            
            return EvaluationResult(
                metrics={
                    'recall_at_10': 0.0,
                    'recall_at_20': 0.0,
                    'precision_at_10': 0.0,
                    'precision_at_20': 0.0,
                    'ndcg_at_10': 0.0,
                    'ndcg_at_20': 0.0,
                    'hit_rate_at_10': 0.0,
                    'hit_rate_at_20': 0.0,
                    'execution_time': float(execution_time),
                    'combined_score': 0.0,
                    'error': error_msg
                },
                artifacts=artifacts
            )
        
    except Exception as e:
        error_msg = f"Program loading failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Prepare detailed loading error artifacts
        artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "failure_stage": "program_loading",
            "full_traceback": traceback.format_exc()
        }
        
        # Add specific loading error analysis
        if "SyntaxError" in str(e):
            artifacts["error_analysis"] = "CRITICAL: Syntax error in evolved code - invalid Python syntax."
            artifacts["suggested_fix"] = "Fix syntax errors: check for unmatched parentheses, brackets, quotes, or indentation issues."
            if "was never closed" in str(e):
                artifacts["specific_issue"] = "Unmatched parentheses or brackets detected."
        elif "ImportError" in str(e) or "ModuleNotFoundError" in str(e):
            artifacts["error_analysis"] = "Missing import or module dependency."
            artifacts["suggested_fix"] = "Ensure all required imports are included and modules are available."
        elif "NameError" in str(e):
            artifacts["error_analysis"] = "Undefined variable or function name."
            artifacts["suggested_fix"] = "Check for typos in variable/function names or missing definitions."
        elif "IndentationError" in str(e):
            artifacts["error_analysis"] = "Incorrect code indentation."
            artifacts["suggested_fix"] = "Fix indentation to match Python syntax requirements."
        elif "HRCFModel" in str(e):
            artifacts["error_analysis"] = "Missing or incorrectly defined HRCFModel class."
            artifacts["suggested_fix"] = "Ensure HRCFModel class is properly defined with required methods."
        else:
            artifacts["error_analysis"] = f"Program loading failed: {str(e)[:200]}"
            artifacts["suggested_fix"] = "Review the error message and fix the code structure."
        
        return EvaluationResult(
            metrics={
                'recall_at_10': 0.0,
                'recall_at_20': 0.0,
                'precision_at_10': 0.0,
                'precision_at_20': 0.0,
                'ndcg_at_10': 0.0,
                'ndcg_at_20': 0.0,
                'hit_rate_at_10': 0.0,
                'hit_rate_at_20': 0.0,
                'execution_time': 0.0,
                'combined_score': 0.0,
                'error': error_msg
            },
            artifacts=artifacts
        )


def run_with_timeout(func, timeout=120):
    """
    Run a function without timeout for multithreading compatibility
    Note: timeout parameter is kept for API compatibility but ignored
    """
    # Simply run the function without timeout mechanism
    # The timeout will be handled by the OpenEvolve framework itself
    return func()


def evaluate_with_custom_data(program_path: str, custom_data=None) -> Dict[str, Any]:
    """
    Alternative evaluation function that can use custom datasets
    """
    if custom_data is None:
        return evaluate(program_path)
    
    # Implementation for custom data evaluation would go here
    # For now, fall back to standard evaluation
    return evaluate(program_path)


# ---- Embedding Analysis Functions ----

def compute_effective_rank(matrix, threshold=0.99):
    """
    Compute effective rank of a matrix
    
    Args:
        matrix: Input matrix (n, d)
        threshold: Cumulative variance threshold for effective rank
    
    Returns:
        Effective rank (float)
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Compute SVD
    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        
        # Normalize singular values
        s_normalized = s / np.sum(s)
        
        # Find effective rank based on cumulative variance
        cumsum = np.cumsum(s_normalized)
        effective_rank = np.argmax(cumsum >= threshold) + 1
        
        return float(effective_rank)
    
    except Exception as e:
        print(f"Error computing effective rank: {e}")
        return 1.0  # Fallback


def analyze_embedding_quality(embeddings):
    """
    Analyze embedding quality using various metrics
    
    Args:
        embeddings: Embedding matrix (num_entities, embedding_dim)
    
    Returns:
        Dictionary with quality metrics
    """
    try:
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Effective rank
        erank = compute_effective_rank(embeddings_np)
        
        # Embedding dimension
        embed_dim = embeddings_np.shape[1]
        
        # Normalized effective rank
        erank_norm = erank / embed_dim
        
        # Standard deviation (diversity measure)
        std_dev = np.std(embeddings_np, axis=0).mean()
        
        # Norm distribution
        norms = np.linalg.norm(embeddings_np, axis=1)
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        
        return {
            'effective_rank': float(erank),
            'effective_rank_normalized': float(erank_norm),
            'embedding_dimension': int(embed_dim),
            'diversity': float(std_dev),
            'norm_mean': float(norm_mean),
            'norm_std': float(norm_std)
        }
        
    except Exception as e:
        print(f"Error in embedding quality analysis: {e}")
        return {
            'effective_rank': 1.0,
            'effective_rank_normalized': 0.1,
            'embedding_dimension': 32,
            'diversity': 0.0,
            'norm_mean': 1.0,
            'norm_std': 0.0
        }


def evaluate_stage1(program_path):
    """
    Stage 1 evaluation for cascade evaluation support.
    For HRCF, we don't need real cascade - this just calls the main evaluate function.
    This is to work around the _direct_evaluate bug that doesn't support EvaluationResult.
    """
    return evaluate(program_path)


if __name__ == "__main__":
    # Test the evaluator with the initial program
    import sys
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = "initial_program.py"
    
    results = evaluate(program_path)
    print("\nEvaluation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}") 