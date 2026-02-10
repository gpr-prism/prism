import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def get_rank(target_score, candidate_score):
    tmp_list = target_score - candidate_score
    rank = len(tmp_list[tmp_list < 0]) + 1
    return rank


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
    get metrics for the link prediction task
    :param pos_scores: Tensor, shape (num_samples, )
    :param neg_scores: Tensor, shape (num_samples, neg_size)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    try:
        pos_scores = pos_scores.cpu().detach().numpy()
    except Exception:
        pass
    try:
        neg_scores = neg_scores.cpu().detach().numpy()
    except Exception:
        pass

    pos_scores = np.nan_to_num(pos_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    neg_scores = np.nan_to_num(neg_scores, nan=0.0, posinf=1e6, neginf=-1e6)

    hit_ks = [1, 3, 10, 20, 50, 100, 200, 500, 1000, 2000]
    ndcg_ks = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000]

    metrics = {f'H{k}': [] for k in hit_ks}
    metrics.update({f'NDCG@{k}': [] for k in ndcg_ks})
    metrics.update({'MRR': [], 'AUC': []})

    for i in range(len(pos_scores)):
        rank = get_rank(pos_scores[i], neg_scores[i])
        
        # Hit Rate
        for k in hit_ks:
            metrics[f'H{k}'].append(1 if rank <= k else 0)
        
        # MRR
        metrics['MRR'].append(1.0 / rank)
        
        # AUC
        auc_val = calculate_auc(pos_scores[i], neg_scores[i])
        metrics['AUC'].append(auc_val)
        
        # NDCG@k for different cut-off values
        for k in ndcg_ks:
            metrics[f'NDCG@{k}'].append(calculate_ndcg_at_k(rank, k=k))
    
    # Compute mean value
    result = {key: np.mean(values) for key, values in metrics.items()}
    
    # Add sample count info
    result['num_samples'] = len(pos_scores)
    result['neg_size'] = neg_scores.shape[1] if len(neg_scores.shape) > 1 else 1
    
    return result

def calculate_auc(pos_score, neg_scores):
    """
    Compute AUC (Area Under Curve)
    
    Args:
        pos_score: positive sample score
        neg_scores: negative sample score array
    
    Returns:
        AUC value
    """
    # Count how many times positive score > negative scores
    correct_pairs = np.sum(pos_score > neg_scores)

    # Count ties where positive score == negative score (tie counts as 0.5)
    tie_pairs = np.sum(pos_score == neg_scores)

    # Total number of negative samples
    total_neg = len(neg_scores)

    # Compute AUC
    auc = (correct_pairs + 0.5 * tie_pairs) / total_neg if total_neg > 0 else 0.0

    return auc

def calculate_ndcg_at_k(rank, k=10):
    """
    Compute NDCG@k
    
    Args:
        rank: rank of the positive sample (1-based)
        k: cutoff position
    
    Returns:
        NDCG@k value
    """
    if rank > k:
        return 0.0
    
    # Compute DCG
    # In binary relevance, there is only one relevant document at rank=rank
    dcg = 1.0 / np.log2(rank + 1)  # rank is 1-based, so +1
    
    # Compute IDCG (best possible ranking)
    idcg = 1.0 / np.log2(1 + 1)  # best case: relevant item at rank 1
    
    return dcg / idcg


def calculate_ndcg(rank):
    """
    Keep the original NDCG computation (global NDCG)
    
    Args:
        rank: rank of the positive sample
    
    Returns:
        NDCG value
    """
    # Global NDCG computation
    dcg = 1.0 / np.log2(rank + 1)
    idcg = 1.0 / np.log2(2)  # DCG at the best rank
    return dcg / idcg


def get_retrival_metrics_graph(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
    get metrics for the link prediction task
    :param pos_scores: Tensor, shape (num_samples, )
    :param neg_scores: Tensor, shape (neg_size, num_samples)
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    try:
        pos_scores = pos_scores.cpu().detach().numpy()
    except:
        pass
    try:
        neg_scores = np.array([sub_score.cpu().numpy() for sub_score in neg_scores]).T # num_samples * neg_size
    except:
        neg_scores = np.array([sub_score for sub_score in neg_scores]).T # num_samples * neg_size

    H1, H3, H10 = [], [], []
    for i in range(len(pos_scores)):
        rank = get_rank(pos_scores[i], neg_scores[i])
        if rank <= 1:
            H1.append(1)
        else:
            H1.append(0)
        
        if rank <= 3:
            H3.append(1)
        else:
            H3.append(0)

        if rank <= 10:
            H10.append(1)
        else:
            H10.append(0)

    return {'H1': np.mean(H1), 'H3': np.mean(H3), 'H10': np.mean(H10)}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}

def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the edge classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    P_macro = precision_score(labels, predicts, average="macro")
    R_macro = recall_score(labels, predicts, average="macro")
    F_macro = f1_score(labels, predicts, average="macro")

    P_micro = precision_score(labels, predicts, average="micro")
    R_micro = recall_score(labels, predicts, average="micro")
    F_micro = f1_score(labels, predicts, average="micro")

    P_weight = precision_score(labels, predicts, average="weighted")
    R_weight = recall_score(labels, predicts, average="weighted")
    F_weight = f1_score(labels, predicts, average="weighted")

    return {'p_macro': P_macro, 'R_macro': R_macro, 'F_macro': F_macro, 'p_micro': P_micro, 'R_micro': R_micro, 'F_micro': F_micro, 'p_weight': P_weight, 'R_weight': R_weight, 'F_weight': F_weight}
