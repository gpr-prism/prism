import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json

from utils.metrics import get_link_prediction_metrics, get_edge_classification_metrics, get_retrival_metrics, get_retrival_metrics_graph
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if True:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively
            if model_name in [
                'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
                'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain'
            ]:
                batch_src_node_embeddings = batch_neg_src_node_embeddings = \
                    model[0].compute_src_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  num_neighbors=num_neighbors)
                batch_dst_node_embeddings = \
                    model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_dst_node_ids)
                batch_neg_dst_node_embeddings = \
                    model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_neg_dst_node_ids)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
            negative_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

            logits = torch.cat([positive_logits, negative_logits], dim=0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)

            loss = loss_func(input=logits, target=labels)
            predicts = torch.sigmoid(logits)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_retrival(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: NegativeEdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    all_item_ids: np.ndarray,  # must be numpy
    num_neighbors: int = 20,
    time_gap: int = 2000,
    batch_size_neg: int = 1000,  # negative chunk size (smaller to reduce memory)
    pair_batch_size: int = 64,   # per-step src sub-batch to avoid B*M OOM
):
    """
    Full retrieval evaluation using pair-wise scoring.
    - Only supports non-graph models (e.g., SASRec, HSTU, etc.)
    - Input IDs to model[0] are numpy arrays
    - Correct per-sample positive masking
    - Efficient chunked negative scoring
    """
    _, unique_indices = np.unique(all_item_ids, return_index=True)
    all_item_ids = all_item_ids[np.sort(unique_indices)]  # dedupe while preserving original order
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    non_graph_models = {
        'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
        'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain',
    }
    if model_name not in non_graph_models:
        raise NotImplementedError(f"Full retrieval only supports non-graph models. Got {model_name}")

    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()

    with torch.no_grad():
        # Precompute all item embeddings (input must be numpy)
        all_item_embeddings = model[0].compute_dst_node_temporal_embeddings(all_item_ids)  # [N, D]
        all_item_embeddings = torch.nan_to_num(all_item_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        device = all_item_embeddings.device

        # Build id -> index map once (for fast lookup of pos indices)
        item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(all_item_ids)}
        all_item_ids_tensor = torch.from_numpy(all_item_ids).to(device)  # for masking

        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)

        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]      # numpy
            batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]      # numpy
            batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]  # numpy
            batch_size = len(batch_src_node_ids)

            # Get src embeddings (input must be numpy)
            src_embeddings = model[0].compute_src_node_temporal_embeddings(
                src_node_ids=batch_src_node_ids,
                node_interact_times=batch_node_interact_times,
                num_neighbors=num_neighbors
            )  # [B, D]
            src_embeddings = torch.nan_to_num(src_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

            # Get positive dst embeddings via index mapping
            pos_indices = np.array([item_id_to_index[int(dst_id)] for dst_id in batch_dst_node_ids])
            pos_dst_embeddings = all_item_embeddings[pos_indices]  # [B, D]
            pos_dst_embeddings = torch.nan_to_num(pos_dst_embeddings, nan=0.0, posinf=1e6, neginf=-1e6)

            # Compute positive scores: pair-wise concat -> model[1]
            #pos_pair_input = torch.cat([src_embeddings, pos_dst_embeddings], dim=-1)  # [B, 2D]
            pos_scores = model[1](src_embeddings, pos_dst_embeddings).squeeze(-1)  # [B]
            pos_scores = torch.nan_to_num(pos_scores, nan=0.0, posinf=1e6, neginf=-1e6)

            # Compute negative scores in chunks
            all_neg_scores_list = []
            for i in range(0, len(all_item_ids), batch_size_neg):
                end_i = min(i + batch_size_neg, len(all_item_ids))
                neg_embs = all_item_embeddings[i:end_i]  # [M, D]
                M = neg_embs.size(0)

                # Chunked computation to avoid B*M expansion OOM
                chunk_scores_list = []
                for j in range(0, batch_size, pair_batch_size):
                    src_chunk = src_embeddings[j:j + pair_batch_size]  # [b, D]
                    b = src_chunk.size(0)
                    src_rep = src_chunk.unsqueeze(1).expand(-1, M, -1)    # [b, M, D]
                    neg_rep = neg_embs.unsqueeze(0).expand(b, -1, -1)     # [b, M, D]
                    scores_flat = model[1](
                        src_rep.reshape(-1, src_embeddings.size(1)),
                        neg_rep.reshape(-1, src_embeddings.size(1))
                    ).squeeze(-1)  # [b*M]
                    scores_flat = torch.nan_to_num(scores_flat, nan=0.0, posinf=1e6, neginf=-1e6)
                    chunk_scores_list.append(scores_flat.view(b, M))
                chunk_scores = torch.cat(chunk_scores_list, dim=0)  # [B, M]
                chunk_scores = torch.nan_to_num(chunk_scores, nan=0.0, posinf=1e6, neginf=-1e6)
                all_neg_scores_list.append(chunk_scores)

            all_neg_scores = torch.cat(all_neg_scores_list, dim=1)  # [B, N]
            all_neg_scores = torch.nan_to_num(all_neg_scores, nan=0.0, posinf=1e6, neginf=-1e6)

            # Per-sample positive masking: exclude only the true dst for each sample
            batch_dst_tensor = torch.from_numpy(batch_dst_node_ids).to(device)  # [B]
            is_positive = (all_item_ids_tensor.unsqueeze(0) == batch_dst_tensor.unsqueeze(1))  # [B, N]
            all_neg_scores_masked = all_neg_scores.clone()
            all_neg_scores_masked[is_positive] = -1e9  # push positive to bottom

            # Compute metrics
            batch_metrics = get_retrival_metrics(pos_scores, all_neg_scores_masked)
            evaluate_metrics.append(batch_metrics)

            # Optional: approximate loss with sampled negatives
            if all_neg_scores.size(1) > 100:
                sampled_neg = all_neg_scores[:, :100]  # [B, 100]
                predictions = torch.cat([pos_scores.unsqueeze(1), sampled_neg], dim=1)  # [B, 101]
                labels = torch.cat([
                    torch.ones(batch_size, 1, device=device),
                    torch.zeros(batch_size, 100, device=device)
                ], dim=1)
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = loss_func(predictions, labels)
                evaluate_losses.append(loss.item())

            evaluate_idx_data_loader_tqdm.set_description(f'eval batch {batch_idx + 1}')

        return evaluate_losses, evaluate_metrics

def evaluate_model_retriva_ori(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if True:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # if not graph model, compute the dst rep first
        # if model_name not in ['TGAT', 'CAWN', 'TCL', 'GraphMixer', 'JODIE', 'DyRep', 'TGN', 'DyGFormer']:
        #     all_evaluate_node_embedding = model[0].compute_dst_node_temporal_embeddings(dst_node_ids=evaluate_data.dst_node_ids)

        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            
            num_evaluate_neg = 10000
            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                _, all_batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=num_evaluate_neg,
                                                                                batch_src_node_ids=batch_src_node_ids,
                                                                                batch_dst_node_ids=batch_dst_node_ids,
                                                                                current_batch_start_time=batch_node_interact_times[0],
                                                                                current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, all_batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=num_evaluate_neg)
            batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            all_batch_neg_src_node_embeddings, all_batch_neg_dst_node_embeddings = [], []
            if model_name in [
                'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
                'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain'
            ]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings = \
                        model[0].compute_src_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=num_neighbors)

                    batch_dst_node_embeddings = \
                        model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_dst_node_ids)

                    # get temporal embedding of negative destination nodes
                    # one Tensor, with shape (batch_size, node_feat_dim)
                    batch_neg_dst_node_embeddings = \
                        model[0].compute_dst_node_temporal_embeddings(dst_node_ids=all_batch_neg_dst_node_ids)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative logits
            positive_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1) # batch_size
            batch_src_node_embeddings = batch_src_node_embeddings.unsqueeze(1).tile(1,num_evaluate_neg,1)
            batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.tile(1,len(batch_src_node_ids),1).view_as(batch_src_node_embeddings)
            negative_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)
            logits = torch.cat([positive_logits, negative_logits[:,0]], dim=0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
            labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits[:,0])], dim=0)
            predicts = torch.sigmoid(logits)

            loss = loss_func(input=logits, target=labels)
            evaluate_losses.append(loss.item())

            positive_probabilities = torch.sigmoid(positive_logits)
            negative_probabilities = torch.sigmoid(negative_logits)
            evaluate_metrics.append(get_retrival_metrics(positive_probabilities, negative_probabilities))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


def evaluate_model_edge_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the edge classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()

    with torch.no_grad():
        evaluate_total_loss, evaluate_y_trues, evaluate_y_predicts = 0.0, [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices]

            if model_name in [
                'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
                'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain'
            ]:
                batch_src_node_embeddings = model[0].compute_src_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  num_neighbors=num_neighbors)
                batch_dst_node_embeddings = model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_dst_node_ids)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get predicted probabilities, shape (batch_size, )
            predicts = model[1](x_1=batch_src_node_embeddings, x_2 = batch_dst_node_embeddings, rel_embs = model[0].edge_raw_features)
            pred_labels = torch.max(predicts, dim=1)[1]
            labels = torch.from_numpy(batch_labels).int().type(torch.LongTensor).to(predicts.device)

            loss = loss_func(input=predicts, target=labels)

            evaluate_total_loss += loss.item()

            evaluate_y_trues.append(labels)
            evaluate_y_predicts.append(pred_labels)

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

        evaluate_total_loss /= (batch_idx + 1)
        evaluate_y_trues = torch.cat(evaluate_y_trues, dim=0)
        evaluate_y_predicts = torch.cat(evaluate_y_predicts, dim=0)

        evaluate_metrics = get_edge_classification_metrics(predicts=evaluate_y_predicts, labels=evaluate_y_trues)

    return evaluate_total_loss, evaluate_metrics
