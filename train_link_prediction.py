import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.SASRec import SASRec
from models.HSTU import HSTU
from models.SLA import SLA
from models.GLA import GLA
from models.GSA import GSA
from models.MoM import MoM
from models.MAMBA2 import MAMBA2
from models.ATLAS import ATLAS
from models.GDeltanet import GDeltanet
from models.TTT import TTT
from models.TITANS import TITANS
from models.prism import PRISM
from models.prism_ablate_l1 import PRISM as PRISM_AblateL1
from models.prism_ablate_no_nonlinear import PRISM as PRISM_AblateNoNonlinear
from models.prism_ablate_no_shortconv import PRISM as PRISM_AblateNoShortConv
from models.prism_ablate_no_gain import PRISM as PRISM_AblateNoGain
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, evaluate_model_retrival
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    #args.device = 'cpu'

    if args.nan_debug:
        torch.autograd.set_detect_anomaly(True)

    def _assert_finite(tensor, name, step_info):
        if tensor is None:
            return
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            with torch.no_grad():
                stats = {
                    "min": torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item(),
                    "max": torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item(),
                    "mean": torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).mean().item(),
                }
            logger.error(f"NaN/Inf detected at {step_info} in {name}. stats={stats}")
            raise RuntimeError(f"NaN/Inf detected in {name} at {step_info}")

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, cat_num = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, args = args)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    # new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    # new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}{args.use_feature}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        logger.info(f'node feature size {node_raw_features.shape}')
        logger.info(f'edge feature size {edge_raw_features.shape}')
        logger.info(f'node feature example {node_raw_features[1][:5]}')
        logger.info(f'edge feature example {edge_raw_features[1][:5]}')

        # create model
        if args.model_name == 'SASRec':
            dynamic_backbone = SASRec(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'HSTU':
            dynamic_backbone = HSTU(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'SLA':
            dynamic_backbone = SLA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'ATLAS':
            dynamic_backbone = ATLAS(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GLA':
            dynamic_backbone = GLA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                        time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GSA':
            dynamic_backbone = GSA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                        time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GDeltanet':
            dynamic_backbone = GDeltanet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                       time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TTT':
            dynamic_backbone = TTT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                        time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TITANS':
            dynamic_backbone = TITANS(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                        time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'MAMBA2':
            dynamic_backbone = MAMBA2(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                        time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device, state_dim=16, mimo_rank=1)
        elif args.model_name == 'MoM':
            dynamic_backbone = MoM(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim = args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain']:
            prism_cls_map = {
                'prism': PRISM,
                'prism_ablate_l1': PRISM_AblateL1,
                'prism_ablate_no_nonlinear': PRISM_AblateNoNonlinear,
                'prism_ablate_no_shortconv': PRISM_AblateNoShortConv,
                'prism_ablate_no_gain': PRISM_AblateNoGain,
            }
            dynamic_backbone = prism_cls_map[args.model_name](
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                num_neighbors=args.num_neighbors,
                time_feat_dim=args.time_feat_dim,
                embedding_dim=args.channel_embedding_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                device=args.device,
                solver_steps=args.num_experts,
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        src_out_layer = getattr(dynamic_backbone, 'output_layer_src', None)
        dst_out_layer = getattr(dynamic_backbone, 'output_layer_dst', None)
        src_out_dim = src_out_layer.out_features if hasattr(src_out_layer, 'out_features') else args.channel_embedding_dim
        dst_out_dim = dst_out_layer.out_features if hasattr(dst_out_layer, 'out_features') else args.channel_embedding_dim
        link_predictor = MergeLayer(input_dim1=src_out_dim, input_dim2=dst_out_dim,
                                    hidden_dim=args.channel_embedding_dim, output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()

        for epoch in range(args.num_epochs):

            model.train()
            model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively
                if args.model_name in [
                    'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
                    'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain'
                ]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings = batch_src_node_embeddings = \
                        model[0].compute_src_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors = args.num_neighbors)
                    batch_dst_node_embeddings = \
                        model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_dst_node_ids)

                    # get temporal embedding of negative destination nodes
                    # one Tensor, with shape (batch_size, node_feat_dim)
                    batch_neg_dst_node_embeddings = \
                        model[0].compute_dst_node_temporal_embeddings(dst_node_ids=batch_neg_dst_node_ids)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # NaN/Inf debug checks (embeddings)
                if args.nan_debug:
                    for _name in [
                        "batch_src_node_embeddings",
                        "batch_dst_node_embeddings",
                        "batch_neg_src_node_embeddings",
                        "batch_neg_dst_node_embeddings",
                    ]:
                        _assert_finite(locals().get(_name, None), _name, f"epoch {epoch + 1} batch {batch_idx + 1}")

                # get positive and negative probabilities, shape (batch_size, )
                positive_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
                negative_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

                if args.nan_debug:
                    _assert_finite(positive_logits, "positive_logits", f"epoch {epoch + 1} batch {batch_idx + 1}")
                    _assert_finite(negative_logits, "negative_logits", f"epoch {epoch + 1} batch {batch_idx + 1}")

                logits = torch.cat([positive_logits, negative_logits], dim=0)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
                labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=0)

                loss = loss_func(input=logits, target=labels)
                predicts = torch.sigmoid(logits)

                if args.nan_debug:
                    _assert_finite(logits, "logits", f"epoch {epoch + 1} batch {batch_idx + 1}")
                    _assert_finite(loss, "loss", f"epoch {epoch + 1} batch {batch_idx + 1}")
                    _assert_finite(predicts, "predicts", f"epoch {epoch + 1} batch {batch_idx + 1}")
                # if epoch >= 9:
                #     loss += 0.1*distrill_loss

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if (epoch + 1) % args.test_interval_epochs == 0:
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after training so it can be used for new validation nodes
                    train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                        model=model,
                                                                        neighbor_sampler=full_neighbor_sampler,
                                                                        evaluate_idx_data_loader=val_idx_data_loader,
                                                                        evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                        evaluate_data=val_data,
                                                                        loss_func=loss_func,
                                                                        num_neighbors=args.num_neighbors,
                                                                        time_gap=args.time_gap)

                # if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                #     # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                #     val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                #     # reload training memory bank for new validation nodes
                #     model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

                # new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                #                                                                         model=model,
                #                                                                         neighbor_sampler=full_neighbor_sampler,
                #                                                                         evaluate_idx_data_loader=new_node_val_idx_data_loader,
                #                                                                         evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                #                                                                         evaluate_data=new_node_val_data,
                #                                                                         loss_func=loss_func,
                #                                                                         num_neighbors=args.num_neighbors,
                #                                                                         time_gap=args.time_gap)

                # if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                #     # reload validation memory bank for testing nodes or saving models
                #     # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                #     model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
                # logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
                # for metric_name in new_node_val_metrics[0].keys():
                #     logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')
                
                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                #if args.model_name in ['JODIE', 'DyRep', 'TGN', 'MemGPR']:
                    # reload validation memory bank for new testing nodes
                   # model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                # new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                #                                                                              model=model,
                #                                                                              neighbor_sampler=full_neighbor_sampler,
                #                                                                              evaluate_idx_data_loader=new_node_test_idx_data_loader,
                #                                                                              evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                #                                                                              evaluate_data=new_node_test_data,
                #                                                                              loss_func=loss_func,
                #                                                                              num_neighbors=args.num_neighbors,
                #                                                                              time_gap=args.time_gap)

                # if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    # model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                # logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                # for metric_name in new_node_test_metrics[0].keys():
                #     logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

        # pmy for debug
        # eff_seq_len = np.concatenate(model[0].eff_seq_len, axis=0)  # (num_of_sample, )
        # plt.figure(figsize=(10, 6))
        # plt.hist(eff_seq_len, bins=range(0, np.max(eff_seq_len)+2), color='skyblue', edgecolor='black')
        # plt.xlabel("effective item num in each sequence")
        # plt.ylabel("num of sample")
        # plt.axvline(np.mean(eff_seq_len), color='r', linestyle='--', label=f"mean val: {np.mean(eff_seq_len):.2f}")
        # plt.axvline(np.median(eff_seq_len), color='g', linestyle='--', label=f"median val: {np.median(eff_seq_len)}")
        # plt.legend()
        # plt.show()
        # plt.savefig("nuique_seq_len_distribution.png", dpi=300, bbox_inches='tight')

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        non_graph_retrieval_models = {
            'SASRec', 'HSTU', 'SLA', 'GLA', 'GSA', 'MoM', 'MAMBA2', 'ATLAS', 'GDeltanet', 'TTT', 'TITANS',
            'prism', 'prism_ablate_l1', 'prism_ablate_no_nonlinear', 'prism_ablate_no_shortconv', 'prism_ablate_no_gain',
        }

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            if args.model_name not in non_graph_retrieval_models:
                logger.warning(f'Full retrieval only supports non-graph models. Got {args.model_name}. '
                               'fallback to link prediction evaluation for validation.')
                val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                         model=model,
                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                         evaluate_idx_data_loader=val_idx_data_loader,
                                                                         evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                         evaluate_data=val_data,
                                                                         loss_func=loss_func,
                                                                         num_neighbors=args.num_neighbors,
                                                                         time_gap=args.time_gap)
            else:
                try:
                    val_losses, val_metrics = evaluate_model_retrival(model_name=args.model_name,
                                                                      model=model,
                                                                      neighbor_sampler=full_neighbor_sampler,
                                                                      evaluate_idx_data_loader=val_idx_data_loader,
                                                                      evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                      evaluate_data=val_data,
                                                                      loss_func=loss_func,
                                                                      all_item_ids=full_data.dst_node_ids,
                                                                      num_neighbors=args.num_neighbors,
                                                                      time_gap=args.time_gap)
                except NotImplementedError as exc:
                    logger.warning(f'{exc}. fallback to link prediction evaluation for validation.')
                    val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap)

            # new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
            #                                                                            model=model,
            #                                                                            neighbor_sampler=full_neighbor_sampler,
            #                                                                            evaluate_idx_data_loader=new_node_val_idx_data_loader,
            #                                                                            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
            #                                                                            evaluate_data=new_node_val_data,
            #                                                                            loss_func=loss_func,
            #                                                                            num_neighbors=args.num_neighbors,
            #                                                                            time_gap=args.time_gap)

        #if args.model_name in ['JODIE', 'DyRep', 'TGN', 'MemGPR']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            #val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        if args.model_name not in non_graph_retrieval_models:
            logger.warning(f'Full retrieval only supports non-graph models. Got {args.model_name}. '
                           'fallback to link prediction evaluation for test.')
            test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap)
        else:
            try:
                test_losses, test_metrics = evaluate_model_retrival(model_name=args.model_name,
                                                                    model=model,
                                                                    neighbor_sampler=full_neighbor_sampler,
                                                                    evaluate_idx_data_loader=test_idx_data_loader,
                                                                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                    evaluate_data=test_data,
                                                                    loss_func=loss_func,
                                                                    all_item_ids=full_data.dst_node_ids,
                                                                    num_neighbors=args.num_neighbors,
                                                                    time_gap=args.time_gap)
            except NotImplementedError as exc:
                logger.warning(f'{exc}. fallback to link prediction evaluation for test.')
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

        # if args.model_name in ['JODIE', 'DyRep', 'TGN']:
        #     # reload validation memory bank for new testing nodes
        #     model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        # new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                              model=model,
        #                                                                              neighbor_sampler=full_neighbor_sampler,
        #                                                                              evaluate_idx_data_loader=new_node_test_idx_data_loader,
        #                                                                              evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
        #                                                                              evaluate_data=new_node_test_data,
        #                                                                              loss_func=loss_func,
        #                                                                              num_neighbors=args.num_neighbors,
        #                                                                              time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

            # logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            # for metric_name in new_node_val_metrics[0].keys():
            #     average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            #     logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            #     new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        # logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        # for metric_name in new_node_test_metrics[0].keys():
        #     average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        #     logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        #     new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
            # new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        # new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                # "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                # "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                # "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        # for metric_name in new_node_val_metric_all_runs[0].keys():
        #     logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
        #     logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
        #                 f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    # for metric_name in new_node_test_metric_all_runs[0].keys():
    #     logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
    #     logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
    #                 f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
