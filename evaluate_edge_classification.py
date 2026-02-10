import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn

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
from models.modules import MLPClassifier_edge
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler
from evaluate_models_utils import evaluate_model_edge_classification
from utils.DataLoader import get_idx_data_loader, get_edge_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_edge_classification_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_edge_classification_args()

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_number = \
        get_edge_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, args=args)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.load_model_name = f'edge_classification_{args.model_name}_seed{args.seed}{args.use_feature}'
        args.save_result_name = f'evaluate_edge_classification_{args.model_name}_seed{args.seed}{args.use_feature}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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
            dynamic_backbone = SASRec(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'HSTU':
            dynamic_backbone = HSTU(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'SLA':
            dynamic_backbone = SLA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                  time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GLA':
            dynamic_backbone = GLA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                  time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GSA':
            dynamic_backbone = GSA(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                  time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'MoM':
            dynamic_backbone = MoM(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                  time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'MAMBA2':
            dynamic_backbone = MAMBA2(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                      time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device, state_dim=16, mimo_rank=1)
        elif args.model_name == 'ATLAS':
            dynamic_backbone = ATLAS(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                    time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GDeltanet':
            dynamic_backbone = GDeltanet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                         time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TTT':
            dynamic_backbone = TTT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                   time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TITANS':
            dynamic_backbone = TITANS(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler, num_neighbors=args.num_neighbors,
                                      time_feat_dim=args.time_feat_dim, embedding_dim=args.channel_embedding_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
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
                neighbor_sampler=full_neighbor_sampler,
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
        edge_classifier = MLPClassifier_edge(input_dim=node_raw_features.shape[1], dropout=args.dropout, cat_num=cat_number)
        model = nn.Sequential(dynamic_backbone, edge_classifier)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # load the saved model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                       save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device
        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                new_node_raw_messages = []
                for node_raw_message in node_raw_messages:
                    new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

        loss_func = nn.CrossEntropyLoss()

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_total_loss, val_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             loss_func=loss_func,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap)

        test_total_loss, test_metrics = evaluate_model_edge_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {val_total_loss:.4f}')
            for metric_name in val_metrics.keys():
                val_metric = val_metrics[metric_name]
                logger.info(f'validate {metric_name}, {val_metric:.4f}')
                val_metric_dict[metric_name] = val_metric

        logger.info(f'test loss: {test_total_loss:.4f}')
        for metric_name in test_metrics.keys():
            test_metric = test_metrics[metric_name]
            logger.info(f'test {metric_name}, {test_metric:.4f}')
            test_metric_dict[metric_name] = test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save edge classification results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
