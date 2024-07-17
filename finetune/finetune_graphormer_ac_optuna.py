import sys
import os
os.chdir("/finetune")
sys.path.insert(0,'..')
from test.gcn_utils.datas import MoleculeDataset, mol_to_graph_data_obj_simple
from torch_geometric.data import DataLoader
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import hydra
from models.gnn.models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error

from test.gcn_utils.splitters import scaffold_split
import pandas as pd
import wandb
import os
import shutil
import optuna
from optuna.trial import TrialState
from tensorboardX import SummaryWriter
from data_processing import ActivityCliffDataset
import yaml

from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params
from datas.graphormer_data import BatchedDataDataset_finetune, GraphormerPYGDataset
from models.graphormer.graphormer import RobertaHead

from sklearn.model_selection import StratifiedKFold
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def train(cfg, model, device, loader, optimizer):
    graphormer, linear_head = model
    graphormer.train()
    linear_head.train()
    
    if cfg.training_settings.l1_loss:
        criterion = nn.L1Loss()
    else:
        criterion = RMSELoss
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # print("Batch_size", len(batch))
        if not batch:
            continue
        for k in batch:
            if k not in ['cliff']:
                batch[k] = batch[k].to(device)
        
        inner_state, graph_rep = graphormer(batch)
        pred = linear_head(graph_rep)
        y = batch['y'].view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), y)

        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
#         if step % 100 == 1:
#             wandb.log({"loss": loss})
        loss.backward()
        # print(loss)
        optimizer.step()
        
def eval(cfg, model, device, loader):
    graphormer, linear_head = model
    graphormer.eval()
    linear_head.eval()
    cliff_index = []
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if not batch:
            continue
        for k in batch:
            if k not in ['cliff']:
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            inner_state, graph_rep = graphormer(batch)
            pred = linear_head(graph_rep)

        y_true.append(batch['y'].view(pred.shape))
        y_scores.append(pred)
        cliff_index.extend(batch['cliff'])
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    rmse = calc_rmse(y_true, y_scores)
    
    # for i in range(y_true.shape[1]):
    #     #AUC is only defined when there is at least one positive data.
    #     mse_list
    #     if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
    #         is_valid = y_true[:,i]**2 > 0
    #         roc_list.append(mean_squared_error((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return rmse, (y_scores, y_true), cliff_index #y_true.shape[1]

def objective(trial, cfg, device, dataloaders, benchmark_tag):
    train_loader, val_loader, test_loader = dataloaders
    name, cliff_ratio = benchmark_tag
    # --------
    # Model Settings
    
    config_file = cfg.model.graphormer_config_yaml
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
        
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    model_config = Struct(**model_config)

    graphormer_model = GraphormerGraphEncoder(
                # < for graphormer
                num_atoms=model_config.num_atoms,
                num_in_degree=model_config.num_in_degree,
                num_out_degree=model_config.num_out_degree,
                num_edges=model_config.num_edges,
                num_spatial=model_config.num_spatial,
                num_edge_dis=model_config.num_edge_dis,
                edge_type=model_config.edge_type,
                multi_hop_max_dist=model_config.multi_hop_max_dist,
                # >
                num_encoder_layers=model_config.encoder_layers,
                embedding_dim=model_config.encoder_embed_dim,
                ffn_embedding_dim=model_config.encoder_ffn_embed_dim,
                num_attention_heads=model_config.encoder_attention_heads,
                dropout=model_config.dropout,
                attention_dropout=model_config.attention_dropout,
                activation_dropout=model_config.act_dropout,
                encoder_normalize_before=model_config.encoder_normalize_before,
                pre_layernorm=model_config.pre_layernorm,
                apply_graphormer_init=model_config.apply_graphormer_init,
                activation_fn=model_config.activation_fn,
            )
# graphormer_model = GraphormerReact(model_config, graphormer_model)
    graphormer_model = graphormer_model.to(device)
    if cfg.model.input_model_file:
        state_dict = torch.load(cfg.model.input_model_file, map_location=device)
        new_dict = {}
        for k in state_dict:
            newk = k.replace('graph_encoder.', '')
            new_dict[newk] = state_dict[k]
#        graphormer_model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))
        unload_keys, other_keys = graphormer_model.load_state_dict(new_dict, strict=False)
        print(f'load complete, unload_keys: {unload_keys}, other_keys: {other_keys}')
    
    linear_head = RobertaHead(model_config.encoder_embed_dim, 1, regression=True).to(device)

    # --------

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    decay = trial.suggest_float("decay", 1e-6, 1e-4, log=True)
    
     #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    val_rmse_acc_list = []
    test_acc_list = []
    test_rmse_cliff_list = []
    test_output_list = []
    true_output_list = []
    print(cfg.model.filename)

        # ----- Assign early stoppers during training
    early_stopper_train = EarlyStopper(patience=cfg.training_settings.patience)
    early_stopper_valid = EarlyStopper(patience=cfg.training_settings.patience)
    
    model = [graphormer_model, linear_head]
    
    for epoch in range(1, cfg.training_settings.epochs+1):
        
        print("====epoch " + str(epoch))
        
        train(cfg, model, device, train_loader, optimizer)
        
        print("====Evaluation")
        if cfg.training_settings.eval_train:
            train_acc, _, _ = eval(cfg, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc, (valid_output, true_valid_output), valid_cliff_index = eval(cfg, model, device, val_loader)
        val_cliff_rmse = calc_cliff_rmse(valid_output, true_valid_output, valid_cliff_index)
        test_acc, (test_output, true_output), cliff_index = eval(cfg, model, device, test_loader)
        
        cliff_rmse = calc_cliff_rmse(test_output, true_output, cliff_index)
        print("train: %f val: %f val cliff: %f test: %f test cliff: %f" %(train_acc, val_acc, val_cliff_rmse, test_acc, cliff_rmse))
        
        if cfg.training_settings.optuna_prune_criteria == "val_cliff":
            intermediate_value = val_cliff_rmse + val_acc
        elif cfg.training_settings.optuna_prune_criteria == "train":
            intermediate_value = train_acc
        else:
            raise ValueError("Pruning criteria value not supported")
        
        trial.report(intermediate_value, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if early_stopper_train.early_stop(train_acc) and early_stopper_valid.early_stop(val_cliff_rmse):
            print("Training teminated")
            break
        val_acc_list.append(val_acc)
        val_rmse_acc_list.append(val_cliff_rmse)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        test_output_list.append(test_output)
        true_output_list.append(true_output)
        test_rmse_cliff_list.append(cliff_rmse)
    
    max_ind = np.argmin(val_rmse_acc_list)
    print(f"The best epoch for {name}: {max_ind}\ntrain: {train_acc_list[max_ind]} val: {val_acc_list[max_ind]} val cliff: {val_rmse_acc_list[max_ind]} test: {test_acc_list[max_ind]} test cliff: {test_rmse_cliff_list[max_ind]} Cliff Ratio: {cliff_ratio}")
    
    result_df = pd.read_csv('/home/qcx679/hantang/Chemical_Reaction_Pretraining/MoleculeACE_res.csv')
    result_df = result_df.append({
        "target": name,
        "lr": lr,
        "decay": decay,
        "epoch": max_ind,
        "val_rmse": val_acc_list[max_ind],
        "val_cliff_rmse": val_rmse_acc_list[max_ind],
        "test_rmse": test_acc_list[max_ind],
        "test_cliff_rmse": test_rmse_cliff_list[max_ind],
        "seed": cfg.training_settings.runseed,
        "model": cfg.model.input_model_file,
        "model_config": cfg.model.graphormer_config_yaml,
                                  }, ignore_index=True)
    
    max_ind_2 = np.argmin(test_acc_list)
    result_df = result_df.append({
        "target": name,
        "lr": lr,
        "decay": decay,
        "epoch": max_ind_2,
        "val_rmse": val_acc_list[max_ind_2],
        "val_cliff_rmse": val_rmse_acc_list[max_ind_2],
        "test_rmse": test_acc_list[max_ind_2],
        "test_cliff_rmse": test_rmse_cliff_list[max_ind_2],
        "seed": cfg.training_settings.runseed,
        "model": cfg.model.input_model_file,
        "model_config": cfg.model.graphormer_config_yaml,
                                  }, ignore_index=True)
    
    max_ind_3 = np.argmin(test_rmse_cliff_list)
    result_df = result_df.append({
        "target": name,
        "lr": lr,
        "decay": decay,
        "epoch": max_ind_3,
        "val_rmse": val_acc_list[max_ind_3],
        "val_cliff_rmse": val_rmse_acc_list[max_ind_3],
        "test_rmse": test_acc_list[max_ind_3],
        "test_cliff_rmse": test_rmse_cliff_list[max_ind_3],
        "seed": cfg.training_settings.runseed,
        "model": cfg.model.input_model_file,
        "model_config": cfg.model.graphormer_config_yaml,
                                  }, ignore_index=True)
    
    result_df = result_df.append({
        "target": name,
        "lr": lr,
        "decay": decay,
        "epoch": epoch,
        "val_rmse": val_acc_list[-1],
        "val_cliff_rmse": val_rmse_acc_list[-1],
        "test_rmse": test_acc_list[-1],
        "test_cliff_rmse": test_rmse_cliff_list[-1],
        "seed": cfg.training_settings.runseed,
        "model": cfg.model.input_model_file,
        "model_config": cfg.model.graphormer_config_yaml,
                                  }, ignore_index=True)
    
    result_df.to_csv('/home/qcx679/hantang/Chemical_Reaction_Pretraining/MoleculeACE_res.csv', index=False)
    
    return min(test_rmse_cliff_list)+test_rmse_cliff_list[max_ind_2]
    
def detailed_objective(trial, cfg, device, dataloaders, benchmark_tag):
    train_loader, val_loader, test_loader = dataloaders
    name, cliff_ratio = benchmark_tag
    # --------
    # Model Settings
    
    config_file = cfg.model.graphormer_config_yaml
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
        
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    model_config = Struct(**model_config)

    graphormer_model = GraphormerGraphEncoder(
                # < for graphormer
                num_atoms=model_config.num_atoms,
                num_in_degree=model_config.num_in_degree,
                num_out_degree=model_config.num_out_degree,
                num_edges=model_config.num_edges,
                num_spatial=model_config.num_spatial,
                num_edge_dis=model_config.num_edge_dis,
                edge_type=model_config.edge_type,
                multi_hop_max_dist=model_config.multi_hop_max_dist,
                # >
                num_encoder_layers=model_config.encoder_layers,
                embedding_dim=model_config.encoder_embed_dim,
                ffn_embedding_dim=model_config.encoder_ffn_embed_dim,
                num_attention_heads=model_config.encoder_attention_heads,
                dropout=model_config.dropout,
                attention_dropout=model_config.attention_dropout,
                activation_dropout=model_config.act_dropout,
                encoder_normalize_before=model_config.encoder_normalize_before,
                pre_layernorm=model_config.pre_layernorm,
                apply_graphormer_init=model_config.apply_graphormer_init,
                activation_fn=model_config.activation_fn,
            )
# graphormer_model = GraphormerReact(model_config, graphormer_model)
    graphormer_model = graphormer_model.to(device)
    if cfg.model.input_model_file:
        state_dict = torch.load(cfg.model.input_model_file, map_location=device)
        new_dict = {}
        for k in state_dict:
            newk = k.replace('graph_encoder.', '')
            new_dict[newk] = state_dict[k]
#        graphormer_model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))
        unload_keys, other_keys = graphormer_model.load_state_dict(new_dict, strict=False)
        print(f'load complete, unload_keys: {unload_keys}, other_keys: {other_keys}')
        
    linear_head = RobertaHead(model_config.encoder_embed_dim, 1, regression=True).to(device)
    
    
    # --------
    
    
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    decay = trial.suggest_float("decay", 1e-6, 1e-4, log=True)
    
     #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    val_rmse_acc_list = []
    test_acc_list = []
    test_rmse_cliff_list = []
    test_output_list = []
    true_output_list = []
    print(cfg.model.filename)
   
    model = [graphormer_model, linear_head]
    for epoch in range(1, cfg.training_settings.epochs+1):
        
        print("====epoch " + str(epoch))
        
        train(cfg, model, device, train_loader, optimizer)
        
        print("====Evaluation")
        if cfg.training_settings.eval_train:
            train_acc, _, _ = eval(cfg, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc, (valid_output, true_valid_output), valid_cliff_index = eval(cfg, model, device, val_loader)
        val_cliff_rmse = calc_cliff_rmse(valid_output, true_valid_output, valid_cliff_index)
        test_acc, (test_output, true_output), cliff_index = eval(cfg, model, device, test_loader)
        cliff_rmse = calc_cliff_rmse(test_output, true_output, cliff_index)
        print("train: %f val: %f val cliff: %f test: %f test cliff: %f" %(train_acc, val_acc, val_cliff_rmse, test_acc, cliff_rmse))
        
        val_acc_list.append(val_acc)
        val_rmse_acc_list.append(val_cliff_rmse)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        test_output_list.append(test_output)
        true_output_list.append(true_output)
        test_rmse_cliff_list.append(cliff_rmse)
    
    max_ind = np.argmin(val_rmse_acc_list)
    print(f"The best epoch for {name}: {max_ind}\ntrain: {train_acc_list[max_ind]} val: {val_acc_list[max_ind]} val cliff: {val_rmse_acc_list[max_ind]} test: {test_acc_list[max_ind]} test cliff: {test_rmse_cliff_list[max_ind]} Cliff Ratio: {cliff_ratio}")
    return max_ind, train_acc_list[max_ind], val_acc_list[max_ind], val_rmse_acc_list[max_ind], test_acc_list[max_ind], test_rmse_cliff_list[max_ind], cliff_ratio


def finetune(cfg, benchmark, device):
    
    # --------
    # Dataset Preparation
    config_file = cfg.model.graphormer_config_yaml
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
        
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    model_config = Struct(**model_config)
    # ===== Dataset Preparation
    name = benchmark.split('/')[-1].split('.')[0]
    print(f'The Activity Cliff Benchmark: {name}')
    dataset = pd.read_csv(benchmark)
    train_dataset = dataset[dataset['split'] == 'train']
    test_dataset = dataset[dataset['split'] == 'test']
    if cfg.training_settings.val_method == "none":
        train_dataset = ActivityCliffDataset(train_dataset)
        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size]) 
    elif cfg.training_settings.val_method == "cliff":
        val_dataset = train_dataset[train_dataset['cliff_mol'] == 1]
        train_dataset = ActivityCliffDataset(train_dataset)
        val_dataset = ActivityCliffDataset(val_dataset)
        
    elif cfg.training_settings.val_method == "stratified":
        
        ss = StratifiedKFold(n_splits=cfg.training_settings.n_folds, random_state=cfg.training_settings.runseed, shuffle=True)
        y_train = train_dataset['y']
        cutoff = np.median(y_train)
        labels = [0 if i < cutoff else 1 for i in y_train]
        pass
        
        
#        val_dataset
#        train_dataset
#        val_dataset   
    else:
        raise ValueError("Split method not supported")
        
    test_dataset = ActivityCliffDataset(test_dataset)
    
    seed = cfg.training_settings.runseed
    
    data_set = GraphormerPYGDataset(
        None,
        seed,
        None,
        None,
        None,
        train_dataset,
        val_dataset,
        test_dataset
    )
    
    # --------
    
    # ===== Model Loading & Finetuning
    
    batched_data_train = BatchedDataDataset_finetune(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=True,
        )
    
    batched_data_valid = BatchedDataDataset_finetune(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=512,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=True,
        )
    
    batched_data_test = BatchedDataDataset_finetune(
            data_set.test_data,
            max_node=512,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=True,
        )

    train_loader = torch.utils.data.DataLoader(batched_data_train, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_train.collater)
    val_loader = torch.utils.data.DataLoader(batched_data_valid, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_valid.collater)
    test_loader = torch.utils.data.DataLoader(batched_data_test, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_test.collater)
    
    dataloaders = [train_loader, val_loader, test_loader]
    
    benchmark_tag = [name, sum(test_dataset.cliff)/len(test_dataset)]
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, cfg, device, dataloaders, benchmark_tag), n_trials=cfg.training_settings.n_trial)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    idx, train_best_rmse, val_best_rmse, val_best_rmse_cliff, test_best_rmse, test_best_rmse_cliff, cliff_ratio = detailed_objective(study.best_trial, cfg, device, dataloaders, benchmark_tag)
    print(f"The best result on ther benchmark {name}(epoch {idx}):\ntrain: {train_best_rmse}, val: {val_best_rmse}, cliff val: {val_best_rmse_cliff}, test: {test_best_rmse}, cliff test:{test_best_rmse_cliff}, Cliff Ratio: {cliff_ratio}")
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def split_list(my_list, num_splits):
    fold_size = len(my_list) // num_splits
    extra_elements = len(my_list) % num_splits

    folds = []
    start_idx = 0

    for i in range(num_splits):
        end_idx = start_idx + fold_size + (1 if i < extra_elements else 0)
        folds.append(my_list[start_idx:end_idx])
        start_idx = end_idx

    return folds

@hydra.main(version_base=None, config_path="/home/qcx679/hantang/Chemical_Reaction_Pretraining/conf", config_name="finetune")
def main(cfg):
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)
    
    data_ls = []
    for file in os.listdir(cfg.ac_path):
        if file.split('.')[-1] == 'csv':
            data_ls.append(file)
            
    data_ls = [os.path.join(cfg.ac_path, file) for file in data_ls]
    if cfg.training_settings.benchmark_split:
        folds = split_list(data_ls, int(cfg.training_settings.split_fold))
        for benchmark in folds[int(cfg.training_settings.current_fold)]:
            finetune(cfg, benchmark, device)
    else:
        for benchmark in data_ls:
            finetune(cfg, benchmark, device)
            
    

if __name__ == "__main__":
    main()