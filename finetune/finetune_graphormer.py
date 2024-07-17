# import argparse
import sys
import os
os.chdir("/finetune/")
sys.path.insert(0,'..')
from test.gcn_utils.datas import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import hydra
from models.gnn.models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from test.gcn_utils.splitters import scaffold_split
import pandas as pd
import wandb
import os
import shutil
import optuna
from optuna.trial import TrialState
import yaml
from tensorboardX import SummaryWriter
from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params
from datas.graphormer_data import BatchedDataDataset_finetune, GraphormerPYGDataset
from models.graphormer.graphormer import RobertaHead
# criterion = nn.BCELoss(reduction = "none")
criterion = nn.BCEWithLogitsLoss(reduction = "none")
def train(cfg, model, device, loader, optimizer):
    graphormer, linear_head = model
    graphormer.train()
    linear_head.train()
    
    

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            batch[k] = batch[k].to(device)
        inner_state, graph_rep = graphormer(batch)
        pred = linear_head(graph_rep)
        y = batch['y'].view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
#         if step % 100 == 1:
#             wandb.log({"loss": loss})
        loss.backward()

        optimizer.step()


def eval(cfg, model, device, loader):
    graphormer, linear_head = model
    graphormer.eval()
    linear_head.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            inner_state, graph_rep = graphormer(batch)
            pred = linear_head(graph_rep)
        

        y_true.append(batch['y'].view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), (y_scores, y_true) #y_true.shape[1]


@hydra.main(version_base=None, config_path="../conf", config_name="finetune")
def main(cfg):
    # Training settings
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)
        
#     wandb.login(key='11aa569e656b0bc19ad079b4bdaaa9ea3553694a')
#     wandb.init(project="Chemical-Reaction-Stage2-Finetune", name=args.run_name)
    
    # Set up models
    
    config_file = '/test/assets/graphormer.yaml'
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
        graphormer_model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))

    
    # ------
    
    #Bunch of classification tasks
    
    
    
    if cfg.dataset == "tox21":
        num_tasks = 12
    elif cfg.dataset == "hiv":
        num_tasks = 1
    elif cfg.dataset == "pcba":
        num_tasks = 128
    elif cfg.dataset == "muv":
        num_tasks = 17
    elif cfg.dataset == "bace":
        num_tasks = 1
    elif cfg.dataset == "bbbp":
        num_tasks = 1
    elif cfg.dataset == "toxcast":
        num_tasks = 617
    elif cfg.dataset == "sider":
        num_tasks = 27
    elif cfg.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    
    linear_head = RobertaHead(model_config.encoder_embed_dim, num_tasks).to(device)
    
    root_dataset = '/chem/dataset'
    #set up dataset
    dataset = MoleculeDataset(f"{root_dataset}/" + cfg.dataset, dataset=cfg.dataset)

    print(dataset)
    
    if cfg.training_settings.split == "scaffold":
        smiles_list = pd.read_csv(f"{root_dataset}/" + cfg.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif cfg.training_settings.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = cfg.training_settings.seed)
        print("random")
    elif cfg.training_settings.split == "random_scaffold":
        smiles_list = pd.read_csv(f"{root_dataset}/" + cfg.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = cfg.training_settings.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    seed = 0
    data_set = GraphormerPYGDataset(
        None,
        seed,
        None,
        None,
        None,
        train_dataset,
        valid_dataset,
        test_dataset
    )
    batched_data_train = BatchedDataDataset_finetune(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
        )
    
    batched_data_valid = BatchedDataDataset_finetune(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
        )
    
    batched_data_test = BatchedDataDataset_finetune(
            data_set.test_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
        )

    train_loader = torch.utils.data.DataLoader(batched_data_train, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_train.collater)
    val_loader = torch.utils.data.DataLoader(batched_data_valid, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_valid.collater)
    test_loader = torch.utils.data.DataLoader(batched_data_test, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_test.collater)


    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    test_output_list = []
    true_output_list = []
    print(cfg.model.filename)
    if not cfg.model.filename == "":
        fname = f'runs/finetune_cls_runseed{str(cfg.training_settings.runseed)}-dset_{cfg.dataset}' + '/' + cfg.model.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)
    model = [graphormer_model, linear_head]
    for epoch in range(1, cfg.training_settings.epochs+1):
        
        print("====epoch " + str(epoch))
        
        train(cfg, model, device, train_loader, optimizer)

        print("====Evaluation")
        if cfg.training_settings.eval_train:
            train_acc, _ = eval(cfg, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc, _ = eval(cfg, model, device, val_loader)
        test_acc, (test_output, true_output) = eval(cfg, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        test_output_list.append(test_output)
        true_output_list.append(true_output)
        
        if not cfg.model.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not cfg.model.filename == "":
        writer.close()
    max_ind = np.argmax(val_acc_list)
    print(f"The best epoch: {max_ind}\ntrain: {train_acc_list[max_ind]} val: {val_acc_list[max_ind]} test: {test_acc_list[max_ind]}")
    test_output = test_output_list[max_ind]
    true_output = true_output_list[max_ind]
    # print(test_output)
    # print(test_output.shape)
    # print(true_output)
    # print(true_output.shape)
    # test_output_save = test_output.numpy()
    # true_output_save = true_output.numpy()
                
    test_output_save = pd.DataFrame(test_output)
    if cfg.save_output:
        test_output_save.to_csv(f'{cfg.model.model_name}_{cfg.dataset}_{cfg.training_settings.runseed}.csv', index=False)
        path = f'/{cfg.dataset}_true.csv'
        if not os.path.exists(path):
            pd.DataFrame(true_output).to_csv(path, index=False)
if __name__ == "__main__":
    main()
