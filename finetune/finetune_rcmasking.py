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
from tensorboardX import SummaryWriter

# criterion = nn.BCELoss(reduction = "none")
criterion = nn.BCEWithLogitsLoss(reduction = "none")
def train(cfg, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

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
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
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


@hydra.main(version_base=None, config_path="/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/conf", config_name="finetune")
def main(cfg):
    # Training settings
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)
        
#     wandb.login(key='11aa569e656b0bc19ad079b4bdaaa9ea3553694a')
#     wandb.init(project="Chemical-Reaction-Stage2-Finetune", name=args.run_name)
    
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

    root_dataset = '/share/project/tanghan/sharefs-hantang/chem/dataset'
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

    train_loader = DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers)

    #set up model
    model = GNN_graphpred(cfg.model.num_layer, cfg.model.emb_dim, num_tasks, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, graph_pooling = cfg.model.graph_pooling, gnn_type = cfg.model.gnn_type)
    if cfg.model.input_model_file:
        print("Load from a pretrained model")
        model.from_pretrained(cfg.model.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if cfg.model.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
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
