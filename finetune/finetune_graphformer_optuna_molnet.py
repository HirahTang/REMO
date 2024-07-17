import argparse
import sys
import os
sys.path.insert(0,'..')
from test.gcn_utils.datas import MoleculeDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from models.graphormer.graphormer import RobertaHead
from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
from datas.graphormer_data import GraphormerPYGDataset
from datas.graphormer_data import BatchedDataDataset_finetune

from test.gcn_utils.splitters import scaffold_split
import pandas as pd
import wandb
import os
import shutil
import optuna
from optuna.trial import TrialState
from tensorboardX import SummaryWriter

import yaml

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    # model.train()
    graphormer, linear_head = model
    graphormer.train()
    linear_head.train()


    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            if k not in ['cliff']:
                batch[k] = batch[k].to(device)
        # batch = batch.to(device)
        # pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

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
        # if step % 100 == 1:
        #     wandb.log({"loss": loss})
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    graphormer, linear_head = model
    graphormer.eval()
    linear_head.eval()
    # model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            if k not in ['cliff']:
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

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def define_gfmodel(args, model_config):

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
    if args.input_model_file:
        state_dict = torch.load(args.input_model_file, map_location=device)
        new_dict = {}
        for k in state_dict:
            newk = k.replace('graph_encoder.', '')
            new_dict[newk] = state_dict[k]
        unload_keys, other_keys = graphormer_model.load_state_dict(new_dict, strict=False)
        print(f'load complete, unload_keys: {unload_keys}, other_keys: {other_keys}')
    
    linear_head = RobertaHead(model_config.encoder_embed_dim, args.num_tasks, regression=True).to(device)
    return graphormer_model, linear_head

def define_model(trial):
    graph_pooling = trial.suggest_categorical("graph_pooling", ['sum', 'mean' , 'max', 'attention'])
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)
    return model, graph_pooling

def parser_collect():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
#    parser.add_argument('--lr', type=float, default=0.001,
#                        help='learning rate (default: 0.001)')
#    parser.add_argument('--lr_scale', type=float, default=1,
#                        help='relative learning rate for the feature extraction layer (default: 1)')
#    parser.add_argument('--decay', type=float, default=0,
#                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
#    parser.add_argument('--graph_pooling', type=str, default="mean",
#                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--run_name', type=str, default = "training", help="running name in WanDB")
    parser.add_argument('--n_trials', type=int, default = 30, help='Number of trials for hyperparameter sweeping')
    parser.add_argument('--patience', type=int, default = 30, help='early stop patience')
    parser.add_argument('--num_tasks', type=int, default = 1, help='Number of tasks')
    parser.add_argument('--graphormer_config_yaml', type=str, default = "training", help="graphformer config")
    args = parser.parse_args()
    
    return args

def objective(trial, args, device, dataloaders, model_config):
    train_loader, val_loader, test_loader = dataloaders
 
    # wandb.login(key='11aa569e656b0bc19ad079b4bdaaa9ea3553694a')
    # wandb.init(project="Chemical-Reaction-Stage2-Finetune", name=args.run_name)
    

    #set up model
    # Graph pooling to tune

    # early_stopper_train = EarlyStopper(patience=args.patience)
    early_stopper_valid = EarlyStopper(patience=args.patience)


    dropout = trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.5])

    model_config.dropout = dropout
    graphormer_model, linear_head = define_gfmodel(args, model_config)

    #set up optimizer
    #different learning rate for different part of GNN
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    # decay = trial.suggest_float("decay", 1e-6, 1e-4, log=True)
    decay = trial.suggest_float("decay", 1e-6, 1e-3, log=True)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    
    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":lr * 1}) # TODO fix me
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = f'runs/finetune_cls_runseed{str(args.runseed)}-dset_{args.dataset}' + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    model = [graphormer_model, linear_head]

    for epoch in range(1, args.epochs+1):
        # if epoch == args.epochs:
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)
        if epoch == args.epochs:
            print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)
        if epoch == args.epochs:
            print('Trail finished')
        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        # early_stopper_train.early_stop(train_acc) and 
        if early_stopper_valid.early_stop(val_acc):
            print("Training early teminated")
            break


        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        # print("")

    if not args.filename == "":
        writer.close()
    
    return val_acc
    # return test_acc

def detailed_objective(trial, args, device, dataloaders, model_config):
    train_loader, val_loader, test_loader = dataloaders
    # early_stopper_train = EarlyStopper(patience=args.patience)
    early_stopper_valid = EarlyStopper(patience=args.patience)
    dropout = trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.5])
    model_config.dropout = dropout    

    graphormer_model, linear_head = define_gfmodel(args, model_config)

    #set up optimizer
    #different learning rate for different part of GNN
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # # lr_scale = trial.suggest_float("lr_scale", 0.5, 2, step=0.1)
    # decay = trial.suggest_float("weight_decay", 0, 1e-3)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    decay = trial.suggest_float("decay", 1e-6, 1e-4, log=True)


    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":lr*1})
    optimizer = optim.Adam(model_param_group, lr=lr, weight_decay=decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = f'runs/finetune_cls_runseed{str(args.runseed)}-dset_{args.dataset}' + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)
    
    model = [graphormer_model, linear_head]
    for epoch in range(1, args.epochs+1):
        
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        if early_stopper_valid.early_stop(val_acc):
            print("Training early teminated")
            break

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        
        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")
        
    if not args.filename == "":
        writer.close()
    
    return test_acc


if __name__ == "__main__":
    args = parser_collect()
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
        
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
        args.epochs = 10
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
        args.epochs = 50
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        args.batch_size = 32
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    args.num_tasks = num_tasks
    root_dataset = '/data/protein/SKData/MOLNet/dataset'
    #set up dataset
    dataset = MoleculeDataset(f"{root_dataset}/" + args.dataset, dataset=args.dataset)
    print(dataset)
    
    # create train val and test dataset
    if args.split == "scaffold":
        smiles_list = pd.read_csv(f"{root_dataset}/" + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(f"{root_dataset}/" + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    seed = args.runseed
    
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

    config_file = args.graphormer_config_yaml
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
        
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    model_config = Struct(**model_config)

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
            max_node=512,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
        )

    batch_size = args.batch_size # may change for sider
    train_loader = DataLoader(batched_data_train, batch_size=batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = batched_data_train.collater)
    val_loader = DataLoader(batched_data_valid, batch_size=batch_size, shuffle=False, num_workers = args.num_workers, collate_fn = batched_data_valid.collater)
    test_loader = DataLoader(batched_data_test, batch_size=batch_size, shuffle=False, num_workers = args.num_workers, collate_fn = batched_data_test.collater)

    dataloaders = [train_loader, val_loader, test_loader]    


    



    # study = optuna.create_study(direction="maximize")
    # args.epochs = args.epochs
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, device, dataloaders, model_config), args.n_trials, timeout=None)
    # study.optimize(objective, n_trials=args.n_trials, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    print("Best trial test acc:")
    # args.epochs = args.epochs
    detailed_objective(study.best_trial, args, device, dataloaders, model_config)
