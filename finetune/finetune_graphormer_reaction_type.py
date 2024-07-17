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
from data_analysis.USPTO_CONFIG import USPTO_CONFIG
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
from data_processing import ActivityCliffDataset, USPTO_1k_TPL
import yaml

from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params
from datas.graphormer_data import BatchedDataDataset_finetune, GraphormerPYGDataset, GraphormerUSPTODataset
from models.graphormer.graphormer import RobertaHead

from sklearn.model_selection import StratifiedKFold
import numpy as np

criterion = nn.CrossEntropyLoss()


class ReactionClassificationHead(torch.nn.Module):
    def __init__(self, graphormer, model_config, output_size):
        super(ReactionClassificationHead, self).__init__()
        self.graphormer = graphormer
        self.output_size = output_size
        self.linear = RobertaHead(model_config.encoder_embed_dim * 2, output_size, regression=False)
        
    def forward(self, batch_input):
        react_graph = batch_input['graph_rc']
        prod_graph = batch_input['graph_pd']
        
        _, react_rep = self.graphormer(react_graph)
        _, prod_rep = self.graphormer(prod_graph)
        
        # concatenetate the react and prod rep
        rep = torch.cat([react_rep, prod_rep], dim=1)
        pred = self.linear(rep)
        return pred
    
    
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def GraphormerDFMaker(cfg, dataset):
    dataset = USPTO_1k_TPL(dataset, 
                           reagents=cfg.training_settings.reagents, 
                           filter=cfg.training_settings.filter,
                           filter_thres=cfg.training_settings.filter_thres)
    return dataset

def GraphormerDFMaker2(trainset, valtest, testset, cfg, model_config):
    seed = cfg.training_settings.runseed
    data_set = GraphormerUSPTODataset(
        None,
        seed,
        None,
        None,
        None,
        trainset,
        valtest,
        testset
    )
    
    batched_data_train = BatchedDataDataset_finetune(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
            reaction_type=True,
        )
    
    batched_data_valid = BatchedDataDataset_finetune(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
            reaction_type=True,
        )
    
    batched_data_test = BatchedDataDataset_finetune(
            data_set.test_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False,
            reaction_type=True,
        )
    
    train_loader = torch.utils.data.DataLoader(batched_data_train, 
                                               batch_size=cfg.training_settings.batch_size, 
                                               shuffle=True, 
                                               num_workers = cfg.training_settings.num_workers, 
                                               collate_fn = batched_data_train.collater)
    val_loader = torch.utils.data.DataLoader(batched_data_valid, 
                                             batch_size=cfg.training_settings.batch_size, 
                                             shuffle=False, 
                                             num_workers = cfg.training_settings.num_workers, 
                                             collate_fn = batched_data_valid.collater)
    test_loader = torch.utils.data.DataLoader(batched_data_test, 
                                              batch_size=cfg.training_settings.batch_size, 
                                              shuffle=False, 
                                              num_workers = cfg.training_settings.num_workers, 
                                              collate_fn = batched_data_test.collater)
    
    return [train_loader, val_loader, test_loader]
    
    
def Graphormer_setup(cfg, model_config, device):
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
    
    graphormer_model = graphormer_model.to(device)
    
    if cfg.training_settings.continue_training:
        intermediate_models = os.listdir(cfg.model.intermediate_model_dir)
        # take the model with the largest epoch number
        epoch = 0
        for model in intermediate_models:
            if model.startswith('reaction_classification_model_epoch'):
                epoch = max(epoch, int(model.split('.')[0].split('epoch')[1]))
                
        load_path = os.path.join(cfg.model.intermediate_model_dir, "reaction_classification_model_epoch{}.pt".format(epoch))
        
        cfg.model.input_model_file = None
        return graphormer_model, load_path, epoch
        
    if cfg.model.input_model_file:
        state_dict = torch.load(cfg.model.input_model_file, map_location=device)
        new_dict = {}
        for k in state_dict:
            newk = k.replace('graph_encoder.', '')
            new_dict[newk] = state_dict[k]
#        graphormer_model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))
        unload_keys, other_keys = graphormer_model.load_state_dict(new_dict, strict=False)
        print(f'load complete, unload_keys: {unload_keys}, other_keys: {other_keys}')
        
    return graphormer_model
    

def ReactionClassificationHead_setup(graphormer_model, model_config, cfg, device):
    linear_model = ReactionClassificationHead(graphormer_model, model_config, output_size=1000).to(device)
    return linear_model

def train_epoch(cfg, model, device, train_loader, optimizer, epoch):
    model.train()
    output_acc = 0
    train_loss_accum = 0
    accuracy_accum = 0
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
#        if step > 10:
#            break
        for key in batch['graph_rc']:
            batch['graph_rc'][key] = batch['graph_rc'][key].to(device)
        for key in batch['graph_pd']:
            batch['graph_pd'][key] = batch['graph_pd'][key].to(device)
        batch['y'] = batch['y'].to(torch.int64)
        batch['y'] = batch['y'].to(device)
        
        # Change batch['y'] to int type tensors
        
        

        pred = model(batch)
        
#        print(pred.shape, batch['y'].shape)
        loss = criterion(pred, batch['y'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = compute_accuracy(pred, batch['y'])
        train_loss_accum += float(loss.detach().cpu().item())
        accuracy_accum += acc
        output_acc += acc
        if not cfg.training_settings.testing_stage:
            if (step + 1) % cfg.training_settings.log_train_freq == 0:
                wandb.log({
                    "epoch": epoch,
                    "traing_loss_accum": train_loss_accum / cfg.training_settings.log_train_freq,
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_accuracy_accum": accuracy_accum / cfg.training_settings.log_train_freq,
                    
                })
                
                print("traing_loss_accum: ", train_loss_accum / cfg.training_settings.log_train_freq,
                        "\ntrain_loss: ", float(loss.detach().cpu().item()),
                        "\ntrain_accuracy_accum: ", accuracy_accum / cfg.training_settings.log_train_freq)
                
                train_loss_accum = 0
                accuracy_accum = 0
    return output_acc / (step+1)
#    print(train_loss_accum, accuracy_accum)

def eval_epoch(cfg, model, device, val_loader, val=True):
    model.eval()
    eval_loss_accum = 0
    accuracy_accum = 0
    for step, batch in enumerate(tqdm(val_loader, desc="Iteration")):
#        if step > 10:
#            break
        for key in batch['graph_rc']:
            batch['graph_rc'][key] = batch['graph_rc'][key].to(device)
        for key in batch['graph_pd']:
            batch['graph_pd'][key] = batch['graph_pd'][key].to(device)
        batch['y'] = batch['y'].to(torch.int64)
        batch['y'] = batch['y'].to(device)
        pred = model(batch)
        
        loss = criterion(pred, batch['y'])

        acc = compute_accuracy(pred, batch['y'])
        
        eval_loss_accum += float(loss.detach().cpu().item())
        accuracy_accum += acc
        
    if not cfg.training_settings.testing_stage:
        if not val:
            wandb.log({
            "testing_loss": eval_loss_accum / (step+1),
            "testing_accuracy": accuracy_accum / (step+1),
            
        })
            print("testing_loss: ", eval_loss_accum / (step+1),
                  "\ntesting_accuracy: ", accuracy_accum / (step+1))
        else:
            wandb.log({
            "validation_loss": eval_loss_accum / (step+1),
            "validation_accuracy": accuracy_accum / (step+1),
            })
            print("validation_loss: ", eval_loss_accum / (step+1),
                  "\nvalidation_accuracy: ", accuracy_accum / (step+1))
    
    return accuracy_accum / (step+1)

def trainning(cfg, model, device, dataloader, epoch_c):
    train_laoder, val_loader, test_loader = dataloader
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    
    
    
    model_param_group = []
    model_param_group.append({"params": model.graphormer.parameters()})
    model_param_group.append({"params": model.linear.parameters(), 
                              "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    
    for epoch in range(epoch_c, cfg.training_settings.epochs+1):
        print("====epoch " + str(epoch))
        
        train_acc = train_epoch(cfg, model, device, train_laoder, optimizer, epoch)
        train_acc_list.append(train_acc)
        print("====Evaluation")
        
        val_acc = eval_epoch(cfg, model, device, val_loader, val=True)
        val_acc_list.append(val_acc)
        
        test_acc = eval_epoch(cfg, model, device, test_loader, val=False)
        test_acc_list.append(test_acc)
        
        if cfg.training_settings.intermediate_model:
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, "reaction_classification_model_epoch{}.pt".format(epoch)))
        
        
        
        # val_epoch(cfg, model, device, val_loader)
        # test_epoch(cfg, model, device, test_loader)
    
    
    return train_acc_list, val_acc_list, test_acc_list


@hydra.main(version_base=None, config_path="/conf", config_name="finetune")
def main(cfg):
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)

    if not cfg.training_settings.testing_stage:
        wandb.login(key=cfg.wandb.login_key)
        wandb.init(project="Chemical-Reaction-Pretraining", name=cfg.wandb.run_name+str(cfg.training_settings.runseed))
    
    config_file = cfg.model.graphormer_config_yaml
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
        
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    model_config = Struct(**model_config)
    
    train_dataset = pd.read_csv(USPTO_CONFIG.uspto_1k_tpl_train, delimiter='\t')
    test_dataset = pd.read_csv(USPTO_CONFIG.uspto_1k_tpl_test, delimiter='\t')
#    val_dataset = torch.util.data.random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset)-int(len(train_dataset)*0.9)])[1]
    # --TODO: add the graphormer dataset here
    graphormer_trainset = GraphormerDFMaker(cfg, train_dataset)
    graphormer_valset = torch.utils.data.random_split(graphormer_trainset, [int(len(graphormer_trainset)*0.98), len(graphormer_trainset)-int(len(graphormer_trainset)*0.98)])[1]
    graphormer_testset = GraphormerDFMaker(cfg, test_dataset)
    train_loader, val_loader, test_loader = GraphormerDFMaker2(graphormer_trainset, graphormer_valset, graphormer_testset, cfg, model_config)
    # --TODO: add the graphormer dataloader here
    # train_loader, val_loader, test_loader 
    if cfg.training_settings.continue_training:
        graphormer_encoder, intermediate_path, epoch_c = Graphormer_setup(cfg, model_config, device)
    else:
        graphormer_encoder = Graphormer_setup(cfg, model_config, device)
        epoch_c = 1
    reaction_classification_head = ReactionClassificationHead_setup(graphormer_encoder, model_config, cfg, device)
    if cfg.training_settings.continue_training:
        reaction_classification_head.load_state_dict(torch.load(intermediate_path))
    train_acc_list, val_acc_list, test_acc_list = trainning(cfg, reaction_classification_head, device, (train_loader, val_loader, test_loader), epoch_c)
    # model load dict
    # --TODO: model trainning here
    
    if not cfg.training_settings.testing_stage:
        # Save the model
        torch.save(reaction_classification_head.state_dict(), os.path.join(wandb.run.dir, "reaction_classification_model.pt"))
        wandb.finish()
    
    print(f"train: {train_acc_list[-1]} val: {val_acc_list[-1]} test: {test_acc_list[-1]}")
#    max_ind = np.argmax(val_acc_list)
#    print(f"The best epoch: {max_ind}\ntrain: {train_acc_list[max_ind]} val: {val_acc_list[max_ind]} test: {test_acc_list[max_ind]}")
    
    
if __name__ == "__main__":
    main()