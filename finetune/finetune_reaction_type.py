import sys
import os
os.chdir("/finetune/")
sys.path.insert(0,'..')
from test.gcn_utils.datas import MoleculeDataset, mol_to_graph_data_obj_simple
from torch.utils.data import DataLoader
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
from data_processing import USPTO_1k_TPL, USPTO1kTPLCollator

criterion = nn.CrossEntropyLoss()

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

class PredictionHead(torch.nn.Module):
    def __init__(self, input_size, output_size, emb_dim=2048):
        super(PredictionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, emb_dim)
        self.fc2 = nn.Linear(emb_dim, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
        
class ReactionClassificationHead(torch.nn.Module):
    """_summary_
    Args:
        torch (_type_): _description_
    """
    def __init__(self, gnn, pool="sum"):
        super(ReactionClassificationHead, self).__init__()
        self.gnn = gnn
        self.pool = pool
    def forward(self, batch_input, device):
        reaction_readout = []
        
        batch_graph = batch_input['graph'].to(device)
        
        batch_size = len(batch_input['mol_idx'])
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        
        for batch_num in range(batch_size):
            node_rep_single = node_rep[batch_graph.batch==batch_num]
            # For mol_idx, 0 stands for reactants and 1 stands for products
            reactant = node_rep_single[batch_input['mol_idx'][batch_num]==0]
            product = node_rep_single[batch_input['mol_idx'][batch_num]==1]
            if self.pool == 'sum':
                reactant_pool = torch.sum(reactant, dim=0, keepdim=True)
                product_pool = torch.sum(product, dim=0, keepdim=True)
            elif self.pool == 'mean':
                reactant_pool = torch.mean(reactant, dim=0, keepdim=True)
                product_pool = torch.mean(product, dim=0, keepdim=True)
            else:
                return None
            
            reaction_vector = torch.cat((reactant_pool, product_pool), dim=0)
            reaction_vector = reaction_vector.view(-1)
            
            reaction_readout.append(reaction_vector)
            
        return torch.stack(reaction_readout), torch.tensor(batch_input['graph'].y).to(device)



def train(cfg, model, device, loader, optimizer, epoch):
    readout_model, pred_head = model
    
    readout_model.train()
    pred_head.train()
    
    train_loss_accum = 0
    accuracy_accum = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        react_emb, label = readout_model(batch, device)
        pred = pred_head(react_emb)
        
        train_loss = criterion(pred.double(), label)
        acc_step = compute_accuracy(pred.double(), label)
        
        accuracy_accum += acc_step
        train_loss_accum += float(train_loss.cpu().item())
        
        optimizer.zero_grad()
        
        train_loss.backward()
        
        optimizer.step()
        if not cfg.training_settings.testing_stage:
            if (step + 1) % cfg.training_settings.log_train_freq == 0:
                wandb.log({
                    "epoch": epoch,
                    "traing_loss_accum": train_loss_accum / cfg.training_settings.log_train_freq,
                    "train_loss": train_loss,
                    "train_accuracy_accum": accuracy_accum / cfg.training_settings.log_train_freq,
                    
                })
                train_loss_accum = 0
                accuracy_accum = 0
            

def eval(cfg, model, device, loader, epoch, stage):
    readout_model, pred_head = model
    readout_model.eval()
    pred_head.eval()
    
    eval_loss_accum = 0
    accuracy_accum = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        react_emb, label = readout_model(batch, device)
        pred = pred_head(react_emb)
        
        loss = criterion(pred.double(), label)
        acc_step = compute_accuracy(pred.double(), label)
        
        eval_loss_accum += float(loss.cpu().item())
        accuracy_accum += acc_step
        
    if not cfg.training_settings.testing_stage:
        wandb.log({
            "epoch": epoch,
            f"{stage} eval loss": eval_loss_accum / (step+1),
            f"{stage} eval accuracy": accuracy_accum / (step+1),
            
        })
        
    return accuracy_accum / (step+1)

@hydra.main(version_base=None, config_path="/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/conf", config_name="finetune")
def main(cfg):
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)
    
    if not cfg.training_settings.testing_stage:
        wandb.login(key=cfg.wandb.login_key)
        wandb.init(project="Chemical-Reaction-Pretraining", name=cfg.wandb.run_name+str(cfg.training_settings.runseed))
        
    train_dataset = pd.read_csv(USPTO_CONFIG.uspto_1k_tpl_train, delimiter='\t')
    
    test_dataset = pd.read_csv(USPTO_CONFIG.uspto_1k_tpl_test, delimiter='\t')
    
    train_dataset = USPTO_1k_TPL(train_dataset)
    
    if not cfg.training_settings.testing_stage:
        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size]) 
        test_dataset = USPTO_1k_TPL(test_dataset)
    else:
        
        val_size = cfg.training_settings.testing_val_size
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        test_dataset = val_dataset
        train_dataset = val_dataset 
    
    # ===== Model Loading & Finetuning
    collate_fn = USPTO1kTPLCollator()
    train_loader = DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn=collate_fn)
    
    #set up model
    model = GNN(cfg.model.num_layer, cfg.model.emb_dim, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, gnn_type = cfg.model.gnn_type)
    
    # model = GNN_graphpred(cfg.model.num_layer, cfg.model.emb_dim, 1, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, graph_pooling = cfg.model.graph_pooling, gnn_type = cfg.model.gnn_type)
    if cfg.model.input_model_file:
        model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))
        print(f"Load from a pretrained model {cfg.model.input_model_file}")
#         model.from_pretrained(cfg.model.input_model_file)
    readout_model = ReactionClassificationHead(model, pool=cfg.model.pool_method).to(device)
    prediction_head = PredictionHead(input_size=cfg.model.emb_dim * 2, output_size=1000).to(device)
    model = [readout_model, prediction_head]
     #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": readout_model.parameters()})
    
    model_param_group.append({"params": prediction_head.parameters(), "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    print(optimizer)
    
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    
    for epoch in range(1, cfg.training_settings.epochs+1):
        
        print("====epoch " + str(epoch))
        
        train(cfg, model, device, train_loader, optimizer, epoch)

        print("====Evaluation")
        if cfg.training_settings.eval_train:
            train_acc = eval(cfg, model, device, train_loader, epoch, stage="train")
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(cfg, model, device, val_loader, epoch, stage='valid')
        test_acc = eval(cfg, model, device, test_loader, epoch, stage='test')

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
    if not cfg.training_settings.test_stage:
        wandb.finish()
    max_ind = np.argmax(val_acc_list)
    print(f"The best epoch: {max_ind}\ntrain: {train_acc_list[max_ind]} val: {val_acc_list[max_ind]} test: {test_acc_list[max_ind]}")
    
    
if __name__ == "__main__":
    main()
    