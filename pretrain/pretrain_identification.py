import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
# sys.path.append(sys.path[0]+'/..')
import os
# os.chdir("/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/")
sys.path.insert(0,'..')
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN

from sklearn.metrics import roc_auc_score
import hydra
import pandas as pd
from torchvision import ops
from data_processing import ReactionDataset, CustomCollator, IdentificationCollator


import timeit

criterion = nn.BCELoss()
    
class ReactantCentreIdentify(torch.nn.Module):
    def __init__(self, gnn):
        super(ReactantCentreIdentify, self).__init__()
        self.gnn = gnn
        
    def forward(self, batch_input, device):
        batch_size = batch_input['graph'].idx.shape[0]
        # print(batch_input['graph'].idx.get_device())
        batch_graph = batch_input['graph'].to(device)
        # print(batch_graph.x.get_device())
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # print(node_rep.get_device())
        tensor_2_head = torch.zeros((node_rep.size(0), node_rep.size(1) * 2), dtype=node_rep.dtype, device=node_rep.device)
        # batch_size = batch_input['graph'].idx.detach().cpu().shape[0]
        for batch_num in range(batch_size):
            
            node_rep_single = node_rep[batch_graph.batch.detach().cpu()==batch_num]
            
            node_class_single = batch_input['primary_label'][batch_graph.batch.detach().cpu()==batch_num]
            
            if node_class_single[-1] == -1: # The last atom is from condition reactants, means the reaction has condition reactants
                cond_rep = node_rep_single[node_class_single==-1]
                cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            else:
                cond_pool = torch.zeros((1, node_rep.size(1)), dtype=node_rep.dtype, device=node_rep.device)
                
            tensor_2_head[batch_graph.batch.detach().cpu()==batch_num] = torch.cat([node_rep_single, torch.broadcast_to(cond_pool, node_rep_single.shape)], dim=1)
                
        return batch_input['graph'], tensor_2_head, batch_input['primary_label']
    
    
def compute_accuracy(pred, target):
    # print(pred, target)
    return float(torch.sum((pred.detach() > 0.5) == target).cpu().item()) / len(pred)
    # return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train_identification(cfg, model_list, loader, optimizer_list, device):
    train_loader, val_loader = loader
    model, linear_pred_atoms = model_list
    optimizer_model, optimizer_linear_pred_atoms = optimizer_list
    
    
        
    
    model.train()
    linear_pred_atoms.train()
    
    loss_accum = 0
    acc_node_accum = 0
    valid_loss_accum = 0
    acc_node_valid_accum = 0
    
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        temp_graph, node_rep, atom_label = model(batch, device)
        
        pred_node = linear_pred_atoms(node_rep[atom_label >= 0])
        pred_node = torch.nn.Sigmoid()(pred_node)
        tgt = atom_label[atom_label >= 0].reshape(pred_node.shape).to(device)
        # print(pred_node.double().shape, tgt.shape)
        if cfg.training_settings.focal_loss:
            
            loss = criterion(pred_node.double(), tgt)
            # print(loss)
            # print('---------')
            loss = ops.sigmoid_focal_loss(pred_node.double(), tgt, alpha=cfg.training_settings.focal_loss_alpha, gamma=cfg.training_settings.focal_loss_gamma).mean()
            # print(loss)
            # break
        else:
            loss = criterion(pred_node.double(), tgt)
        # print(loss)
        # print(pred_node.double(), tgt)
        # print(pred_node.shape, tgt.shape)
        acc_node = compute_accuracy(pred_node, tgt)
        
        acc_node_accum += acc_node
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        
        loss.backward() 
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        
        loss_accum += float(loss.cpu().item())
        
        #if step % 5000 == 1 and not cfg.training_settings.testing_stage:
        #    wandb.log({
        #         "train_loss": loss,
        #         "train_loss_accum": loss_accum/step,
        #         "train_acc": acc_node,
        #         "train_acc_accum": acc_node_accum/step,
        #     }, commit=False, step=step)
        
    model.eval()
    linear_pred_atoms.eval()
    
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        temp_graph, node_rep, atom_label = model(valid_batch, device)
        pred_node = linear_pred_atoms(node_rep[atom_label >= 0])
        pred_node = torch.nn.Sigmoid()(pred_node)
        tgt = atom_label[atom_label >= 0].reshape(pred_node.shape).to(device)
        valid_loss = criterion(pred_node.double(), tgt)
        acc_node_valid = compute_accuracy(pred_node, tgt)
        # print(acc_node_valid)
        acc_node_valid_accum += acc_node_valid
        
        valid_loss_accum += float(valid_loss.cpu().item())
        
    if not cfg.training_settings.testing_stage:
        wandb.log({
                "train_loss": loss,
                "train_loss_accum": loss_accum/step,
                "train_acc": acc_node,
                "train_acc_accum": acc_node_accum/step,
                "valid_loss": valid_loss,
                "valid_loss_accum": valid_loss_accum/valid_step,
                "valid_acc": acc_node_valid,
                "valid_acc_accum": acc_node_valid_accum/valid_step
            })
    
    return loss_accum/step, acc_node_accum/step, valid_loss_accum/valid_step, acc_node_valid_accum/valid_step

@hydra.main(version_base=None, config_path="/conf", config_name="config")
def main(cfg):
    
    
    
    torch.manual_seed(cfg.training_settings.seed)
    np.random.seed(cfg.training_settings.seed)
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.seed)
    if not cfg.training_settings.testing_stage:
        wandb.login(key=cfg.wandb.login_key)
        wandb.init(project="Chemical-Reaction-Pretraining", name=cfg.wandb.run_name)
    model = GNN(cfg.model.num_layer, cfg.model.emb_dim, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, gnn_type = cfg.model.gnn_type)
    if cfg.training_settings.stage2_on:
        model.load_state_dict(torch.load(cfg.training_settings.stage_one_model))
        print("Train from model {}".format(cfg.training_settings.stage_one_model))
    else:
        print("Train from scratch")
    
    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    
    val_size = cfg.training_settings.validation_size
    if cfg.training_settings.testing_stage:
        train_size = cfg.training_settings.validation_size
        test_size = len(reaction_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = torch.utils.data.random_split(reaction_dataset, [train_size, val_size, test_size])
    else:
        train_size = len(reaction_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(reaction_dataset, [train_size, val_size])
    
    collator_identification = IdentificationCollator(mask_stage="reaction_centre")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_identification)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_identification)
    
    data_loader = [train_loader, val_loader]
    
    model_identify = ReactantCentreIdentify(model).to(device)
    
    linear_pred_atoms = torch.nn.Linear(cfg.model.emb_dim * 2, 1).to(device)
    
    model_list = [model_identify, linear_pred_atoms]
    
    # set up optimizers
    
    optimizer_model = optim.Adam(model_identify.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms]
    
    print("=====Stage2: Reaction Centre Identification")
        
    for epoch in range(cfg.training_settings.epochs_start, cfg.training_settings.epochs_RC+cfg.training_settings.epochs_start):
        print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
        
        train_loss, train_acc, val_loss, val_acc = train_identification(cfg, model_list, data_loader, optimizer_list, device)
        
        print(train_loss, train_acc, val_loss, val_acc)
        
        if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage and epoch % 20 == 1 and epoch != 1:
           torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_{}_reaction_centre_identification.pth".format(cfg.model.target, epoch)) 
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
           torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_{}_reaction_centre_identification.pth".format(cfg.model.target, epoch))
        
        
if __name__ == "__main__":
    main()