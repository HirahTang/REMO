import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
# sys.path.append(sys.path[0]+'/..')
import os
os.chdir("/pretrain/")
sys.path.insert(0,'..')
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN

from sklearn.metrics import roc_auc_score
import hydra
import pandas as pd
from data_processing import ReactionDataset, CustomCollator
from torch.utils.data import Dataset, ConcatDataset
criterion = nn.CrossEntropyLoss()

import timeit
class ReactantStage2(torch.nn.Module):
    """_summary_
    Args:
        torch (_type_): _description_
    """
    def __init__(self, gnn):
        super(ReactantStage2, self).__init__()
        self.gnn = gnn
        
    def forward(self, batch_input, device):
        batch_graph = batch_input['graph'].to(device)
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # print(node_rep.shape)
        # print(torch.isnan(node_rep).sum())
        # tensor_2_head = torch.tensor([]).to(device)
        

        tensor_2_head = torch.zeros((node_rep.size(0), node_rep.size(1) * 2), dtype=node_rep.dtype, device=node_rep.device)
        batch_size = len(batch_input['primary_idx'])
        
        for batch_num in range(batch_size):
            # node_rep_single = torch.index_select(node_rep, 0, torch.LongTensor([i for i, x in enumerate(batch_graph.batch) if x == batch_num]).to(device))
            node_rep_single = node_rep[batch_graph.batch==batch_num]
            
            pri_num = max(batch_input['primary_idx'][batch_num]) + 1
            # cond_rep = node_rep_single
            atom_num = (batch_graph.batch==batch_num).sum().item()
            
            if pri_num < atom_num: # has conditional mol
                cond_rep = node_rep_single[pri_num:]
                cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            else: # no conditional mol
                cond_pool = torch.zeros((1, node_rep.size(1)), dtype=node_rep.dtype, device=node_rep.device)
            # print(pri_num, atom_num, batch_input['primary_idx'], node_rep_single.shape)
            # concat
            tensor_2_head[batch_graph.batch==batch_num] = torch.cat([node_rep_single, torch.broadcast_to(cond_pool, node_rep_single.shape)], dim=1)
            # print(tensor_2_head.shape)
            
            # cond_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(batch_input['condition_idx'][batch_num]).to(device))
            
            # pri_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(batch_input['primary_idx'][batch_num]).to(device))
            # # print(torch.isnan(pri_rep).sum(), torch.isnan(cond_rep).sum(), torch.isnan(node_rep_single).sum())
            # cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            # # if torch.isnan(cond_pool).sum() > 0:
            # #     print(cond_pool, cond_rep, batch_input['condition_idx'][batch_num])
            # # print(torch.isnan(cond_pool).sum())
            # if len(batch_input['condition_idx'][batch_num]) > 0:
            #     new_rep = torch.cat([pri_rep, torch.broadcast_to(cond_pool, pri_rep.shape)], dim=1)
            # # print(torch.isnan(new_rep).sum())
            # else:
            #     new_rep = torch.cat([pri_rep, torch.broadcast_to(torch.zeros_like(cond_pool), pri_rep.shape)], dim=1)
            # tensor_2_head = torch.cat([tensor_2_head, new_rep, torch.zeros(cond_rep.shape[0], new_rep.shape[1]).to(device)], dim=0)
            # print(torch.isnan(tensor_2_head).sum())
        # print(tensor_2_head.shape)
        
        # print(torch.isnan(tensor_2_head).sum())
        return batch_input['graph'], tensor_2_head, batch_input['mlabes']
    
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train_NodeEdge(cfg, model_list, loader, optimizer_list, device):
    train_loader, val_loader = loader
    model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    linear_pred_mlabes.train()
    
    loss_accum = 0
    valid_loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    acc_mlabes_accum = 0
    acc_node_valid_accum = 0
    acc_edge_valid_accum = 0
    acc_mlabes_valid_accum = 0
    
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        temp_graph, node_rep, mlabes = model(batch, device)
        
        ## loss for nodes
        
        pred_node = linear_pred_atoms(node_rep[temp_graph.masked_atom_indices])
        # tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        loss_node = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        
        acc_node = compute_accuracy(pred_node, temp_graph.mask_node_label[:,0])
        acc_node_accum += acc_node
         # Edge prediction
        masked_edge_index = temp_graph.edge_index[:, temp_graph.connected_edge_indices]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear_pred_bonds(edge_rep)
        loss_edge = criterion(pred_edge.double(), temp_graph.mask_edge_label[:,0])
        loss = loss_node + loss_edge

        acc_edge = compute_accuracy(pred_edge, temp_graph.mask_edge_label[:,0])
        acc_edge_accum += acc_edge
        
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()
        
        loss_accum += float(loss.cpu().item())

    #    if step % 5000 == 1 and not cfg.training_settings.testing_stage:
    #        wandb.log({
    #           "train_loss": loss,
    #            "train_loss_accum": loss_accum/step,
    #            "train_node_acc": acc_node,
    #            "train_node_acc_accum": acc_node_accum/step,
    #            "train_edge_acc": acc_edge,
    #            "train_edge_acc_accum": acc_edge_accum/step
    #        }, commit=False)
    model.eval()
    linear_pred_atoms.eval()
    linear_pred_bonds.eval()
    linear_pred_mlabes.eval()
    
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        temp_graph, node_rep, mlabes = model(valid_batch, device)
        
        ## loss for nodes
        
        pred_node = linear_pred_atoms(node_rep[temp_graph.masked_atom_indices])
        # tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        valid_loss_node = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        
        valid_acc_node = compute_accuracy(pred_node, temp_graph.mask_node_label[:,0])
        acc_node_valid_accum += valid_acc_node
         # Edge prediction
        masked_edge_index = temp_graph.edge_index[:, temp_graph.connected_edge_indices]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear_pred_bonds(edge_rep)
        valid_loss_edge = criterion(pred_edge.double(), temp_graph.mask_edge_label[:,0])
        valid_loss = valid_loss_node + valid_loss_edge

        valid_acc_edge = compute_accuracy(pred_edge, temp_graph.mask_edge_label[:,0])
        acc_edge_valid_accum += valid_acc_edge
        valid_loss_accum += float(valid_loss.cpu().item())
        
    if not cfg.training_settings.testing_stage:
        wandb.log({
                "train_loss": loss,
                "train_loss_accum": loss_accum/step,
                "train_node_acc": acc_node,
                "train_node_acc_accum": acc_node_accum/step,
                "train_edge_acc": acc_edge,
                "train_edge_acc_accum": acc_edge_accum/step,
                "valid_loss": valid_loss,
                "valid_loss_accum": valid_loss_accum/valid_step,
                "valid_node_acc": valid_acc_node,
                "valid_edge_acc": valid_acc_edge,
                "valid_node_acc_accum": acc_node_valid_accum/valid_step,
                "valid_edge_acc_accum": acc_edge_valid_accum/valid_step
            })
    
    return loss_accum/step, acc_node_accum/step, acc_edge_accum/step, valid_loss_accum/valid_step, acc_node_valid_accum/valid_step, acc_edge_valid_accum/valid_step
    
def train_mlabes(cfg, model_list, loader, optimizer_list, device):
    train_loader, val_loader = loader
    model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    linear_pred_mlabes.train()
    
    loss_accum = 0
    valid_loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    acc_mlabes_accum = 0
    acc_node_valid_accum = 0
    acc_edge_valid_accum = 0
    acc_mlabes_valid_accum = 0
    
    
    
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        temp_graph, node_rep, mlabes = model(batch, device)
        
        ## loss for nodes
        
        pred_node = linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        loss = criterion(pred_node.double(), tgt)
        
        acc_mlabes = compute_accuracy(pred_node, tgt)
        acc_mlabes_accum += acc_mlabes

        optimizer_model.zero_grad()
        optimizer_linear_pred_mlabes.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_mlabes.step()
        loss_accum += float(loss.cpu().item())
        
    #    if step % 5000 == 1 and not cfg.training_settings.testing_stage:
    #        wandb.log({"train_loss": loss,
    #           "train_loss_accum": loss_accum/step,
    #           "train_Mlabes_acc": acc_mlabes,
    #           "train_Mlabes_acc_accum": acc_mlabes_accum/step}, commit=False)
        # if step > 300:
        #     break
    
    model.eval()
    linear_pred_atoms.eval()
    linear_pred_bonds.eval()
    linear_pred_mlabes.eval()
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        temp_graph, node_rep, mlabes = model(valid_batch, device)
        pred_node = linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        tgt = torch.tensor([mlabe for i, mlabe in enumerate(valid_batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        valid_loss = criterion(pred_node.double(), tgt)
        acc_mlabes_valid = compute_accuracy(pred_node, tgt)
        acc_mlabes_valid_accum += acc_mlabes_valid
        valid_loss_accum += float(valid_loss.cpu().item())
    if not cfg.training_settings.testing_stage:
        wandb.log({"train_loss": loss,
               "train_loss_accum": loss_accum/step,
               "train_Mlabes_acc": acc_mlabes,
               "train_Mlabes_acc_accum": acc_mlabes_accum/step,
               "valid_loss": valid_loss,
               "valid_loss_accum": valid_loss_accum/valid_step,
               "valid_Mlabes_acc": acc_mlabes_valid,
               "valid_Mlabes_acc_accum": acc_mlabes_valid_accum/valid_step
                   })
    return loss_accum/step, acc_mlabes_accum/step, valid_loss_accum/valid_step, acc_mlabes_valid_accum/valid_step 

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
        model.load_state_dict(torch.load(cfg.training_settings.stage_one_model, map_location=device))
        print("Train from model {}".format(cfg.training_settings.stage_one_model))
    else:
        print("Train from scratch")
    
    # reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    
    # val_size = cfg.training_settings.validation_size
    # if cfg.training_settings.testing_stage:
    #     train_size = cfg.training_settings.validation_size
    #     test_size = len(reaction_dataset) - train_size - val_size
    #     train_dataset, val_dataset, _ = torch.utils.data.random_split(reaction_dataset, [train_size, val_size, test_size])
    # else:
    #     train_size = len(reaction_dataset) - val_size
    #     train_dataset, val_dataset = torch.utils.data.random_split(reaction_dataset, [train_size, val_size])
    
    train_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_train, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    train_dataset2 = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_valid, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    val_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_test, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    train_dataset = ConcatDataset([train_dataset, train_dataset2])
    
    if cfg.training_settings.testing_stage:
        train_size = cfg.training_settings.validation_size
        val_size = cfg.training_settings.validation_size
        test_size = len(train_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size])
    
    
    collator_rc = CustomCollator(mask_stage='reaction_centre')
    collator_1hop = CustomCollator(mask_stage='one_hop')
    collator_2hop = CustomCollator(mask_stage='two_hop')
    collator_3hop = CustomCollator(mask_stage='three_hop')
    
    data_loader_train_rc = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_rc)
    data_loader_val_rc = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_rc)
    data_loader_rc = [data_loader_train_rc, data_loader_val_rc]
    
    data_loader_train_1hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_1hop)
    data_loader_val_1hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_1hop)
    data_loader_1hop = [data_loader_train_1hop, data_loader_val_1hop]
    
    data_loader_train_2hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_2hop)
    data_loader_val_2hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_2hop)
    data_loader_2hop = [data_loader_train_2hop, data_loader_val_2hop]
    
    data_loader_train_3hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_3hop)
    data_loader_val_3hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_3hop)
    data_loader_3hop = [data_loader_train_3hop, data_loader_val_3hop]
    
    model_react = ReactantStage2(model).to(device)
    linear_pred_atoms = torch.nn.Linear(cfg.model.emb_dim * 2, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(cfg.model.emb_dim * 2, 4).to(device)
    linear_pred_mlabes = torch.nn.Linear(cfg.model.emb_dim * 2, 2458).to(device)
    
    model_list = [model_react, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes]
    
    #set up optimizers
    optimizer_model = optim.Adam(model_react.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_mlabes = optim.Adam(linear_pred_mlabes.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes]
    
    print("=====Stage2: Phase 1 - original reaction centre masking")
    for epoch in range(1, cfg.training_settings.epochs_RC+1):
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_list, data_loader_rc, optimizer_list, device)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)
        
        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_rc, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
        else:
            print("Wrong Target Setting")
    
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_rc.pth".format(cfg.model.target))
       
    print("=====Stage2: Phase 2 - One Hop neighbours masking")
    for epoch in range(1, cfg.training_settings.epochs_1hop+1):
        
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_list, data_loader_1hop, optimizer_list, device)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)

        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_1hop, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
            
        else:
            print("Wrong Target Setting")
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_one_hop.pth".format(cfg.model.target))
       
    print("=====Stage2: Phase 3 - Two Hop neighbours masking")
    for epoch in range(1, cfg.training_settings.epochs_2hop+1):
        
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_list, data_loader_2hop, optimizer_list, device)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)

        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_2hop, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
            
        else:
            print("Wrong Target Setting")
            
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_two_hop.pth".format(cfg.model.target))
    
    print("=====Stage2: Phase 4 - Three Hop neighbours masking")
    for epoch in range(1, cfg.training_settings.epochs_3hop+1):
        
    
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_list, data_loader_3hop, optimizer_list, device)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)
    
        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_3hop, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
            
        else:
            print("Wrong Target Setting")
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_three_hop.pth".format(cfg.model.target))
    """
    for step, batch in tqdm(enumerate(data_loader)):
        # batch = batch.to(device)
        temp_graph, node_rep, mlabes = model(batch, device)
        print(temp_graph)
        print(node_rep.shape)
        print(len(mlabes))
        print(temp_graph.masked_atom_indices)
        # print(temp_graph.mask_node_label)
        # print(temp_graph.mask_edge_label)
        # print(mlabes[temp_graph.masked_atom_indices])
        print([mlabe for i, mlabe in enumerate(mlabes) if i in temp_graph.masked_atom_indices])
        print(temp_graph.x[temp_graph.masked_atom_indices])
        if step > 0:
            break
    """
if __name__ == "__main__":
    main()