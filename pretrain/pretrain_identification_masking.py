# python -u pretrain_identification_masking.py --config-name indentifi_mlabel_graphmae training_settings.gnn_decoder=false training_settings.adv=true model.output_model_file=/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/indentifi_mlabel_graphmae_focal_loss_multi_task_adv/model wandb.run_name=indentifi_mlabel_graphmae_focal_loss_multi_task_adv > indentifi_mlabel_graphmae_focal_loss_multi_task_adv.log

# python -u pretrain_identification_masking.py --config-name indentifi_mlabel_graphmae training_settings.gnn_decoder=false model.output_model_file=/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/indentifi_mlabel_graphmae_focal_loss_multi_task/model wandb.run_name=indentifi_mlabel_graphmae_focal_loss_multi_task > indentifi_mlabel_graphmae_focal_loss_multi_task.log


# python -u pretrain_identification_masking.py --config-name indentifi_mlabel_graphmae training_settings.gnn_decoder=false training_settings.adv=true model.output_model_file=/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/indentifi_mlabel_graphmae_focal_loss_multi_task_adv_brics_plus/model wandb.run_name=indentifi_mlabel_graphmae_focal_loss_multi_task_adv_brics_plus training_settings.use_frag=brics_plus > indentifi_mlabel_graphmae_focal_loss_multi_task_adv_brics_plus.log

# python -u pretrain_identification_masking.py --config-name indentifi_mlabel_graphmae training_settings.gnn_decoder=false model.output_model_file=/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/indentifi_mlabel_graphmae_focal_loss_multi_task_brics_plus/model wandb.run_name=indentifi_mlabel_graphmae_focal_loss_multi_task_brics_plus training_settings.use_frag=brics_plus  > indentifi_mlabel_graphmae_focal_loss_multi_task_brics_plus.log

# training_settings.use_frag=brics_plus

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
# sys.path.append(sys.path[0]+'/..')
import os

sys.path.insert(0,'..')
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from test.gcn_utils.datas import MoleculeDataset, DataLoaderMasking, MaskAtom
from test.gcn_utils.adv import PGD

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN, GNNDecoder

from sklearn.metrics import roc_auc_score
import hydra
import pandas as pd
from data_processing import ReactionDataset, CustomCollator, MaskIndentiCollator
from torchvision import ops

criterion = nn.CrossEntropyLoss()
criterion_indenti = nn.BCELoss()

import timeit
class ReactantStage2(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, gnn, emb_size, stage1_on=False, stage_one_head_model=None, gnn_decoder=None):
        super(ReactantStage2, self).__init__()
        self.gnn = gnn
        self.emb_size = emb_size
        # self.linear_pred_atoms = torch.nn.Linear(emb_size * 2, 119)
        # self.linear_pred_bonds = torch.nn.Linear(emb_size * 2, 4)
        self.linear_pred_mlabes = torch.nn.Linear(emb_size * 2, 2458) # todo, why one hop, two hop are still 2458

        self.linear_pred_atoms = torch.nn.Linear(emb_size * 2, 1)

        if stage1_on:
            self.org_linear_pred_atoms = torch.nn.Linear(emb_size, 119)
            self.org_linear_pred_atoms.load_state_dict(torch.load(stage_one_head_model))
            # self.org_linear_pred_bonds = torch.nn.Linear(emb_size, 4)
        
        self.gnn_decoder = gnn_decoder
    
    def forward_single(self, batch_input, device):
        batch_size = batch_input['graph'].idx.shape[0]
        # print(batch_input['graph'].idx.get_device())
        batch_graph = batch_input['graph'].to(device)
        # print(batch_graph.x.get_device())
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # print(node_rep.get_device())

        if self.gnn_decoder:
            node_rep = self.gnn_decoder(node_rep, batch_graph.edge_index, batch_graph.edge_attr)
        
        
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
                
        return batch_input['graph'], tensor_2_head
        # batch_input['mlabes']

    def forward(self, batch_input, device):
        mask_input_graph, mask_head = self.forward_single(batch_input[0], device)
        indenti_input_graph, indenti_head = self.forward_single(batch_input[1], device)
        return mask_input_graph, mask_head, batch_input[0]['mlabes'], indenti_input_graph, indenti_head, batch_input[1]['primary_label']
    
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def compute_accuracy_indenti(pred, target):
    # print(pred, target)
    return float(torch.sum((pred.detach() > 0.5) == target).cpu().item()) / len(pred)

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
    
def train_mlabes(cfg, model_react, loader, optimizer, device, org_dataloader_iterator=None, data_loader_org=None, adv=None):
    train_loader, val_loader = loader
    
    # optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes = optimizer_list[:4]

    # model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes = model_list[:4]
    # # model.train()
    # # linear_pred_atoms.train()
    # # linear_pred_bonds.train()
    # # linear_pred_mlabes.train()
    # for model in model_list:
    #     model.train()
    model_react.train()


    # if org_dataloader_iterator is not None:
    #     org_linear_pred_atoms, org_linear_pred_bonds = model_list[-2:]
    #     org_optimizer_linear_pred_atoms, org_optimizer_linear_pred_bonds = optimizer_list[-2:]
        
    
    loss_accum = 0
    valid_loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    acc_mlabes_accum = 0
    acc_node_valid_accum = 0
    acc_edge_valid_accum = 0
    acc_mlabes_valid_accum = 0
    acc_node_accum_org = 0

    acc_node_accum_indenti = 0
    
    
    
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    # for step, batch in enumerate(train_loader):    
        if org_dataloader_iterator is not None:
        # org_mask loss
            try:
                org_batch = next(org_dataloader_iterator)
            except StopIteration:
                org_dataloader_iterator = iter(data_loader_org)
                org_batch = next(org_dataloader_iterator)
            org_batch.to(device)
            node_rep_org = model_react.gnn(org_batch.x, org_batch.edge_index, org_batch.edge_attr)
            pred_node_org = model_react.org_linear_pred_atoms(node_rep_org[org_batch.masked_atom_indices])
            org_loss = criterion(pred_node_org.double(), org_batch.mask_node_label[:,0])

            acc_node_org = compute_accuracy(pred_node_org, org_batch.mask_node_label[:,0])
            acc_node_accum_org += acc_node_org        
        
        temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(batch, device)
        
        ## loss for nodes
        
        pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        loss_mlabel = criterion(pred_node.double(), tgt)
        
        acc_mlabes = compute_accuracy(pred_node, tgt)
        acc_mlabes_accum += acc_mlabes

        ## loss for indenti
        pred_node = model_react.linear_pred_atoms(indenti_node_rep[indenti_labels >= 0])
        pred_node = torch.nn.Sigmoid()(pred_node)
        tgt = indenti_labels[indenti_labels >= 0].reshape(pred_node.shape).to(device)
        
        # focal loss
        indenti_loss = ops.sigmoid_focal_loss(pred_node.double(), tgt, alpha=cfg.training_settings.focal_loss_alpha, gamma=cfg.training_settings.focal_loss_gamma).mean()
        
        # indenti_loss = criterion_indenti(pred_node.double(), tgt)

        acc_node_indenti = compute_accuracy_indenti(pred_node, tgt)
        acc_node_accum_indenti += acc_node_indenti


        optimizer.zero_grad()
        # optimizer_linear_pred_mlabes.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        if org_dataloader_iterator is not None:
            loss = loss_mlabel + org_loss + indenti_loss
        else:
            loss = loss_mlabel + indenti_loss
        
        loss.backward()

        if adv is not None:
            adv.backup_grad()
            for t in range(adv.K):
                adv.attack(is_first_attack=(t==0))
                if t != adv.K - 1:
                    # all zero grad
                    optimizer.zero_grad()
                else:
                    adv.restore_grad()

                # calculate loss
                if org_dataloader_iterator is not None:
                    try:
                        org_batch = next(org_dataloader_iterator)
                    except StopIteration:
                        org_dataloader_iterator = iter(data_loader_org)
                        org_batch = next(org_dataloader_iterator)
                    org_batch.to(device)
                    node_rep_org = model_react.gnn(org_batch.x, org_batch.edge_index, org_batch.edge_attr)
                    pred_node_org = model_react.org_linear_pred_atoms(node_rep_org[org_batch.masked_atom_indices])
                    org_loss = criterion(pred_node_org.double(), org_batch.mask_node_label[:,0])

                # temp_graph, node_rep, mlabes = model_react(batch, device)
                temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(batch, device)
                ## loss for nodes
                
                pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
                tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
                # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
                loss_mlabel = criterion(pred_node.double(), tgt)

                pred_node = model_react.linear_pred_atoms(indenti_node_rep[indenti_labels >= 0])
                pred_node = torch.nn.Sigmoid()(pred_node)
                tgt = indenti_labels[indenti_labels >= 0].reshape(pred_node.shape).to(device)
                indenti_loss = criterion_indenti(pred_node.double(), tgt)

                if org_dataloader_iterator is not None:
                    loss_adv = loss_mlabel + org_loss + indenti_loss
                else:
                    loss_adv = loss_mlabel + indenti_loss
                loss_adv.backward() # 
            adv.restore() # 
        
        optimizer.step()
        loss_accum += float(loss.cpu().item())
        
        if step % 100 == 1 and not cfg.training_settings.testing_stage:
            if org_dataloader_iterator is not None:
                wandb.log({"loss_accum": loss_accum/step,
                            "train_Mlabes_acc": acc_mlabes,
                            "train_Mlabes_acc_accum": acc_mlabes_accum/step,
                            "stage1_mask_attr": acc_node_accum_org/step,
                            "loss": loss,
                            "state1_mask_loss": org_loss,
                            "stage1_mask_acc": acc_node_org,
                            "loss_indenti": indenti_loss,
                            "train_indenti_acc": acc_node_indenti,
                            "train_indenti_acc_accum": acc_node_accum_indenti / step,
                        })
            else:
                wandb.log({"train_loss": loss,
                            "train_loss_accum": loss_accum/step,
                            "train_Mlabes_acc": acc_mlabes,
                            "train_Mlabes_acc_accum": acc_mlabes_accum/step,
                            "loss_mlabel": loss_mlabel,
                            "loss_indenti": indenti_loss,
                            "train_indenti_acc": acc_node_indenti,
                            "train_indenti_acc_accum": acc_node_accum_indenti / step,
                })        
        # if step > 300:
        #     break
    
    model_react.eval()
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        # temp_graph, node_rep, mlabes = model_react(valid_batch, device)
        temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(valid_batch, device)
        pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        tgt = torch.tensor([mlabe for i, mlabe in enumerate(valid_batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        valid_loss = criterion(pred_node.double(), tgt)
        acc_mlabes_valid = compute_accuracy(pred_node, tgt)
        acc_mlabes_valid_accum += acc_mlabes_valid
        valid_loss_accum += float(valid_loss.cpu().item())
    if not cfg.training_settings.testing_stage:
        wandb.log({
               "valid_loss": valid_loss,
               "valid_loss_accum": valid_loss_accum/valid_step,
               "valid_Mlabes_acc": acc_mlabes_valid,
               "valid_Mlabes_acc_accum": acc_mlabes_valid_accum/valid_step
                   })
    return loss_accum/step, acc_mlabes_accum/step, valid_loss_accum/valid_step, acc_mlabes_valid_accum/valid_step

@hydra.main(version_base=None, config_path="/conf", config_name="pretrain_mlabels_multi_task")
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

    if cfg.training_settings.stage1_on:
        # org mask dataset
        root_dataset = '/home/fengshikun/Backup/chem/dataset'
        dataset = MoleculeDataset("{}/".format(root_dataset) + cfg.training_settings.dataset, dataset=cfg.training_settings.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = cfg.training_settings.mask_rate, mask_edge=cfg.training_settings.mask_edge))
        data_loader_org = DataLoaderMasking(dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers)
        
        # org_linear_pred_atoms = torch.nn.Linear(cfg.model.emb_dim, 119).to(device)
        # org_linear_pred_atoms.load_state_dict(torch.load(cfg.training_settings.stage_one_head_model))
        # org_linear_pred_bonds = torch.nn.Linear(cfg.model.emb_dim, 4).to(device)

        # org_optimizer_linear_pred_atoms = optim.Adam(org_linear_pred_atoms.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
        # org_optimizer_linear_pred_bonds = optim.Adam(org_linear_pred_bonds.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    
    # use_frag: "brics", "brics_plus", "mlabel"
    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file, brics_file=USPTO_CONFIG.brics_file, brics_plus_file=USPTO_CONFIG.brics_plus_file, use_frag=cfg.training_settings.use_frag)
    
    val_size = cfg.training_settings.validation_size
    if cfg.training_settings.testing_stage:
        train_size = cfg.training_settings.validation_size
        test_size = len(reaction_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = torch.utils.data.random_split(reaction_dataset, [train_size, val_size, test_size])
    else:
        train_size = len(reaction_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(reaction_dataset, [train_size, val_size])
    
    # collator_rc = CustomCollator(mask_stage='reaction_centre')
    # collator_1hop = CustomCollator(mask_stage='one_hop')
    # collator_2hop = CustomCollator(mask_stage='two_hop')
    # collator_3hop = CustomCollator(mask_stage='three_hop')
    

    collator_rc = MaskIndentiCollator(mask_stage='reaction_centre')
    collator_1hop = MaskIndentiCollator(mask_stage='one_hop')
    collator_2hop = MaskIndentiCollator(mask_stage='two_hop')
    collator_3hop = MaskIndentiCollator(mask_stage='three_hop')
    



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
    

    if cfg.training_settings.gnn_decoder:
        gnn_dec = GNNDecoder(cfg.model.emb_dim, cfg.model.emb_dim, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, gnn_type = cfg.model.gnn_type)
    else:
        gnn_dec = None

    model_react = ReactantStage2(model, cfg.model.emb_dim, cfg.training_settings.stage1_on, cfg.training_settings.stage_one_head_model, gnn_decoder=gnn_dec).to(device)
    
    
    # model_list = [model_react, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes]

    # if cfg.training_settings.stage1_on:
    #     model_list.append(org_linear_pred_atoms)
    #     model_list.append(org_linear_pred_bonds)

    
    #set up optimizers
    optimizer_model = optim.Adam(model_react.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    # optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    # optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    # optimizer_linear_pred_mlabes = optim.Adam(linear_pred_mlabes.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)

    # optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes]

    # if cfg.training_settings.stage1_on:
    #     optimizer_list.append(org_optimizer_linear_pred_atoms)
    #     optimizer_list.append(org_optimizer_linear_pred_bonds)
    

    if cfg.training_settings.stage1_on:
        org_dataloader_iterator = iter(data_loader_org)
    else:
        org_dataloader_iterator = None
        data_loader_org = None

    # adv
    if cfg.training_settings.adv:
        adv = PGD(model_react, K=cfg.training_settings.K)
    else:
        adv = None

    
    if cfg.model.output_model_file:
        base_dir = os.path.dirname(cfg.model.output_model_file)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    print("=====Stage2: Phase 1 - original reaction centre masking")
    for epoch in range(1, cfg.training_settings.epochs_RC+1):
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_react, data_loader_rc, optimizer_model, device, org_dataloader_iterator, data_loader_org, adv)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)
        
        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_rc, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
        else:
            print("Wrong Target Setting")

        torch.save(model.state_dict(), cfg.model.output_model_file + "_epoch{}_{}_rc.pth".format(epoch, cfg.model.target)) 
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(model.state_dict(), cfg.model.output_model_file + "_{}_rc.pth".format(cfg.model.target))
       
    print("=====Stage2: Phase 2 - One Hop neighbours masking")
    for epoch in range(1, cfg.training_settings.epochs_1hop+1):
        
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_react, data_loader_1hop, optimizer_model, device, org_dataloader_iterator, data_loader_org, adv)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom, adv)

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
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_react, data_loader_2hop, optimizer_model, device, org_dataloader_iterator, data_loader_org, adv)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom, adv)

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
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_react, data_loader_3hop, optimizer_model, device, org_dataloader_iterator, data_loader_org, adv)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom, adv)
    
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