import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
import yaml
sys.path.append(sys.path[0]+'/..')
import os
os.chdir("/pretrain/")

sys.path.insert(0,'..')
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from test.gcn_utils.datas import MoleculeDataset, DataLoaderMasking, MaskAtom
from test.gcn_utils.adv import PGD

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN

from models.graphormer import GraphormerEncoder

from sklearn.metrics import roc_auc_score
import hydra
import pandas as pd
from data_processing import ReactionDataset, CustomCollator, MaskIndentiCollator
from torch.utils.data import Dataset, ConcatDataset
from datas.graphormer_data import GraphormerREACTDataset, BatchedDataDataset
from models.graphormer.graphormer import RobertaHead
from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params

criterion = nn.CrossEntropyLoss()
criterion_indenti = nn.BCELoss()

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class GraphormerReact(nn.Module):
    def __init__(self, args, graphormer):
        super().__init__()
        self.max_nodes = args.max_nodes
        self.graph_encoder = graphormer

        self.share_input_output_embed = args.share_encoder_input_output_embed
        
    def forward(self, batched_data, perturb=None):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )
        
        x = inner_states[-1].transpose(0, 1)
        
        return self.get_reaction_node_embedding(x)       
        
    def get_reaction_node_embedding(self, node_embedding):
        batch_size = node_embedding.size(0)
        # batch_size x max_nodes_num x embeddings_size
        tensor_2_head = torch.zeros((node_embedding.size(0), node_embedding.size(1), node_embedding.size(2) * 2), dtype=node_embedding.dtype, device=node_embedding.device)
        # use the cls node as conditional embedding
        cls_emb = node_embedding[:, :1, :]
        tensor_2_head = torch.cat([node_embedding, torch.broadcast_to(cls_emb, node_embedding.shape)], dim=2)
        return tensor_2_head

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes
        
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def compute_accuracy_indenti(pred, target):
    # print(pred, target)
    return float(torch.sum((pred.detach() > 0.5) == target).cpu().item()) / len(pred)


def train_mlabes(cfg, model_list, loader, optimizer_list, device, epoch):
    train_loader, val_loader = loader
    model, linear_pred_atoms,linear_pred_bonds, linear_pred_mlabes = model_list
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
        for k in batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes', 'mask_node_label', 'mask_edge_label', 'edge_index', 'edge_attr', 'connected_edge_indices']:
                batch[k] = batch[k].to(device)

        node_rep = model(batch)
        mlabel_node_logits = linear_pred_mlabes(node_rep)
        mlabel_node_logits = mlabel_node_logits[:, 1:, :]
        logits_array = []
        mlabel_array = []
        for i in range(node_rep.size(0)):
            mask_idx = batch['mask_atom_indices'][i]
            node_logits = mlabel_node_logits[i][mask_idx]
            logits_array.append(node_logits)
            mlabel = torch.tensor(batch['mlabes'][i])
            node_mlabel = mlabel[mask_idx]
            mlabel_array.append(node_mlabel)
            
        all_logits = torch.cat(logits_array, dim=0)
        all_labels = torch.cat(mlabel_array).to(device)
        mlabel_loss = criterion(all_logits, all_labels)
        
        acc_mlabes = compute_accuracy(all_logits, all_labels)
        acc_mlabes_accum += acc_mlabes
        
        optimizer_model.zero_grad()
        optimizer_linear_pred_mlabes.zero_grad()
        
        mlabel_loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_mlabes.step()
        loss_accum += float(mlabel_loss.cpu().item())
        if not cfg.training_settings.testing_stage:
            if (step + 1) % cfg.training_settings.log_train_freq == 0:
                wandb.log({
                    "epoch": epoch,
                    "traing_loss_accum": loss_accum / cfg.training_settings.log_train_freq,
                    "train_loss": mlabel_loss,
                    "train_accuracy_accum": acc_mlabes_accum / cfg.training_settings.log_train_freq,
                    
                })
                loss_accum = 0
                acc_mlabes_accum = 0
    model.eval()
    linear_pred_atoms.eval()
    linear_pred_bonds.eval()
    linear_pred_mlabes.eval()
    
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        for k in valid_batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes', 'mask_node_label', 'mask_edge_label', 'edge_index', 'edge_attr', 'connected_edge_indices']:
                valid_batch[k] = valid_batch[k].to(device)
                
        node_rep = model(valid_batch)
        mlabel_node_logits = linear_pred_mlabes(node_rep)
        mlabel_node_logits = mlabel_node_logits[:, 1:, :]
        logits_array = []
        mlabel_array = []
    
        for i in range(node_rep.size(0)):
            mask_idx = valid_batch['mask_atom_indices'][i]
            node_logits = mlabel_node_logits[i][mask_idx]
            logits_array.append(node_logits)
            mlabel = torch.tensor(valid_batch['mlabes'][i])
            node_mlabel = mlabel[mask_idx]
            mlabel_array.append(node_mlabel)
            
        all_logits = torch.cat(logits_array, dim=0)
        all_labels = torch.cat(mlabel_array).to(device)
        mlabel_loss_valid = criterion(all_logits, all_labels)
        
        acc_mlabes_valid = compute_accuracy(all_logits, all_labels)
        acc_mlabes_valid_accum += acc_mlabes_valid
        
        valid_loss_accum += float(mlabel_loss_valid.cpu().item())
        
            
    if not cfg.training_settings.testing_stage:
        wandb.log({
            "epoch": epoch,
            "valid_loss_accum": valid_loss_accum / (valid_step+1),
            "valid_Mlabes_acc_accum": acc_mlabes_valid_accum / (valid_step+1),
            
        })
                
#    if not cfg.training_settings.testing_stage:
#        wandb.log({"train_loss": mlabel_loss,
#                            "train_loss_accum": loss_accum/step,
#                            "train_Mlabes_acc": acc_mlabes,
#                            "train_Mlabes_acc_accum": acc_mlabes_accum/step,
#                    "valid_loss": mlabel_loss_valid,
#                    "valid_loss_accum": valid_loss_accum/valid_step,
#                    "valid_Mlabes_acc": acc_mlabes_valid,
#                    "valid_Mlabes_acc_accum": acc_mlabes_valid_accum/valid_step
#                })
    return loss_accum/(step+1), acc_mlabes_accum/(step+1), valid_loss_accum/(valid_step+1), acc_mlabes_valid_accum/(valid_step+1) 
        
def train_NodeEdge(cfg, model_list, loader, optimizer_list, device, epoch):
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
        for k in batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes', 'mask_node_label', 'mask_edge_label', 'edge_index', 'edge_attr', 'connected_edge_indices']:
                batch[k] = batch[k].to(device)
                
        node_rep = model(batch) 
        node_rep = node_rep[:, 1:, :]
        atom_rep_array = []           
        atom_label_array = []
        edge_rep_array = []
        edge_label_array = []
        
        for i in range(node_rep.shape[0]):
            # Atom level rep construction
            mask_idx = batch['mask_atom_indices'][i]
            node_rep_mask = node_rep[i][mask_idx]
            atom_rep_array.append(node_rep_mask)
            # Atom level label construction
            atom_label_array.append(batch['mask_node_label'][i][:, 0])
            
            # Edge level rep construction
            mask_edge_index = batch['edge_index'][i][:, batch['connected_edge_indices'][i]]
            edge_rep = node_rep[i][mask_edge_index[0]] + node_rep[i][mask_edge_index[1]]
            edge_rep_array.append(edge_rep)
            # Edge level label construction
            edge_label_array.append(batch['mask_edge_label'][i][:, 0])
        
        atom_rep_agg = torch.cat(atom_rep_array, dim=0)
        atom_label_array = torch.cat(atom_label_array).to(device)
        
        edge_rep_agg = torch.cat(edge_rep_array, dim=0)
        edge_label_array = torch.cat(edge_label_array).to(device)
        
        atom_logits = linear_pred_atoms(atom_rep_agg)
        edge_logits = linear_pred_bonds(edge_rep_agg)
        
        atom_loss = criterion(atom_logits, atom_label_array)
        acc_atom = compute_accuracy(atom_logits, atom_label_array)
        
        edge_loss = criterion(edge_logits, edge_label_array)
        acc_edge = compute_accuracy(edge_logits, edge_label_array)
        
        acc_node_accum += acc_atom
        acc_edge_accum += acc_edge
        
        loss = atom_loss + edge_loss
        
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()
        
        loss_accum += float(loss.cpu().item())
        # concatenate the tensor
        # prediction with Roberta Heads
        # Calculate the loss, backprop
        if not cfg.training_settings.testing_stage:
            if (step + 1) % cfg.training_settings.log_train_freq == 0:
                wandb.log({
                    "epoch": epoch,
                    "traing_loss_accum": loss_accum / cfg.training_settings.log_train_freq,
                    "train_loss": loss,
                    "train_edge_acc_accum": acc_edge_accum / cfg.training_settings.log_train_freq,
                    "train_node_acc_accum": acc_node_accum / cfg.training_settings.log_train_freq,
                    
                })
                loss_accum = 0
                acc_node_accum = 0
                acc_edge_accum = 0
    model.eval()
    linear_pred_atoms.eval()
    linear_pred_bonds.eval()
    linear_pred_mlabes.eval()
    
    for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
        for k in valid_batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes', 'mask_node_label', 'mask_edge_label', 'edge_index', 'edge_attr', 'connected_edge_indices']:
                valid_batch[k] = valid_batch[k].to(device)
                
        node_rep = model(valid_batch) 
        node_rep = node_rep[:, 1:, :]
        atom_rep_array = []           
        atom_label_array = []
        edge_rep_array = []
        edge_label_array = []
        
        for i in range(node_rep.shape[0]):
            # Atom level rep construction
            mask_idx = valid_batch['mask_atom_indices'][i]
            node_rep_mask = node_rep[i][mask_idx]
            atom_rep_array.append(node_rep_mask)
            # Atom level label construction
            atom_label_array.append(valid_batch['mask_node_label'][i][:, 0])
            
            # Edge level rep construction
            mask_edge_index = valid_batch['edge_index'][i][:, valid_batch['connected_edge_indices'][i]]
            edge_rep = node_rep[i][mask_edge_index[0]] + node_rep[i][mask_edge_index[1]]
            edge_rep_array.append(edge_rep)
            # Edge level label construction
            edge_label_array.append(valid_batch['mask_edge_label'][i][:, 0])
        
        atom_rep_agg = torch.cat(atom_rep_array, dim=0)
        atom_label_array = torch.cat(atom_label_array).to(device)
        
        edge_rep_agg = torch.cat(edge_rep_array, dim=0)
        edge_label_array = torch.cat(edge_label_array).to(device)
        
        atom_logits = linear_pred_atoms(atom_rep_agg)
        edge_logits = linear_pred_bonds(edge_rep_agg)
        
        atom_loss = criterion(atom_logits, atom_label_array)
        acc_atom_valid = compute_accuracy(atom_logits, atom_label_array)
        
        edge_loss = criterion(edge_logits, edge_label_array)
        acc_edge_valid = compute_accuracy(edge_logits, edge_label_array)
        
        acc_node_valid_accum += acc_atom_valid
        acc_edge_valid_accum += acc_edge_valid
        
        valid_loss = atom_loss + edge_loss
        valid_loss_accum += float(valid_loss.cpu().item())
    
    if not cfg.training_settings.testing_stage:
        wandb.log({
            "epoch": epoch,
            "valid_loss_accum": valid_loss_accum / (valid_step+1),
            "valid_node_acc_accum": acc_node_valid_accum / (valid_step+1),
            "valid_edge_acc_accum": acc_edge_valid_accum / (valid_step+1), 
            
        })
        
        
#    if not cfg.training_settings.testing_stage:
#        wandb.log({
#                "train_loss": loss,
#                "train_loss_accum": loss_accum/step,
#                "train_node_acc": acc_atom,
#                "train_node_acc_accum": acc_node_accum/step,
#                "train_edge_acc": acc_edge,
#                "train_edge_acc_accum": acc_edge_accum/step,
#                "valid_loss": valid_loss,
#                "valid_loss_accum": valid_loss_accum/valid_step,
#                "valid_node_acc": acc_atom_valid,
#                "valid_edge_acc": acc_edge_valid,
#                "valid_node_acc_accum": acc_node_valid_accum/valid_step,
#                "valid_edge_acc_accum": acc_edge_valid_accum/valid_step
#            })
        
    return loss_accum/(step+1), acc_node_accum/(step+1), acc_edge_accum/(step+1), valid_loss_accum/(valid_step+1), acc_node_valid_accum/(valid_step+1), acc_edge_valid_accum/(valid_step+1)
    # The evaluation part 
    # Sum up the wandb log
    # Demo testing

    
    
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
        
    # ---
    # Define the models here
    config_file = '/test/assets/graphormer_small.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
    model_config = Struct(**model_config)
    
    graphormer_encoder = GraphormerGraphEncoder(
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
    
    # ---
    # ---
    # graphormer to device
    
    graphormer = GraphormerReact(model_config, graphormer_encoder).to(device)
    
    # ---
    if cfg.training_settings.stage2_on:
        graphormer.load_state_dict(torch.load(cfg.training_settings.stage_one_model, map_location=device))
        print("Train from model {}".format(cfg.training_settings.stage_one_model))
    else:
        print("Train from scratch")

    if cfg.training_settings.reaction_dataset == "480K":
        train_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_train, atom_vocab=USPTO_CONFIG.atom_vocab_file)
        train_dataset2 = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_valid, atom_vocab=USPTO_CONFIG.atom_vocab_file)
        val_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset_480k_test, atom_vocab=USPTO_CONFIG.atom_vocab_file)
        train_dataset = ConcatDataset([train_dataset, train_dataset2])
    
    elif cfg.training_settings.reaction_dataset == "full":
        reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
        val_size = cfg.training_settings.validation_size
        train_size = len(reaction_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(reaction_dataset, [train_size, val_size])
    
    else:
        raise ValueError("Reaction Dataset shall from {'full', '480K'}")
            
    if cfg.training_settings.testing_stage:
        train_size = cfg.training_settings.validation_size
        val_size = cfg.training_settings.validation_size
        test_size = len(train_dataset) - train_size - val_size
        train_dataset, val_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size, test_size])
        
    # ---
    # Define the dataset class for graphomers here
    
    seed = 0
    
    data_set = GraphormerREACTDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_dataset,
                    val_dataset,
                    # test_set=val_dataset,
                )
    
    batched_data = BatchedDataDataset(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
        )
    
    val_batched_data = BatchedDataDataset(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
        )
    
    # ---
    
    # ---
    # Define the novel dataloader
    
    data_loader_train = torch.utils.data.DataLoader(batched_data, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data.collater)
    data_loader_val = torch.utils.data.DataLoader(val_batched_data, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data.collater)
    data_loader_rc = [data_loader_train, data_loader_val]
    # ---
    
    
    
    
    pred_atoms_head = RobertaHead(model_config.encoder_embed_dim * 2, 119).to(device)
    pred_bonds_head = RobertaHead(model_config.encoder_embed_dim * 2, 4).to(device)
    pred_mlabes_head = RobertaHead(model_config.encoder_embed_dim * 2, 2458).to(device)
    
    model_list = [graphormer, pred_atoms_head, pred_bonds_head, pred_mlabes_head]
    
    #set up optimizers
    optimizer_model = optim.Adam(graphormer.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_atoms = optim.Adam(pred_atoms_head.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_bonds = optim.Adam(pred_bonds_head.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    optimizer_linear_pred_mlabes = optim.Adam(pred_mlabes_head.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes]
    
    print("=====Stage2: Phase 1 - original reaction centre masking")
    for epoch in range(1, cfg.training_settings.epochs_RC+1):
        
        if cfg.model.target == "mlabes":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_list, data_loader_rc, optimizer_list, device, epoch)
            print(train_loss, train_acc_atom, val_loss, val_acc_atom)
        
        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_rc, optimizer_list, device, epoch)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
        else:
            print("Wrong Target Setting")
    
    if not cfg.model.output_model_file == "" and not cfg.training_settings.testing_stage:
       torch.save(graphormer_encoder.state_dict(), cfg.model.output_model_file + "graphormer_masking_{}_rc.pth".format(cfg.model.target))
       
if __name__ == "__main__":
    main()

    