import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
# sys.path.append(sys.path[0]+'/..')
sys.path.insert(0,'..')

import yaml
from data_analysis.USPTO_CONFIG import USPTO_CONFIG
from models.graphormer import GraphormerEncoder
from datas.graphormer_data import MyZINC, GraphormerPYGDataset, BatchedDataDataset
from utils.features import get_mask_atom_feature, get_bond_mask_feature
from tqdm import tqdm
from data_processing import ReactionDataset, CustomCollator
from utils.torchvocab import MolVocab
from dataclasses import dataclass, field
import numpy as np
from torch_geometric.data import Data, Batch

from models.gnn.models import GNN
from sklearn.metrics import roc_auc_score

import pandas as pd



from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
"""
@dataclass
class Collatorwithmasking:
    mlm: bool = True
    padding: Union[bool, str, PaddingStrategy] = True
    multi_hop_max_dist: int = 20
    spatial_pos_max: int = 20

    def __call__(self, items):
        graph_input = []
        batch = {}
"""

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def mol_mask(graph, mask_idx):
    mask_node_labels_list = []
    for atom_idx in mask_idx:
        # print(graph.x[atom_idx])
        # print(graph.x[atom_idx].view(1, -1))
        mask_node_labels_list.append(graph.x[atom_idx].view(1, -1))
    graph.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
    graph.masked_atom_indices = torch.tensor(mask_idx)
    for atom_idx in mask_idx:
        graph.x[atom_idx] = torch.tensor(get_mask_atom_feature(False))
        
    # mask edge
    connected_edge_indices = []
    for bond_idx, (u, v) in enumerate(graph.edge_index.cpu().numpy().T):
        for atom_idx in mask_idx:
            if atom_idx in set((u, v)) and \
                bond_idx not in connected_edge_indices:
                connected_edge_indices.append(bond_idx)
    # print(connected_edge_indices)
    if len(connected_edge_indices) > 0:
        mask_edge_labels_list = []
        for bond_idx in connected_edge_indices[::2]: 
            # because the
            # edge ordering is such that two directions of a single
            # edge occur in pairs, so to get the unique undirected
            # edge indices, we take every 2nd edge index from list
            mask_edge_labels_list.append(
                graph.edge_attr[bond_idx].view(1, -1))

        graph.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
        for bond_idx in connected_edge_indices:
            graph.edge_attr[bond_idx] = torch.tensor(get_bond_mask_feature(False))

            graph.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
    # print(graph)
    # print(graph.x)
    # print(graph.edge_index)
    # print(graph.edge_attr)
    return graph
def main():
    
     # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()
    
     # init model
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # config_file = 'assets/graphormer.yaml'
    # with open(config_file, 'r') as cr:
    #     model_config = yaml.safe_load(cr)
    # model_config = Struct(**model_config)
    # encoder = GraphormerEncoder(model_config).to(device)
    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    model.load_state_dict(torch.load("/sharefs/sharefs-hantang/chem/model_gin/masking.pth"))
    collator = CustomCollator(mask_stage='reaction_centre')
    data_loader = torch.utils.data.DataLoader(reaction_dataset, batch_size=4, shuffle=True, num_workers = 1, collate_fn = collator)
    for step, batch in enumerate(data_loader):
        # print(batch.keys())
        print(batch)
        
        if step > 2:
            break
    if False:
        condition_idx = [[i for i, x in enumerate(reaction_dataset[j]['molecule_idx']) if x != 0] for j in range(20, 25)]
        primary_idx = [[i for i, x in enumerate(reaction_dataset[j]['molecule_idx']) if x == 0] for j in range(20, 25)]
        print(condition_idx)
        print(primary_idx)
        test = [reaction_dataset[i]['graph'] for i in range(20, 25)]
        mask_idx = [reaction_dataset[i]['one_hop'] for i in range(20, 25)]
        mlabes = [reaction_dataset[i]['mlabes'] for i in range(20, 25)]
        mlabes = [i for sublist in mlabes for i in sublist]
        print(len(mlabes))
        for i in test:
            print(i)
        test_batch = Batch.from_data_list(test)
        mask_idx_batch = []
        for batch_idx, mask in enumerate(mask_idx):
            mask_idx_batch.extend([i+test_batch.ptr[batch_idx] for i in mask])
        mask_idx_batch = [int(t.item()) for t in mask_idx_batch]
        # print(mask_idx)
        # print(mask_idx_batch)
        test_batch_mask = mol_mask(test_batch, mask_idx_batch)
        print(test_batch)
        print(test_batch.batch, test_batch.ptr)
        print(test_batch_mask.mask_node_label)
        print(test_batch_mask.masked_atom_indices)
        # test = mol_mask(test, mask_idx)
        test_batch_mask = test_batch_mask.to(device)
        # print(test)
        # ool = nn.AdaptiveAvgPool1d(1).to(device)
        node_rep = model(test_batch_mask.x, test_batch_mask.edge_index, test_batch_mask.edge_attr)
        print(node_rep.shape)
        tensor_2_head = torch.tensor([]).to(device)
        for batch_num in range(5):
            node_rep_single = torch.index_select(node_rep, 0, torch.LongTensor([i for i, x in enumerate(test_batch_mask.batch) if x == batch_num]).to(device))
            cond_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(condition_idx[batch_num]).to(device))
            pri_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(primary_idx[batch_num]).to(device))
            
            cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            print(cond_rep.shape)
            print(cond_pool.shape)
            print(pri_rep.shape)
            new_rep = torch.cat([pri_rep, torch.broadcast_to(cond_pool, pri_rep.shape)], dim=1)
            # print(node_rep_single.shape)
            print(new_rep.shape)
            tensor_2_head = torch.cat([tensor_2_head, new_rep, torch.rand(cond_rep.shape[0], new_rep.shape[1]).to(device)], dim=0)
            
            
            print("\n")
        print(tensor_2_head.shape)
    # print(cond_rep.shape, pri_rep.shape)
    # print(reaction_dataset[20].keys())
    # print(reaction_dataset[20]['idx_list'])
    # print(reaction_dataset[20]['molecule_idx'])
    # print(len(reaction_dataset[20]['mlabes']))
    
    

    

    # print(reaction_dataset[0])
    # batched_data = BatchedDataDataset(
    #         reaction_dataset,
    #         max_node=model_config.max_nodes,
    #         multi_hop_max_dist=model_config.multi_hop_max_dist,
    #         spatial_pos_max=model_config.spatial_pos_max,
    #     )
    # print(batched_data[0])
    # data_loader = torch.utils.data.DataLoader(batched_data, batch_size=4, shuffle=True, num_workers = 1, collate_fn = batched_data.collater)
    # for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
    #     print(batch)
    #      for k in batch:
    #         batch[k] = batch[k]['graph']
        # batch = batch.to(device)
    #     x = batch
        # node_rep = encoder(batch)
    #     print('finish')

#     pass
if __name__ == '__main__':
    main()



