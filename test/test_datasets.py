import torch
import argparse
import sys
# sys.path.append(sys.path[0]+'/..')
sys.path.insert(0,'..')

import yaml
from test.gcn_utils.datas import MoleculeDataset, DataLoaderMasking, MaskAtom
from models.graphormer import GraphormerEncoder
from datas.graphormer_data import MyZINC, GraphormerPYGDataset, BatchedDataDataset
from tqdm import tqdm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from tensorboardX import SummaryWriter
import numpy as np
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main1():
    # init model
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    config_file = 'assets/graphormer.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
    model_config = Struct(**model_config)

    root = '/sharefs/sharefs-hantang/ZINC/dataset'
    inner_dataset = MyZINC(root=root)
    train_set = MyZINC(root=root, split="train")
    valid_set = MyZINC(root=root, split="val")
    test_set = MyZINC(root=root, split="test")
    seed = 0

    print("Exact dataset:", train_set[0])

    data_set = GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )

    batched_data = BatchedDataDataset(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
        )

    
    
    # for index, item in enumerate(data_set):
    num_instances = len(data_set.train_data)
    print("num_insatnecs:", len(data_set))
    for i in range(num_instances):
        instance = data_set.train_data[i]
        print(instance)
        if i > 1:
            break
    # init dataloader
    data_loader = torch.utils.data.DataLoader(batched_data, batch_size=4, shuffle=True, num_workers = 1, collate_fn = batched_data.collater)

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        #for k in batch:
        #    batch[k] = batch[k].to(device)
        # print("Batches:", batch)
        if step > 1:
            break

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

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #set up dataset and transform function.
    root_dataset = 'chem/dataset'
    dataset = MoleculeDataset("{}/".format(root_dataset) + args.dataset, dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    print(dataset[0].x)
    print(dataset[0].edge_index)
    print(dataset[0].edge_attr)
if __name__ == "__main__":
    main()