import argparse
import math
import os

import sys
sys.path.insert(0,'..')

from test.gcn_utils.datas import MoleculeDataset, DataLoaderMasking, MaskAtom


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN
from sklearn.metrics import roc_auc_score

import pandas as pd



from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss()

import timeit

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)


def train(args, model_list, loader, optimizer_list, device, tf_writer=None, base_step=0):
    model, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0

    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch.x: [num_node, 2] atom_type & chirality_tag;
        # edge_index: [2, num_edge]
        # edge_attr: [num_edge, 2]
        batch = batch.to(device)

        if args.adv:
            embed_init = model.x_embedding1(batch.x[:, 0]) + model.x_embedding2(batch.x[:, 1]) # [num_node, dim_embed]
            if args.norm_type == 'l2':
                delta = torch.zeros_like(embed_init).uniform_(-1, 1)
                delta = (delta * 0.1 / math.sqrt(embed_init.size(-1))).detach()
            elif args.norm_type == 'linf':
                delta = torch.zeros_like(embed_init).uniform_(-0.1, 0.1)
            else:
                assert False, 'Not supported'
            
            for _ in range(args.adv_step):
                delta.requires_grad_()
                x_embed = embed_init + delta
                node_rep = model(x_embed, batch.edge_index, batch.edge_attr, args.adv)
                ## loss for nodes
                pred_node = linear_pred_atoms(node_rep[batch.masked_atom_indices])
                loss = criterion(pred_node.double(), batch.mask_node_label[:,0])
                loss.backward()
                delta_grad = delta.grad.clone().detach()

                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad, dim=1).view(-1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:  # restrict norm
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                        + (1-exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:
                    print("Norm type {} not specified.".format(args.norm_type))
                
                embed_init = model.x_embedding1(batch.x[:, 0]) + model.x_embedding2(batch.x[:, 1])

            x_embed = embed_init + delta
            node_rep = model(x_embed, batch.edge_index, batch.edge_attr, args.adv)
        else:
            node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        ## loss for nodes
        pred_node = linear_pred_atoms(node_rep[batch.masked_atom_indices])
        # print(pred_node.shape, batch.mask_node_label[:,0].shape)
        loss = criterion(pred_node.double(), batch.mask_node_label[:,0])

        acc_node = compute_accuracy(pred_node, batch.mask_node_label[:,0])
        acc_node_accum += acc_node

        if tf_writer is not None:
            tf_writer.add_scalar('mask node loss', loss.item(), base_step + step)
            tf_writer.add_scalar('mask node acc', acc_node, base_step + step)



        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = linear_pred_bonds(edge_rep)
            mask_edge_loss = criterion(pred_edge.double(), batch.mask_edge_label[:,0])
            loss += mask_edge_loss

            acc_edge = compute_accuracy(pred_edge, batch.mask_edge_label[:,0])


            if tf_writer is not None:
                tf_writer.add_scalar('mask edge loss', mask_edge_loss.item(), base_step + step)
                tf_writer.add_scalar('mask edge acc', acc_edge, base_step + step)
            acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()

        loss_accum += float(loss.cpu().item())

    return loss_accum/step, acc_node_accum/step, acc_edge_accum/step

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
    parser.add_argument('--output_folder', type=str, default = 'pertrain_save', help='filefolder to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')

    # adv args
    parser.add_argument('--adv', action='store_true', default=False, help='adversarial training')
    parser.add_argument('--adv_lr', type=float, default=0.1, help='attack step size', choices=[0.1, 0.2, 0.03, 0.04, 0.05])
    parser.add_argument('--adv_step', type=int, default=3, help='attack iterations')
    parser.add_argument('--norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="0 is unlimited")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.mask_edge))


    #set up dataset and transform function.
    root_dataset = '/home/fengshikun/Backup/chem/dataset'
    dataset = MoleculeDataset("{}/".format(root_dataset) + args.dataset, dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up models, one for pre-training and one for context embeddings
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]



    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        pass

    tf_writer = SummaryWriter(f'{args.output_folder}/log')

    model_output_prefix = f'{args.output_folder}/{args.output_model_file}'

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    base_step = 0
    epoch_iters = len(dataset) // args.batch_size 

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        

        

        train_loss, train_acc_atom, train_acc_bond = train(args, model_list, loader, optimizer_list, device, tf_writer, base_step)
        base_step += epoch_iters
        print(train_loss, train_acc_atom, train_acc_bond)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_output_prefix + f"_{epoch}.pth")
            # save linear_pred_atoms and linear_pred_bonds
            torch.save(linear_pred_atoms.state_dict(), model_output_prefix + f"_linear_pred_atoms_{epoch}.pth")
            torch.save(linear_pred_bonds.state_dict(), model_output_prefix + f"_linear_pred_bonds_{epoch}.pth")

    if not args.output_model_file == "":
        torch.save(model.state_dict(), model_output_prefix + "_final.pth")
        # save linear_pred_atoms and linear_pred_bonds
        torch.save(linear_pred_atoms.state_dict(), model_output_prefix + "_linear_pred_atoms_final.pth")
        torch.save(linear_pred_bonds.state_dict(), model_output_prefix + "_linear_pred_bonds_final.pth")

if __name__ == "__main__":
    main()
