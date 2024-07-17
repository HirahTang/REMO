import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
import yaml
# sys.path.append(sys.path[0]+'/..')
import os


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
from datas.graphormer_data import GraphormerREACTDataset, BatchedDataDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


criterion = nn.CrossEntropyLoss()
criterion_indenti = nn.BCELoss()

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
# class ReactantStage2(torch.nn.Module):
#     """_summary_

#     Args:
#         torch (_type_): _description_
#     """
#     def __init__(self, gnn, emb_size, stage1_on=False, stage_one_head_model=None):
#         super(ReactantStage2, self).__init__()
#         self.gnn = gnn
#         self.emb_size = emb_size
#         # self.linear_pred_atoms = torch.nn.Linear(emb_size * 2, 119)
#         # self.linear_pred_bonds = torch.nn.Linear(emb_size * 2, 4)
#         self.linear_pred_mlabes = torch.nn.Linear(emb_size * 2, 2458) # todo, why one hop, two hop are still 2458

#         self.linear_pred_atoms = torch.nn.Linear(emb_size * 2, 1)

#         if stage1_on:
#             self.org_linear_pred_atoms = torch.nn.Linear(emb_size, 119)
#             self.org_linear_pred_atoms.load_state_dict(torch.load(stage_one_head_model))
#             # self.org_linear_pred_bonds = torch.nn.Linear(emb_size, 4)

#     def forward_single(self, batch_input, device):
#         batch_size = batch_input['graph'].idx.shape[0]
#         # print(batch_input['graph'].idx.get_device())
#         batch_graph = batch_input['graph'].to(device)
#         # print(batch_graph.x.get_device())
#         node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
#         # print(node_rep.get_device())
#         tensor_2_head = torch.zeros((node_rep.size(0), node_rep.size(1) * 2), dtype=node_rep.dtype, device=node_rep.device)
#         # batch_size = batch_input['graph'].idx.detach().cpu().shape[0]
#         for batch_num in range(batch_size):

#             node_rep_single = node_rep[batch_graph.batch.detach().cpu()==batch_num]

#             node_class_single = batch_input['primary_label'][batch_graph.batch.detach().cpu()==batch_num]

#             if node_class_single[-1] == -1: # The last atom is from condition reactants, means the reaction has condition reactants
#                 cond_rep = node_rep_single[node_class_single==-1]
#                 cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
#             else:
#                 cond_pool = torch.zeros((1, node_rep.size(1)), dtype=node_rep.dtype, device=node_rep.device)

#             tensor_2_head[batch_graph.batch.detach().cpu()==batch_num] = torch.cat([node_rep_single, torch.broadcast_to(cond_pool, node_rep_single.shape)], dim=1)

#         return batch_input['graph'], tensor_2_head
#         # batch_input['mlabes']

#     def forward(self, batch_input, device):
#         mask_input_graph, mask_head = self.forward_single(batch_input[0], device)
#         indenti_input_graph, indenti_head = self.forward_single(batch_input[1], device)
#         return mask_input_graph, mask_head, batch_input[0]['mlabes'], indenti_input_graph, indenti_head, batch_input[1]['primary_label']

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

def train_mlabes(cfg, model_react, loader, optimizer, org_dataloader_iterator=None, data_loader_org=None, adv=None, world_size=1, rank=0):
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
    acc_node_accum_indenti_valid = 0


    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
    # for step, batch in enumerate(train_loader):    
        if org_dataloader_iterator is not None:
        # org_mask loss
            try:
                org_batch = next(org_dataloader_iterator)
            except StopIteration:
                org_dataloader_iterator = iter(data_loader_org)
                org_batch = next(org_dataloader_iterator)
            org_batch.to(rank)
            node_rep_org = model_react.gnn(org_batch.x, org_batch.edge_index, org_batch.edge_attr)
            pred_node_org = model_react.org_linear_pred_atoms(node_rep_org[org_batch.masked_atom_indices])
            org_loss = criterion(pred_node_org.double(), org_batch.mask_node_label[:,0])

            acc_node_org = compute_accuracy(pred_node_org, org_batch.mask_node_label[:,0])
            acc_node_accum_org += acc_node_org        

        # temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(batch, device)
        for k in batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes', 'mask_node_label', 'mask_edge_label', 'edge_index', 'edge_attr', 'connected_edge_indices']:
                batch[k] = batch[k].to(rank)
        return_dict = model_react(batch)

        ## loss for nodes
        loss_mlabel = None
        indenti_loss = None
        if 'mask_loss' in return_dict:
            loss_mlabel = return_dict['mask_loss']
            # acc
            pred_node = return_dict['mask_logits']
            tgt = return_dict['mask_labels']
            acc_mlabes = compute_accuracy(pred_node, tgt)
            acc_mlabes_accum += acc_mlabes

        if 'identi_loss' in return_dict:
            indenti_loss = return_dict['identi_loss']
            # acc
            pred_node = return_dict['identi_logits']
            tgt = return_dict['identi_labels']
            acc_node_indenti = compute_accuracy_indenti(pred_node, tgt)
            acc_node_accum_indenti += acc_node_indenti




        # pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        # tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        # loss_mlabel = criterion(pred_node.double(), tgt)

        # acc_mlabes = compute_accuracy(pred_node, tgt)
        # acc_mlabes_accum += acc_mlabes

        # ## loss for indenti
        # pred_node = model_react.linear_pred_atoms(indenti_node_rep[indenti_labels >= 0])
        # pred_node = torch.nn.Sigmoid()(pred_node)
        # tgt = indenti_labels[indenti_labels >= 0].reshape(pred_node.shape).to(device)
        # indenti_loss = criterion_indenti(pred_node.double(), tgt)

        # acc_node_indenti = compute_accuracy_indenti(pred_node, tgt)
        # acc_node_accum_indenti += acc_node_indenti


        optimizer.zero_grad()
        # optimizer_linear_pred_mlabes.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        if org_dataloader_iterator is not None:
            loss = loss_mlabel + org_loss + indenti_loss
        else:
            loss = 0
            if loss_mlabel is not None:
                loss += loss_mlabel
            if indenti_loss is not None:
                loss += indenti_loss
            # loss = loss_mlabel + indenti_loss

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
                # temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(batch, device)
                ## loss for nodes
                return_dict = model_react(batch)

                if 'mask_loss' in return_dict:
                    loss_mlabel = return_dict['mask_loss']

                if 'identi_loss' in return_dict:
                    indenti_loss = return_dict['identi_loss']


                # pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
                # tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
                # # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
                # loss_mlabel = criterion(pred_node.double(), tgt)

                # pred_node = model_react.linear_pred_atoms(indenti_node_rep[indenti_labels >= 0])
                # pred_node = torch.nn.Sigmoid()(pred_node)
                # tgt = indenti_labels[indenti_labels >= 0].reshape(pred_node.shape).to(device)
                # indenti_loss = criterion_indenti(pred_node.double(), tgt)

                if org_dataloader_iterator is not None:
                    loss_adv = loss_mlabel + org_loss + indenti_loss
                else:
                    loss_adv = loss_mlabel + indenti_loss
                loss_adv.backward() # 
            adv.restore() # 

        optimizer.step()
        loss_accum += float(loss.cpu().item())
        
        

        if step % 100 == 1 and not cfg.training_settings.testing_stage and rank==0:
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
        if world_size > 1:
            dist.barrier()

    if rank==0:
        model_react.eval()
        for valid_step, valid_batch in enumerate(tqdm(val_loader, desc="Iteration")):
            # temp_graph, node_rep, mlabes = model_react(valid_batch, device)
            # temp_graph, node_rep, mlabes, indenti_graph, indenti_node_rep, indenti_labels = model_react(valid_batch, device)
            # pred_node = model_react.linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
            # tgt = torch.tensor([mlabe for i, mlabe in enumerate(valid_batch[0]["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
            # valid_loss = criterion(pred_node.double(), tgt)
            # acc_mlabes_valid = compute_accuracy(pred_node, tgt)
            # acc_mlabes_valid_accum += acc_mlabes_valid
            # valid_loss_accum += float(valid_loss.cpu().item())
            return_dict = model_react(batch)
            loss_mlabel = None
            indenti_loss = None
            if 'mask_loss' in return_dict:
                loss_mlabel = return_dict['mask_loss']
                # acc
                pred_node = return_dict['mask_logits']
                tgt = return_dict['mask_labels']
                acc_mlabes_valid = compute_accuracy(pred_node, tgt)
                acc_mlabes_valid_accum += acc_mlabes

            if 'identi_loss' in return_dict:
                indenti_loss = return_dict['identi_loss']
                # acc
                pred_node = return_dict['identi_logits']
                tgt = return_dict['identi_labels']
                acc_node_indenti_valid = compute_accuracy_indenti(pred_node, tgt)
                acc_node_accum_indenti_valid += acc_node_indenti_valid

    if not cfg.training_settings.testing_stage and rank==0:
        log_dict = {}
        if loss_mlabel != None:
            log_dict['loss_mlabel_valid'] = loss_mlabel
            log_dict["valid_Mlabes_acc"] = acc_mlabes_valid
            log_dict["valid_Mlabes_acc_accum"] = acc_mlabes_valid_accum/valid_step
        if 'indenti_loss' != None:
            log_dict['loss_indenti_valid'] = indenti_loss
            log_dict["valid_indenti_acc"] = acc_node_indenti_valid
            log_dict["valid_indenti_acc_accum"] = acc_node_accum_indenti_valid/valid_step


        wandb.log(log_dict)
    if world_size > 1:
        dist.barrier()
    if rank==0:
        return loss_accum/step, acc_mlabes_accum/step, valid_loss_accum/valid_step, acc_mlabes_valid_accum/valid_step
    else:
        return loss_accum/step, acc_mlabes_accum/step, 0, 0

# @hydra.main(version_base=None, config_path="../conf", config_name="pretrain_mlabels_multi_task")
def main(args): 

    with open(args.yaml_file) as f:
        # use safe_load instead load
        cfg = yaml.safe_load(f)
        cfg = Struct(**cfg)
        cfg.training_settings = Struct(**cfg.training_settings)
        cfg.model = Struct(**cfg.model)
        cfg.wandb = Struct(**cfg.wandb)
        
    
    # model = GNN(cfg.model.num_layer, cfg.model.emb_dim, JK = cfg.model.JK, drop_ratio = cfg.model.dropout_ratio, gnn_type = cfg.model.gnn_type)


    # graphormer model
    config_file = '../test/assets/graphormer_small.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
    model_config = Struct(**model_config)
    task_info = {'mask': True, 'indenti': True, 'NodeEdge': False, 'mlabel_task_vocab_size': 2458, 'finetune': False, 'task_num': 0}
    task_info = Struct(**task_info)
    
    world_size = torch.cuda.device_count()
    print('Start multi gpu training')
    
    
    rank = args.local_rank
    if world_size > 1:
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    

    torch.manual_seed(cfg.training_settings.seed)
    np.random.seed(cfg.training_settings.seed)


    if not cfg.training_settings.testing_stage and rank==0:
        wandb.login(key=cfg.wandb.login_key)
        wandb.init(project="Chemical-Reaction-Pretraining", name=cfg.wandb.run_name)
        
    
    model_react = GraphormerEncoder(model_config, task_info).to(rank)
    if world_size > 1: 
        model = DistributedDataParallel(model_react, device_ids=[rank], find_unused_parameters=False)
 
    
    if cfg.training_settings.stage2_on:
        model.load_state_dict(torch.load(cfg.training_settings.stage_one_model))
        print("Train from model {}".format(cfg.training_settings.stage_one_model))
    else:
        print("Train from scratch")

    if cfg.training_settings.stage1_on: # todo support the stage1 one in graphformer
        # org mask dataset
        root_dataset = '/home/fengshikun/Backup/chem/dataset'
        dataset = MoleculeDataset("{}/".format(root_dataset) + cfg.training_settings.dataset, dataset=cfg.training_settings.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = cfg.training_settings.mask_rate, mask_edge=cfg.training_settings.mask_edge))
        data_loader_org = DataLoaderMasking(dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers)

        # org_linear_pred_atoms = torch.nn.Linear(cfg.model.emb_dim, 119).to(device)
        # org_linear_pred_atoms.load_state_dict(torch.load(cfg.training_settings.stage_one_head_model))
        # org_linear_pred_bonds = torch.nn.Linear(cfg.model.emb_dim, 4).to(device)

        # org_optimizer_linear_pred_atoms = optim.Adam(org_linear_pred_atoms.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
        # org_optimizer_linear_pred_bonds = optim.Adam(org_linear_pred_bonds.parameters(), lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)

    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)

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
                    single_mask=True
                )
    batched_data = BatchedDataDataset(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            both_task=True, # indenti and mask, doubel tasks
        )

    
    # distribute dataset
    if world_size > 1:
        train_sampler = DistributedSampler(batched_data, num_replicas=world_size,
                                        rank=rank)
        data_loader_train_rc = torch.utils.data.DataLoader(batched_data, batch_size=cfg.training_settings.batch_size, num_workers = 4, collate_fn = batched_data.collater, sampler=train_sampler)
    else:    
        # init dataloader
        data_loader_train_rc = torch.utils.data.DataLoader(batched_data, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = 4, collate_fn = batched_data.collater)

    val_batched_data = BatchedDataDataset(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            both_task=True, # indenti and mask, doubel tasks
        )

    # init dataloader
    data_loader_val_rc = torch.utils.data.DataLoader(val_batched_data, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = 4, collate_fn = batched_data.collater)



    # collator_rc = MaskIndentiCollator(mask_stage='reaction_centre')
    # collator_1hop = MaskIndentiCollator(mask_stage='one_hop')
    # collator_2hop = MaskIndentiCollator(mask_stage='two_hop')
    # collator_3hop = MaskIndentiCollator(mask_stage='three_hop')




    # data_loader_train_rc = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_rc)
    # data_loader_val_rc = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_rc)
    data_loader_rc = [data_loader_train_rc, data_loader_val_rc]

    # data_loader_train_1hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_1hop)
    # data_loader_val_1hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_1hop)
    # data_loader_1hop = [data_loader_train_1hop, data_loader_val_1hop]

    # data_loader_train_2hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_2hop)
    # data_loader_val_2hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_2hop)
    # data_loader_2hop = [data_loader_train_2hop, data_loader_val_2hop]

    # data_loader_train_3hop = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training_settings.batch_size, shuffle=True, num_workers = cfg.training_settings.num_workers, collate_fn = collator_3hop)
    # data_loader_val_3hop = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = collator_3hop)
    # data_loader_3hop = [data_loader_train_3hop, data_loader_val_3hop]

    # model_react = ReactantStage2(model, cfg.model.emb_dim, cfg.training_settings.stage1_on, cfg.training_settings.stage_one_head_model).to(device)


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



    print("=====Stage2: Phase 1 - original reaction centre masking")
    for epoch in range(1, cfg.training_settings.epochs_RC+1):

        if cfg.model.target == "mlabes_identi":
            print("====epoch " + str(epoch) + " Train Loss | Train Accuracy | Validation Loss | Validation Accuracy")
            train_loss, train_acc_atom, val_loss, val_acc_atom = train_mlabes(cfg, model_react, data_loader_rc, optimizer_model, org_dataloader_iterator, data_loader_org, adv, world_size=world_size, rank=rank)
            if rank==0:
                print(train_loss, train_acc_atom, val_loss, val_acc_atom)

        elif cfg.model.target == "NodeEdge":
            print("====epoch " + str(epoch) + " Train Loss | Train Node Accuracy | Train Edge Accuracy | Validation Loss | Validation Node Accuracy | Validation Edge Accuracy")
            train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge = train_NodeEdge(cfg, model_list, data_loader_rc, optimizer_list, device)
            print(train_loss, train_acc_atom, train_acc_edge, val_loss, val_acc_atom, val_acc_edge)
        else:
            print("Wrong Target Setting")

        if not cfg.model.output_model_file_base == "" and not cfg.training_settings.testing_stage and rank==0:
            if not os.path.exists(cfg.model.output_model_file_base):
                os.makedirs(cfg.model.output_model_file_base)
            torch.save(model_react.state_dict(), cfg.model.output_model_file_base + "/{}_epoch_{}_rc.pth".format(cfg.model.target, epoch))

    return   

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
    parser = argparse.ArgumentParser(description='PyTorch multigpu grpahfromer training')
    parser.add_argument('--yaml_file', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    main(args)