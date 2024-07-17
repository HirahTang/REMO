import wandb
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

from tqdm import tqdm
import numpy as np
import hydra
from models.gnn.models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error

from test.gcn_utils.splitters import scaffold_split
import os
from data_processing import FinetuneDDIDataset
import yaml

from models.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder, init_graphormer_params
from datas.graphormer_data import BatchedDataDataset_finetune, GraphormerPYGDataset
from models.graphormer.graphormer import RobertaHead

from data_analysis.USPTO_CONFIG import DDI_CONFIG
from glob import glob
from rdkit.Chem.PandasTools import LoadSDF

from sklearn.metrics import roc_auc_score,accuracy_score, precision_score, recall_score,precision_recall_curve, f1_score

def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error
    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)
    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds
    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    # print(cliff_test_idx)
    # print(y_test_pred.shape)
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def train(cfg, model, device, loader, optimizer):
    graphormer, linear_head = model
    graphormer.train()
    linear_head.train()

    if cfg.training_settings.l1_loss:
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            if k not in ['cliff']:
                batch[k] = batch[k].to(device)

        inner_state, graph_rep = graphormer(batch)
        pred = linear_head(graph_rep.view(graph_rep.size(0)//2, -1))
        even_indices = torch.arange(0, graph_rep.size(0)) % 2 == 0
        y = batch['y'][even_indices].view(pred.shape[0])

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred, y)

        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        if step % 100 == 1:
            wandb.log({"training loss": loss})
        loss.backward()
        # print(loss)
        optimizer.step()

def eval(cfg, model, device, loader, name="default"):
    graphormer, linear_head = model
    graphormer.eval()
    linear_head.eval()
    cliff_index = []
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for k in batch:
            if k not in ['cliff']:
                batch[k] = batch[k].to(device)

        with torch.no_grad():
            inner_state, graph_rep = graphormer(batch)
            pred = linear_head(graph_rep.view(graph_rep.size(0)//2, -1))
            even_indices = torch.arange(0, graph_rep.size(0)) % 2 == 0

        y_true.append(batch['y'][even_indices].view(pred.shape[0]))
        y_scores.append(pred)
        # cliff_index.extend(batch['cliff'])
    y_true = torch.cat(y_true, dim = 0)
    y_scores = torch.cat(y_scores, dim = 0)

    is_valid = y_true**2 > 0
    #Loss matrix
    if cfg.training_settings.l1_loss:
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    loss_mat = criterion(y_scores, y_true)

    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

    loss = torch.sum(loss_mat)/torch.sum(is_valid)

    y_true = y_true.cpu().numpy()
    y_scores = torch.argmax(y_scores, -1).cpu().numpy()
    # for i in range(y_true.shape[1]):
    #     #AUC is only defined when there is at least one positive data.
    #     mse_list
    #     if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
    #         is_valid = y_true[:,i]**2 > 0
    #         roc_list.append(mean_squared_error((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    f1_macro = f1_score(y_true,y_scores,average='macro') #macro-F1
    prec_macro = precision_score(y_true,y_scores,average='macro') #macro-Precision
    rec_macro = recall_score(y_true,y_scores,average='macro') #macro-Recall
    acc_score = accuracy_score(y_true, y_scores)
    wandb.log({f"{name}_f1_macro": f1_macro})
    wandb.log({f"{name}_prec_macro": prec_macro})
    wandb.log({f"{name}_rec_macro": rec_macro})
    wandb.log({f"{name}_acc_score": acc_score})

    return loss, (y_scores, y_true), cliff_index, [f1_macro, prec_macro, rec_macro, acc_score]  #y_true.shape[1]

def finetune(cfg, benchmark, device):

    #set up model
    # --------
    config_file = '/test/assets/graphormer_small.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_config = Struct(**model_config)

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
    # graphormer_model = GraphormerReact(model_config, graphormer_model)
    graphormer_model = graphormer_model.to(device)
    if cfg.model.input_model_file:
        graphormer_model.load_state_dict(torch.load(cfg.model.input_model_file, map_location=device))

    linear_head = RobertaHead(model_config.encoder_embed_dim * 2, 86, regression=False).to(device)

    # ===== Dataset Preparation
    name = benchmark.split('/')[-1].split('.')[0]
    print(f'The DDI Benchmark: {name}')
    datasets = get_ddi_datasets()
    train_dataset = datasets[0]
    val_dataset = datasets[1]
    val_dataset_unlabeled = datasets[2] 
    test_dataset = datasets[3]
    print(f"Check size: train {len(train_dataset)} {len(val_dataset)} {len(test_dataset)}")

    seed = 0

    data_set = GraphormerPYGDataset(
        None,
        seed,
        None,
        None,
        None,
        train_dataset,
        val_dataset,
        test_dataset
    )


    # ===== Model Loading & Finetuning

    batched_data_train = BatchedDataDataset_finetune(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False, ddi=True
        )

    batched_data_valid = BatchedDataDataset_finetune(
            data_set.valid_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False, ddi=True
        )

    batched_data_test = BatchedDataDataset_finetune(
            data_set.test_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
            cliff=False, ddi=True
        )

    train_loader = torch.utils.data.DataLoader(batched_data_train, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_train.collater)
    val_loader = torch.utils.data.DataLoader(batched_data_valid, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_valid.collater)
    test_loader = torch.utils.data.DataLoader(batched_data_test, batch_size=cfg.training_settings.batch_size, shuffle=False, num_workers = cfg.training_settings.num_workers, collate_fn = batched_data_test.collater)




     #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": graphormer_model.parameters()})
    model_param_group.append({"params": linear_head.parameters(), "lr":cfg.training_settings.lr*cfg.training_settings.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=cfg.training_settings.lr, weight_decay=cfg.training_settings.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    test_rmse_cliff_list = []
    test_output_list = []
    true_output_list = []
    print(cfg.model.filename)

    train_metric_list = []
    val_metric_list = []
    test_metric_list = []

    model = [graphormer_model, linear_head]
    for epoch in range(1, cfg.training_settings.epochs+1):

        print("====epoch " + str(epoch))

        train(cfg, model, device, train_loader, optimizer)

        print("====Evaluation")
        if cfg.training_settings.eval_train:
            train_acc, _, _, train_metric = eval(cfg, model, device, train_loader, "train")
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc, _, _, val_metric = eval(cfg, model, device, val_loader, "val")
        test_acc, (test_output, true_output), cliff_index, test_metric = eval(cfg, model, device, test_loader, "test")
        # cliff_rmse = calc_cliff_rmse(test_output, true_output, cliff_index)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        test_output_list.append(test_output)
        true_output_list.append(true_output)

        train_metric_list.append(train_metric)
        val_metric_list.append(val_metric)
        test_metric_list.append(test_metric)
        print("train: %f %s val: %f %s test: %f %s" %(train_acc, train_metric, val_acc, val_metric, test_acc, test_metric))

        wandb.log({f"sum_train_f1_macro": train_metric[0]})
        wandb.log({f"sum_train_prec_macro": train_metric[1]})
        wandb.log({f"sum_train_rec_macro": train_metric[2]})
        wandb.log({f"sum_train_acc_score": train_metric[3]})
        wandb.log({f"sum_val_f1_macro": val_metric[0]})
        wandb.log({f"sum_val_prec_macro": val_metric[1]})
        wandb.log({f"sum_val_rec_macro": val_metric[2]})
        wandb.log({f"sum_val_acc_score": val_metric[3]})
        wandb.log({f"sum_test_f1_macro": test_metric[0]})
        wandb.log({f"sum_test_prec_macro": test_metric[1]})
        wandb.log({f"sum_test_rec_macro": test_metric[2]})
        wandb.log({f"sum_test_acc_score": test_metric[3]})
        # test_rmse_cliff_list.append(cliff_rmse)

    max_ind = np.argmin(val_acc_list)
    print(f"The best epoch for {name}: {max_ind}\ntrain: {train_acc_list[max_ind], train_metric_list[max_ind]} val: {val_acc_list[max_ind], val_metric_list[max_ind]} test: {test_acc_list[max_ind], test_metric_list[max_ind]}")
    wandb.finish()

def get_ddi_datasets():
    label_file = DDI_CONFIG.label_file
    pairs = []
    labels = []
    with open(label_file, 'r') as lr:
        lr.readline()
        for line in lr:
            info_array = line.strip().split()
            pairs.append(info_array[:2])
            labels.append(int(info_array[2]))

    # split to train, val, test
    data_size = len(pairs)
    pairs = np.array(pairs)
    labels = np.array(labels)

    num_labels = np.max(labels) # 86 for deepddi
    labels = labels - 1
    train_size = int(data_size * DDI_CONFIG.train_ratio)
    val_size = int(data_size * (1 - DDI_CONFIG.train_ratio) * 0.5)
    perm = np.random.permutation(data_size)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    num_tasks = 1

    # load all the smiles:
    sdf_files = glob(DDI_CONFIG.data_dir + "/*.sdf")
    smiles_dict = {}
    for sdf_file in sdf_files:
        sdf_info = LoadSDF(sdf_file, smilesName='SMILES')
        smiles = sdf_info['SMILES'].item()
        file_name = os.path.basename(sdf_file)[:-4]
        smiles_dict[file_name] = smiles

    # train_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[train_idx], tokenizer[1], smiles_dict, labels[train_idx])

    train_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[train_idx], None, smiles_dict, labels[train_idx])
    val_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[val_idx], None, smiles_dict, labels[val_idx])

    # no label for the validation and test datasets
    # without labels
    val_dataset_unlabeled = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[val_idx], None, smiles_dict, labels[val_idx], get_labels=False)
    test_dataset = FinetuneDDIDataset(DDI_CONFIG.data_dir, pairs[test_idx], None, smiles_dict, labels[test_idx], get_labels=False)    

    return [train_dataset, val_dataset, val_dataset_unlabeled, test_dataset]


@hydra.main(version_base=None, config_path="/home/qcx679/hantang/Chemical_Reaction_Pretraining/conf", config_name="finetune")
def main(cfg):
    torch.manual_seed(cfg.training_settings.runseed)
    np.random.seed(cfg.training_settings.runseed)
    device = torch.device("cuda:" + str(cfg.training_settings.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training_settings.runseed)

    wandb.login(key=cfg.wandb.login_key)
    wandb.init(project="Chemical-Reaction-DDI-Finetune", name="wo_pretrain_graph_small")

    data_ls = []
    for file in os.listdir(cfg.ac_path):
        if file == 'DrugBank_known_ddi.txt':
            data_ls.append(file)

    data_ls = [os.path.join(cfg.ac_path, file) for file in data_ls]

    for benchmark in data_ls:
        finetune(cfg, benchmark, device)


if __name__ == "__main__":
    main()