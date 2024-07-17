from dataclasses import replace, dataclass, field
from operator import add
import os
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import random
import math
# !pip install datasets
# from datasets import load_dataset
from torch_geometric.data import Data, Batch
from collections import Counter
import pandas as pd
from tqdm import tqdm
# from rdkit import Chem
# from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch.utils.data import Dataset
from data_analysis.USPTO_CONFIG import USPTO_CONFIG
import pickle
from utils.features import atom_to_feature_vector, bond_to_feature_vector, get_mask_atom_feature, get_bond_mask_feature
from datas.graphormer_data import preprocess_item
from utils.torchvocab import MolVocab
from rdkit import RDLogger
from test.gcn_utils.datas import MoleculeDataset, mol_to_graph_data_obj_simple

from datas.mat_data import mat_handle_mol, mol_collate_func

RDLogger.DisableLog('rdApp.*')
def dataset_forming():

     # Adjust the reactant_dict to appropriate for dataset
    
    mol_list = []
    condition_list = []
    reaction_centre_list = []
    one_hop_list = []
    two_hop_list = []
    three_hop_list = []

    with open(USPTO_CONFIG.reaction_centre, 'rb') as handle:
        reactant_dict = pickle.load(handle)

    # reactant_dict = list(reactant_dict.items())[:n]

    for reaction, content in tqdm(reactant_dict.items()):
        for mol, mol_content in content.items():
            reactants = reaction.split(">>")[0]
            condition_group = reactants.split('.')
            condition_group = [i for i in condition_group if i != mol]
            
            mol_list.append(mol)
            condition_list.append('.'.join(condition_group))
            reaction_centre_list.append('.'.join([str(i) for i in mol_content['reaction_centre']]))
            one_hop_list.append('.'.join([str(i) for i in mol_content['one_hop']]))
            two_hop_list.append('.'.join([str(i) for i in mol_content['two_hop']]))
            three_hop_list.append('.'.join([str(i) for i in mol_content['three_hop']]))

    data = {"primary_molecule": mol_list,
            "condition_molecule": condition_list,
            "reaction_centre": reaction_centre_list,
            "one_hop": one_hop_list,
            "two_hop": two_hop_list,
            "three_hop": three_hop_list}
    
    reaction_df = pd.DataFrame(data=data)

    reaction_df.to_csv(USPTO_CONFIG.dataset, index=False)


BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']
def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()

    """
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval(f"bond.Get{bond_feature}")()
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'


def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output

def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output


def conver_mlabels(smiles_lst, atom_vocab):
    mol_lst = []
    mlabes = []
    for mol_idx, smiles in enumerate(smiles_lst):
        mol = Chem.MolFromSmiles(smiles)
        mol_lst.append(mol)
        for atom in mol.GetAtoms():
            mlabes.append(atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index))
    return mol_lst, mlabes

def smiles2graph(smiles_lst, atom_vocab=None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mlabes = []
    atom_features_list = []
    #if primary:    
    atom_idx_list = []
    edges_list = []
    edge_features_list = []
    mol_idx_list = []
    start_idx = 0 
    for mol_idx, smiles in enumerate(smiles_lst):
        mol = Chem.MolFromSmiles(smiles)

        # atoms

        for atom in mol.GetAtoms():
            mol_idx_list.append(mol_idx)
            # if mol_idx == 0:
            atom_idx_list.append(atom.GetIdx() + start_idx)
            atom_features_list.append(atom_to_feature_vector(atom, False))
            if atom_vocab is not None:
                mlabes.append(atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index))
            
        
    # 

    # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(mol.GetBonds()) > 0:  # mol has bonds
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + start_idx
                j = bond.GetEndAtomIdx() + start_idx

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # print(edges_list)
            
            # print(edge_index)
            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            

        else:  # mol has no bonds
            pass
            

        start_idx += mol.GetNumAtoms()
    # graph = dict()
    # graph['edge_index'] = edge_index
    # graph['edge_attr'] = edge_attr
    # graph['node_attr'] = x
    # graph['num_nodes'] = len(x)
    # print(edges_list)
    if len(edges_list):
        edge_index = np.array(edges_list, dtype=np.int64)
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((0, 2), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
    x = np.array(atom_features_list, dtype=np.int64)
    graph = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index.T), edge_attr=torch.tensor(edge_attr))

    return graph, mlabes, np.array(atom_idx_list, dtype=np.int64), np.array(mol_idx_list, dtype=np.int64)

class ActivityCliffDataset(Dataset):
    
    """
    Dataset to load the MoleculeACE Benchmark for evaluation
    """
    
    def __init__(self, dataset):
        super().__init__()
        # self.atom_vocab = MolVocab.load_vocab(atom_vocab)
        # self.atom_vocab_size = len(self.atom_vocab)
        # print("Atom Vocab Size:", self.atom_vocab_size)
        self.data = dataset
        self.length = len(self.data)
        print("Number of lines: " + str(self.length))
        self.smiles_lst = self.data['smiles'].to_list()
        self.y = self.data['y'].to_list()
        self.mol_list = [AllChem.MolFromSmiles(s) for s in self.smiles_lst]
        self.data_list = [mol_to_graph_data_obj_simple(mol) for mol in self.mol_list]
        self.cliff = self.data['cliff_mol'].to_list()
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        y = self.y[idx]
        data.id = torch.tensor([idx])
        data.y = torch.tensor([y])
        data.cliff = self.cliff[idx]
        return data
        
        
        

class FinetuneDDIDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, pair_ids, tokenizer, smiles_dict, labels, data_suffix=".sdf", get_labels=True):
        # collect smiles
        self.smiles_lst = []
        self.pair_labels = []
        self.data_lst = []
        for sdf_ids, label in zip(pair_ids, labels):
            smiles_pair = []
            for sdf_id in sdf_ids:
                s = smiles_dict[sdf_id]
                smiles_pair.append(s)
                mol = AllChem.MolFromSmiles(s)
                self.data_lst.append(mol_to_graph_data_obj_simple(mol))
                # smiles_pair.append(smiles)
            self.smiles_lst.append(smiles_pair)  
            self.pair_labels.append(label)      
            self.pair_labels.append(label)      

        # self.pair_labels = None
        # self.labels = None
        # if labels is not None:

        self.get_labels = get_labels

        assert len(self.pair_labels) == 2*len(self.smiles_lst)
        self.labels = np.array(self.pair_labels)
        
    def __len__(self):
        return len(self.smiles_lst)*2


    def __getitem__(self, idx):      
        data = self.data_lst[idx]
        y = self.labels[idx]
        data.idx = torch.tensor([idx])
        data.y = torch.tensor([y])

        return data


class ReactionDataset(Dataset):

    """
    Dataset to load reaction molecules, with reaction centre masking or reaction centre labels
    """

    def __init__(self, data_path, atom_vocab=None, brics_file=None, brics_plus_file=None, use_frag=None, single_molecule=False):
        super().__init__()
        # self.atom_vocab = atom_vocab
        # self.mask_stage = mask_stage
        self.atom_vocab = MolVocab.load_vocab(atom_vocab)
        self.atom_vocab_size = len(self.atom_vocab)
        print("Atom Vocab Size:", self.atom_vocab_size)
        def load_dataset(data_path):
            data = pd.read_csv(data_path)
            valid_idx = (data['reaction_centre'].isna()==False)
            return data[valid_idx], valid_idx

        self.data, valid_idx = load_dataset(data_path)
        print("Loaded Dataset")
        self.use_frag = use_frag
        self.single_molecule = single_molecule
        if use_frag is not None: # and ('brics' in use_frag):
            if use_frag == 'brics':
                self.brics_labels = np.load(brics_file, allow_pickle=True)[valid_idx]
            else:
                self.brics_labels = np.load(brics_plus_file, allow_pickle=True)[valid_idx]


            



        self.length = len(self.data)
        print("Number of lines: " + str(self.length))

    def __len__(self):
        return self.length    


    def __getitem__(self, idx):
        
        if self.single_molecule:
            data_row = self.data.iloc[idx]
            primary_smi = [data_row['primary_molecule']]
            
            reaction_centre = [int(i) for i in data_row['reaction_centre'].split('.')]
            one_hop = [int(i) for i in data_row['one_hop'].split('.')]
            two_hop = [int(i) for i in data_row['two_hop'].split('.')]
            three_hop = [int(i) for i in data_row['three_hop'].split('.')]
            
            graph, mlabes, idx_list, molecule_idx = smiles2graph(primary_smi, atom_vocab=self.atom_vocab)
            graph.idx = idx
            output = {}
            output['graph'] = graph
            output['idx_list'] = idx_list
            output['molecule_idx'] = molecule_idx
        # output['condition_graph'] = condition_graph
            output['reaction_centre'] = reaction_centre
            output['one_hop'] = one_hop
            output['two_hop'] = two_hop
            output['three_hop'] = three_hop
            
            
            output['mlabes'] = mlabes
            return output
            
        
        data_row = self.data.iloc[idx]
        primary_smi = [data_row['primary_molecule']]
        #print(data_row['condition_molecule'])
        
        try:
            non_condition = math.isnan(data_row['condition_molecule'])
            condition_smi = []
        except:
            condition_smi = data_row['condition_molecule'].split('.')
        
        reaction_centre = [int(i) for i in data_row['reaction_centre'].split('.')]
        one_hop = [int(i) for i in data_row['one_hop'].split('.')]
        two_hop = [int(i) for i in data_row['two_hop'].split('.')]
        three_hop = [int(i) for i in data_row['three_hop'].split('.')]


        if self.use_frag:
            frag_label = np.array(self.brics_labels[idx])
        
        
        molecule_list = primary_smi + condition_smi
        # print(molecule_list)
        graph, mlabes, idx_list, molecule_idx = smiles2graph(molecule_list, atom_vocab=self.atom_vocab)
        # print(graph.x)
        graph.idx = idx
        # graph = preprocess_item(graph)
        # print(graph.x)
        # condition_graph = []
        # for c_smi in condition_smi:
        #     c_graph, _, _ = smiles2graph(c_smi, primary=False)
        #     condition_graph.append(c_graph)

        output = {}
        output['graph'] = graph
        output['idx_list'] = idx_list
        output['molecule_idx'] = molecule_idx
        # output['condition_graph'] = condition_graph
        output['reaction_centre'] = reaction_centre
        output['one_hop'] = one_hop
        output['two_hop'] = two_hop
        output['three_hop'] = three_hop
        output['mlabes'] = mlabes
        if self.use_frag is not None:
            if 'brics' in self.use_frag:
                all_frag_ids = np.unique(frag_label)
                if len(all_frag_ids) == 1: # only one fragment
                    return output
                
                frag_ids = np.unique(frag_label[reaction_centre])
                if len(frag_ids) == 1:
                    frag_id = frag_ids[0]
                    frag_idx =  np.where(frag_label == frag_id)[0]
                    output['reaction_centre'] = list(frag_idx)


        # Output illustration:
        #   graph: Graph(x, edge_index, edge_attr)    
        #   idx_list: a np array of every atom in the reaction start from 0 to num_of_atoms-1
        #   
        return output





class ReactionDatasetMAT(Dataset):

    """
    Dataset to load reaction molecules, with reaction centre masking or reaction centre labels
    """

    def __init__(self, data_path, atom_vocab=None):
        super().__init__()
        # self.atom_vocab = atom_vocab
        # self.mask_stage = mask_stage
        self.atom_vocab = MolVocab.load_vocab(atom_vocab)
        self.atom_vocab_size = len(self.atom_vocab)
        print("Atom Vocab Size:", self.atom_vocab_size)
        def load_dataset(data_path):
            data = pd.read_csv(data_path)
            valid_idx = (data['reaction_centre'].isna()==False)
            return data[valid_idx], valid_idx

        self.data, valid_idx = load_dataset(data_path)
        print("Loaded Dataset")
    

            



        self.length = len(self.data)
        print("Number of lines: " + str(self.length))

    def __len__(self):
        return self.length    


    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        primary_smi = [data_row['primary_molecule']]
        #print(data_row['condition_molecule'])
        
        try:
            non_condition = math.isnan(data_row['condition_molecule'])
            condition_smi = []
        except:
            condition_smi = data_row['condition_molecule'].split('.')
        
        molecule_list = primary_smi + condition_smi
        mol_lst, mlabes = conver_mlabels(molecule_list, self.atom_vocab)
        while len(mlabes) > 512: #max atom number
            print(f'big number {len(mlabes)}')
            idx = idx + 1
            data_row = self.data.iloc[idx]
            primary_smi = [data_row['primary_molecule']]
            #print(data_row['condition_molecule'])
            
            try:
                non_condition = math.isnan(data_row['condition_molecule'])
                condition_smi = []
            except:
                condition_smi = data_row['condition_molecule'].split('.')
            
            molecule_list = primary_smi + condition_smi
            mol_lst, mlabes = conver_mlabels(molecule_list, self.atom_vocab)
            
        
        reaction_centre = [int(i) for i in data_row['reaction_centre'].split('.')]
        one_hop = [int(i) for i in data_row['one_hop'].split('.')]
        two_hop = [int(i) for i in data_row['two_hop'].split('.')]
        three_hop = [int(i) for i in data_row['three_hop'].split('.')]


        
        
        
        # print(molecule_list)
        graph_input, idx_list, molecule_idx = mat_handle_mol(mol_lst, reaction_centre)
        afm_all, adj_matrix, dist_matrix, afm_mask_all = graph_input

        output = {}
        
        output['node_features'] = afm_all
        output['adjacency_matrix'] = adj_matrix
        output['distance_matrix'] = dist_matrix
        output['node_features_mask'] = afm_mask_all
        
        output['idx'] = idx
        
        output['idx_list'] = idx_list
        output['molecule_idx'] = molecule_idx
        # output['condition_graph'] = condition_graph
        output['reaction_centre'] = reaction_centre
        output['one_hop'] = one_hop
        output['two_hop'] = two_hop
        output['three_hop'] = three_hop
        output['mlabes'] = mlabes

        # Output illustration:
        #   graph: Graph(x, edge_index, edge_attr)    
        #   idx_list: a np array of every atom in the reaction start from 0 to num_of_atoms-1
        #   
        return output


@dataclass
class MATCollator:
    mask_batch: bool = True
    mask_stage: str = "reaction_centre"
    
    def __call__(self, raw_batch):
        
        batch = {}
        adjacency_list, features_list, distance_list, mask_features_list = mol_collate_func(raw_batch)
        batch['adjacency_list'] = adjacency_list
        batch['features_list'] = features_list
        batch['distance_list'] = distance_list
        batch['mask_features_list'] = mask_features_list
        
        
        batch['idx_list'] = [ele['idx_list'] for ele in raw_batch]
        batch['molecule_idx'] = [ele['molecule_idx'] for ele in raw_batch]
        batch['reaction_centre'] = [ele['reaction_centre'] for ele in raw_batch]
        batch['mlabes'] = [ele['mlabes'] for ele in raw_batch]
        
        return batch


@dataclass
class CustomCollator:
    mask_batch: bool = True
    mask_stage: str = "reaction_centre"
    
    def __call__(self, raw_batch):
        
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
        
        mol_idx = []
        # reaction_centre = []
        mask_idx = []
        # one_hop = []
        # two_hop = []
        # three_hop = []
        mlabes = []
        graph_inputs = []
        batch = {}
        # condition_idx = []
        primary_idx = []

        rc_idx = []


        for idx, case in enumerate(raw_batch):
            mask_idx.append(case[self.mask_stage])
            # reaction_centre.append(case['reaction_centre'])
            # one_hop.append(case['one_hop'])
            # two_hop.append(case['two_hop'])
            # three_hop.append(case['three_hop'])
            mol_idx.append(case['molecule_idx'])
            mlabes.append(case['mlabes'])
            
            # condition_idx.append([i for i, x in enumerate(case['molecule_idx']) if x != 0])
            primary_idx.append([i for i, x in enumerate(case['molecule_idx']) if x == 0])
            
            graph_inputs.append(case['graph'])

            rc_idx.append(case['reaction_centre'])
            
        graph_inputs = Batch.from_data_list(graph_inputs)
        mlabes = [i for sublist in mlabes for i in sublist]
        
        # Add Masking to graphs
        
        mask_idx_batch = []
        for batch_idx, mask in enumerate(mask_idx):
            mask_idx_batch.extend([i+graph_inputs.ptr[batch_idx] for i in mask])
        mask_idx_batch = [int(t.item()) for t in mask_idx_batch]
        
        graph_mask = mol_mask(graph_inputs, mask_idx_batch)

        rc_idx_batch = []
        for batch_idx, rc in enumerate(rc_idx):
            rc_idx_batch.extend([i + graph_inputs.ptr[batch_idx] for i in rc])    
        # It gives the indexes of all reaction centres atoms in a batch (include all atoms in primary reactants and condition reactants)
        rc_idx_batch = torch.tensor([int(t.item()) for t in rc_idx_batch])

        pri_index_list = torch.tensor(np.concatenate(mol_idx))
        pri_index_list = (pri_index_list==0).nonzero().squeeze(1)
        
        primary_label = torch.ones(graph_inputs.x.shape[0], dtype=torch.double) * -1
        primary_label[pri_index_list] = 0
        primary_label[rc_idx_batch] = 1
        
        batch['graph'] = graph_mask
        batch['mlabes'] = mlabes
        # batch['condition_idx'] = condition_idx
        batch['primary_idx'] = primary_idx
        batch['primary_label'] = primary_label
        return batch

@dataclass
class IdentificationCollator:
    mask_batch: bool = True
    mask_stage: str = "reaction_centre"
    
    def __call__(self, raw_batch):
        mol_idx = []
        rc_idx = []
        graph_inputs = []
        batch = {}
        condition_idx = []
        primary_idx = []
        
        for idx, case in enumerate(raw_batch):
            rc_idx.append(case[self.mask_stage])
            mol_idx.append(case['molecule_idx'])
            graph_inputs.append(case['graph'])
            
        graph_inputs = Batch.from_data_list(graph_inputs)
        rc_idx_batch = []
        for batch_idx, rc in enumerate(rc_idx):
            rc_idx_batch.extend([i+graph_inputs.ptr[batch_idx] for i in rc])    
        # It gives the indexes of all reaction centres atoms in a batch (include all atoms in primary reactants and condition reactants)
        rc_idx_batch = torch.tensor([int(t.item()) for t in rc_idx_batch])
        pri_index_list = torch.tensor(np.concatenate(mol_idx))
        pri_index_list = (pri_index_list==0).nonzero().squeeze(1)
        
        primary_label = torch.ones(graph_inputs.x.shape[0], dtype=torch.double) * -1
        primary_label[pri_index_list] = 0
        primary_label[rc_idx_batch] = 1
        
        # primary_label = torch.tensor(primary_label, dtype=torch.double)
        batch['graph'] = graph_inputs
        batch['primary_idx'] = pri_index_list
        batch['position_centre'] = rc_idx_batch
        batch['primary_label'] = primary_label
        return batch
        
        # graph: GraphBatch(x, edge_attr, edge_index)
        # primary_idx: a 1-d tensor of length (number of all atoms in primary reactants in the batch), include the index of all primary reactants in the batch
        # position_centre: a 1-d tensor of length (number of all atoms in reaction centres in the batch), include the index of all reactant centres in the batch
        # primary_label: a 1-d tensor of length (all atoms in the batch, AKA graph.x.shape[0]), it contains (-1, 0, 1). where -1 stands for condition reactants, 0 for primary atoms but negative reaction centres, and 1 for reaction centres 
        
        
@dataclass
class MaskIndentiCollator:
    def __init__(self, mask_batch = True, mask_stage = "reaction_centre"):
        super(MaskIndentiCollator, self).__init__()
        self.mask_col = CustomCollator(mask_batch=mask_batch, mask_stage=mask_stage)
        self.indenti_col = IdentificationCollator(mask_batch=mask_batch, mask_stage="reaction_centre")

    def __call__(self, raw_batch):
        mask_batch = self.mask_col(raw_batch)
        indenti_batch = self.indenti_col(raw_batch)
        return mask_batch, indenti_batch

@dataclass
class USPTO1kTPLCollator:
    def __call__(self, raw_batch):
        graph_inputs = []
        mol_idx = []
        
        batch = {}

        for idx, case in enumerate(raw_batch):
            graph_inputs.append(case['graph'])
            mol_idx.append(case['mol_idx'])
            
        graph_inputs = Batch.from_data_list(graph_inputs)
        
        batch['graph'] = graph_inputs
        batch['mol_idx'] = mol_idx
        return batch
            
            
            
class USPTO_1k_TPL(Dataset):
    """
        Dataset for preprocessing the USPTO-1k-TPL Dataset for reaction type classification 
    """

    def __init__(self, dataset, reagents=False, filter=False, filter_thres=0.5):
        super().__init__()
        self.data = dataset
        if filter:
            self.data = self.data[self.data['confidence']>=filter_thres]
        self.reagents = reagents
        self.length = len(self.data)
        print("Number of lines: " + str(self.length))
        
    def __len__(self):
        return self.length
    
    def smiles2graph(self, molecule_list, reactant_ind):
        """
        Converts REACTION SMILES list to graphs.
        It returns a molecule index list, where graphs of reactants are denoted to 0, and graphs of products are denoted to 1
        """
        atom_idx_list = [] 
        edges_list = []
        edge_features_list = []
        atom_features_list = []
        start_idx = 0
        for mol_idx, smiles in enumerate(molecule_list):
            mol = Chem.MolFromSmiles(smiles)
            
        # atoms
            for atom in mol.GetAtoms():
                if mol_idx < reactant_ind:
                    atom_idx_list.append(0)
                else:
                    atom_idx_list.append(1)
                atom_features_list.append(atom_to_feature_vector(atom, False))
        
        # bonds
        
            num_bond_features = 2 # bond type, bond direction
            
            if len(mol.GetBonds()) > 0:  # mol has bonds
                
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx() + start_idx
                    j = bond.GetEndAtomIdx() + start_idx

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)
                # print(edges_list)
                
                # print(edge_index)
                # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                

            else:  # mol has no bonds
                continue
            start_idx += mol.GetNumAtoms()
        if len(edges_list):
            edge_index = np.array(edges_list, dtype=np.int64)
            edge_attr = np.array(edge_features_list, dtype=np.int64)
        else:
            edge_index = np.empty((0, 2), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        x = np.array(atom_features_list, dtype=np.int64)
        graph = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index.T), edge_attr=torch.tensor(edge_attr))
        
        return graph, np.array(atom_idx_list, dtype=np.int64)
        
               
        
    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        
        reactant = data_row['reactants']
        if self.reagents:
            reagents = data_row['reagents']
            if type(reagents) == str:
                reagents = reagents.split('.')
            
        product = data_row['products']
        label = data_row['labels']
        
        reactant = reactant.split('.')
        product = product.split('.')
        if self.reagents and type(reagents) == str:
            reactant_ind = len(reactant) + len(reagents)
            reactant_ls = reactant + reagents
        else:
            reactant_ind = len(reactant)
            reactant_ls = reactant
        
        graph_rc, _ = self.smiles2graph(reactant_ls, reactant_ind)
        graph_pd, _ = self.smiles2graph(product, reactant_ind)
        
        graph_rc.idx = idx
        graph_pd.idx = idx
    #    graph.y = label
        
        
        output = {}
        output['graph_rc'] = graph_rc
        output['graph_pd'] = graph_pd
        output['y'] = label
        
        return output
        
