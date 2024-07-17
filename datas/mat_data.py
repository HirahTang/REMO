from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
import logging
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
# use_cuda=True
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False, mask_idx_lst=None):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all, y_all = [], []

    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all, y_all

# mol_lst: reaction item
def mat_handle_mol(mol_lst, mask_idx_lst, add_dummy_node=True, one_hot_formal_charge=True):
    
    
    idx_list, molecule_idx = [], []
    afm_lst = []
    afm_mask_lst = []
    # concat matrix: adj and dist
    all_num = 0
    new_mol_lst = []
    for idx, mol in enumerate(mol_lst):
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            AllChem.Compute2DCoords(mol)
        new_mol_lst.append(mol)
        # atom number may change
        all_num += mol.GetNumAtoms()
        if add_dummy_node:
            all_num += 1
    
    adj_matrix = np.eye(all_num)
    dist_matrix = np.eye(all_num)
    
    dist_matrix = np.full((all_num, all_num), np.inf)
    np.fill_diagonal(dist_matrix, 1)
    # dist_matrix = dist_matrix.masked_fill(dist_matrix==0, np.inf)
    
    start_idx = 0
    for idx, mol in enumerate(new_mol_lst):
        cur_atom_num = mol.GetNumAtoms()
        if add_dummy_node:
            cur_atom_num += 1
        molecule_idx.extend([idx for _ in range(cur_atom_num)]) # include the cls token
        idx_list.extend([i for i in range(cur_atom_num)]) # include the cls token
        if idx == 0:
            afm, adj, dist, maskafm = featurize_mol(mol, add_dummy_node, one_hot_formal_charge, mask_idx_lst)
            afm_mask_lst.append(maskafm)
        else:
            afm, adj, dist, _ = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            afm_mask_lst.append(afm)
        afm_lst.append(afm)
        
        adj_matrix[start_idx:start_idx + cur_atom_num, start_idx:start_idx + cur_atom_num] = adj
        dist_matrix[start_idx:start_idx + cur_atom_num, start_idx:start_idx + cur_atom_num] = dist
        
        
    
    afm_all = np.concatenate(afm_lst, axis=0)
    afm_mask_all = np.concatenate(afm_mask_lst, axis=0)
    
    if afm_all.shape[0] != adj_matrix.shape[0]:
        print('not match size between nodef and adjm')
        
    return (afm_all, adj_matrix, dist_matrix, afm_mask_all), idx_list, molecule_idx


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge, mask_idx_lst=None):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])
    if mask_idx_lst is not None:
        node_features_mask = np.copy(node_features)
        node_features_mask[mask_idx_lst] = get_atom_mask_features(one_hot_formal_charge)
    else:
        node_features_mask = None

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m
        
        if mask_idx_lst is not None:
            m = np.zeros((node_features_mask.shape[0] + 1, node_features_mask.shape[1] + 1))
            m[1:, 1:] = node_features_mask
            m[0, 0] = 1.
            node_features_mask = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m
    
    

    return node_features, adj_matrix, dist_matrix, node_features_mask


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)

def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    if shape[0] < array.shape[0]:
        print('error')
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array

def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    adjacency_list, distance_list, features_list, mask_features_list = [], [], [], []


    max_size = 0
    for molecule in batch:
        if molecule['adjacency_matrix'].shape[0] > max_size:
            max_size = molecule['adjacency_matrix'].shape[0]

    for molecule in batch:
        adjacency_list.append(pad_array(molecule['adjacency_matrix'], (max_size, max_size)))
        distance_list.append(pad_array(molecule['distance_matrix'], (max_size, max_size)))
        features_list.append(pad_array(molecule['node_features'], (max_size, molecule['node_features'].shape[1])))
        mask_features_list.append(pad_array(molecule['node_features_mask'], (max_size, molecule['node_features_mask'].shape[1])))

    
    return [torch.tensor(np.array(features), dtype=torch.float32) for features in (adjacency_list, features_list, distance_list, mask_features_list)]




def get_atom_mask_features(one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        -1,
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        -1,
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        -1,
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            -2,
            [-1, 0, 1]
        )
    else:
        attributes.append(1)

    attributes.append(-1)
    attributes.append(-1)

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


if __name__ == '__main__':
    smiles = ['CN(C)C(=O)c1ccc(cc1)OC']
    y_lst = [-1.874467]
    x_all, y_all = load_data_from_smiles(smiles, y_lst)
    
    