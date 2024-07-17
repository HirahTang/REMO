import sys
sys.path.append('..')
import pickle

from seq_graph_retro.utils.parse import get_reaction_info, extract_leaving_groups, get_reaction_core
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
from data_analysis.USPTO_CONFIG import USPTO_CONFIG
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')




def reaction_inspect(r, p):
    # Get reaction centre: {centre index}
    #r = uspto.iloc[n]['reactant']
    #p = uspto.iloc[n]['product']
    rxn = r+">>"+p
    # rxn = AllChem.ReactionFromSmarts(rxn)
    rxn_core, core_edits = get_reaction_core(r, p)
    return rxn, rxn_core, core_edits

def primary_molecule(r):
    # Get: list of primary reactants
    r_list = r.split('.')
    mol_list = []
    output_r = []
    for i in r_list:
        temp_mol = Chem.MolFromSmiles(i)
        temp_list = [atom.GetAtomMapNum() for atom in temp_mol.GetAtoms() if atom.GetAtomMapNum() > 0]
        involved_atoms = len(temp_list)
        #print(temp_list)
        if involved_atoms > 1:
            mol_list.append(temp_mol)
            output_r.append(i)
    return output_r, mol_list

def masking_atoms(mol, mask_list):
    
    # mol = Chem.MolFromSmiles(mol)
    masked = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in mask_list]
    return masked

def neighbour_mask(mol, atom_mask):
    output_mask = [i for i in atom_mask]

    # mol = Chem.MolFromSmiles(mol)
    
    for atom in mol.GetAtoms():
        if atom.GetIdx() in atom_mask:
            for nei in atom.GetNeighbors():
                output_mask.append(nei.GetIdx())
    
    return list(set(output_mask))
    
def reaction_record(reaction):
    output_dict = {}
    _, react_core, _ = reaction_inspect(reaction['reactant'], reaction['product'])
    react_mol, mol_list = primary_molecule(reaction['reactant'])
    for mol_smi, mol_ in zip(react_mol, mol_list):
        output_dict[mol_smi] = {}

        mask_list = masking_atoms(mol_, react_core)
        output_dict[mol_smi]['reaction_centre'] = mask_list

        one_hop = neighbour_mask(mol_, mask_list)
        output_dict[mol_smi]['one_hop'] = one_hop

        two_hop = neighbour_mask(mol_, one_hop)
        output_dict[mol_smi]['two_hop'] = two_hop

        three_hop = neighbour_mask(mol_, two_hop)
        output_dict[mol_smi]['three_hop'] = three_hop
    return output_dict


def main():
    result = {}
    uspto = pd.read_csv(USPTO_CONFIG.path)
    for index, row in tqdm(uspto.iterrows()):
        reaction = row['reactant'] + '>>' + row['product']
        if reaction in result:
            with open(USPTO_CONFIG.dup_raw, "a+") as f:
                f.write(reaction+'\n')
                f.close()
                
        try:
            record = reaction_record(row)
            result[reaction] = record
        except:
            with open(USPTO_CONFIG.problematic, "a+") as f:
                f.write(reaction+'\n')
                f.close()

    with open(USPTO_CONFIG.reaction_centre, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()