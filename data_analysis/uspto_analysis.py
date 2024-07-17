from USPTO_CONFIG import USPTO_CONFIG
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger
import pickle
RDLogger.DisableLog('rdApp.*')

def SmilesToInchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchi(mol)
    return inchi, mol

def main():

    uspto = pd.read_csv(USPTO_CONFIG.path)
#    uspto = uspto.iloc[:10000]
    reactant_dict = {}
    full = {}
    for index, row in tqdm(uspto.iterrows()):
        reactant = row['reactant'].split('.')
        for i in set(reactant):
           # print(i)
            try:
                temp_inchi, temp_mol = SmilesToInchi(i)
                atom_num = temp_mol.GetNumAtoms()
                #reactant_set.add(temp_inchi)
                if temp_inchi not in reactant_dict:
                    reactant_dict[temp_inchi] = {"count": 1, "atoms": atom_num}
                else:
                    reactant_dict[temp_inchi]['count'] += 1
                #print(reactant_dict)
                #with open("reactant.txt", "a+") as f0:
                #    f0.write(temp_inchi+"\n")
                #    f0.close()
                
                if temp_inchi not in full:
                    full[temp_inchi] = {"count": 1, "atoms": atom_num}
                else:
                    full[temp_inchi]['count'] += 1
            except:
                continue
                #with open("/sharefs/sharefs/failed_samples.txt", "a+") as f:
                #    f.write(i+'\n')
                #    f.close()
        product = row['product'].split('.')
        for i in set(product):
            try:
                temp_inchi, temp_mol = SmilesToInchi(i)
                atom_num = temp_mol.GetNumAtoms()
                #reactant_set.add(temp_inchi)
                if temp_inchi not in full:
                    full[temp_inchi] = {"count": 1, "atoms": atom_num}
                else:
                    full[temp_inchi]['count'] += 1
            except:
                continue
    #print(reactant)
    with open(USPTO_CONFIG.reactant_dict, "wb") as handle:
        pickle.dump(reactant_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(USPTO_CONFIG.full_dict, "wb") as handle:
        pickle.dump(full, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
