from rdkit.Chem import BRICS
from collections import defaultdict
from random import shuffle

def brics_decomp(mol, addition_rule=False, return_all_bonds=True):
    """
    return break bonds, use additional rule or not
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    all_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])
        all_bonds.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))

    cut_bonds_set = [list(ele[0]) for ele in res]
    
    
    if not addition_rule:
        if return_all_bonds:
            return cut_bonds_set, all_bonds
        else:
            return cut_bonds_set

    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge breaks
    cut_bonds_set.extend(breaks)
    if return_all_bonds:
        return cut_bonds_set, all_bonds
    else:
        return cut_bonds_set


def fragment_graph_cutbonds(mol, cut_bonds_set):
    mol_num = len(list(mol.GetAtoms()))
    bond_set = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_set.append([a1, a2])
    
    left_bond_set = []
    
    for ele in bond_set:
        if [ele[0], ele[1]] not in cut_bonds_set and \
            [ele[1], ele[0]] not in cut_bonds_set:
                left_bond_set.append(ele)            
    
    left_bond_set = [list(ele) for ele in list(left_bond_set)]
    graph = defaultdict(list)

    for x, y in left_bond_set:
        graph[x].append(y)
        graph[y].append(x)
    
    visited = set()

    labels = [-1 for _ in range(mol_num)]
    
    def dfs(i, lb=-1):
        visited.add(i)
        labels[i] = lb
        for j in graph[i]:
            if j not in visited:
                dfs(j, lb)
    
    lb = 0
    for i in range(mol_num):
        if i not in visited:
            dfs(i, lb)
            lb += 1
    
    return labels


def brics_decompose_mol(mol, addition_rule=False):
    cut_bonds_set, all_bonds = brics_decomp(mol, addition_rule=addition_rule, return_all_bonds=True)
    g_labels_brics = fragment_graph_cutbonds(mol, cut_bonds_set)
    return g_labels_brics