import copy
import os
import random
import warnings

import numpy as np
import scipy
import scipy.spatial as spa
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from datasets.conformer_matching import (get_torsion_angles,
                                         optimize_rotatable_bonds)
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, GetPeriodicTable, RemoveHs
from rdkit.Chem import rdMolDescriptors as rdescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Geometry import Point3D
from utils.torsion import get_transformation_mask
# from scipy import spatial
# from scipy.special import softmax
# from torch_cluster import radius_graph




DEBUG = False
biopython_parser = PDBParser()    
periodic_table = GetPeriodicTable()
NUM_PHORETYPE = 11
PI = float(np.pi)
PHORETYPES = ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']
PHORE_SMARTS = {
    'MB': {
        '*-P(-O)(-O)=O': [2, 3, 4],
        '*-S(-O)=O': [2, 3],
        '*-S(=O)(-O)=O': [2, 3, 4],
        '*-S(-*)=O': [3],
        '*-C(-O)=O': [2, 3],
        '[O^3]': [0],
        '*-C(-C(-F)(-F)-F)=O': [6],
        '[OH1]-P(-*)(-*)=O': [0, 4],
        '*-C(-N-*)=O': [4],
        '*-[CH1]=O': [2],
        '*-N(-*)-N=O': [4],
        '*-C(-S-*)=O': [4],
        'O=C(-C-O-*)-C-[OH1]': [0],
        '*-C(-S-*)=O': [4],
        '*-C(-C(-[OH1])=C)=O': [5],
        '[S^3D2]': [0],
        '*=N-C=S': [3],
        'S=C(-N-C(-*)=O)-N-C(-*)=O': [0],
        '[#7^2,#7^3;!$([n;H0;X3]);!+;!+2;!+3]': [0],
        '[C,#1]-[Se&H1]': [1],
        'C1:C:C:C:S:1': [4],
        'O2:C:C:C:C:2': [0],
        'a[O,NH2,NH1,SX2,SeH1]': [1],
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
    },

    'NE': {
        '[CX3,SX3,PD3](=[O,S])[O;H0&-1,OH1]': [1, 2],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O;H0&-1,OH1]': [1, 2, 3],
        '[PX4](=[O,S])([O;H0&-1,OH1])[O][*;!H]': [1, 2],
        '[SX4](=[O,S])(=[O,S])([O;H0&-1,OH1])': [1, 2, 3]
    },

    'PO': {
        '[+;!$([N+]-[O-])]': [0],
        'N-C(-N)=N': [1]
    },

    'HD': {
        '[#7,#8,#16;+0,+1,+2;!H0]': [0]
    },

    'HA': {'[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]' : [0]
        # '[S;$(S=C);-0,-1,-2,-3]': [0],
    },
    # 'HA': {
    #     '[#7;!$([#7]~C=[N,O,S]);!$([#7]~S=O);!$([n;H0;X3]);!$([N;H1;X3]);-0,-1,-2,-3]': [0],
    #     '[O,F;-0,-1,-2,-3]': [0],
    #     '[S^3;X2;H0;!$(S=O);-0,-1,-2,-3]': [0],
    #     # '[S;$(S=C);-0,-1,-2,-3]': [0],
    # },

    'CV':{
        '[N]#[C]-[C,#1]': [1],
        '[C,#1]-[C]1-[C](-[C,#1])-[O]-1': [1, 2],
        '[C]=[C]-[C](-[N&H1]-[C,#1])=[O]': [0],
        '[S&H1]-[C,#1]': [0],
        '[C,#1]-[C]1-[C](-[C,#1])-[N]-1': [1, 2],
        '[C]=[C]-[S](=[O])(-[C,#1])=[O]': [0],
        '[F,Cl,Br,I]-[C]-[C,#1]': [1],
        '[C,#1]-[C](-[F,Cl,Br,I])-[C](-[C,N,O]-[C,#1])=[O]': [1],
        '[O]=[C](-[N]-[C,#1])-[C]#[C]': [5],
        '[C,#1]-[S](-[C,#1])=[O]': [1],
        '[C,#1]-[Se&H1]': [1],
        '[O]=[C](-[O]-[C,#1])-[C]#[C]': [5],
        '[S]=[C]=[N]-[C,#1]': [1],
        '[C,#1]-[S]-[S]-[C,#1]': [1, 2],
        '[C,#1]-[N,O]-[C](-[N,O]-[C,#1])=[O]': [2],
        '[C,#1]-[C](-[C](-[N]-[C,#1])=[O])=[O]': [1],
        '[C,#1]-[B](-[O&H1])-[O&H1]': [1],
        '[C,#1]-[C&H1]=[O]': [1],
        '[C,#1]-[S](-[F])(=[O])=[O]': [1],
        '[C,#1]-[S](-[C]=[C])(=[O])=[O]': [3],
        '[F,Cl,Br,I]-[C]-[C](-[C,#1])=[O]': [1]
    },
    'AR': {'[a]': [0]},
    'CR': {'[a]': [0], 
           '[+;!$([N+]-[O-])]': [0],
           'N-C(-N)=N': [1],
           },
    'XB': {'[#6]-[Cl,Br,I;X1]': [1]},
    'HY': {
            # refered to hydrophobic atom in 
            # /home/worker/software/anaconda3/envs/diffphore/lib/python3.9/site-packages/rdkit/Data/BaseFeatures.fdef
            '[c,s,S&H0&v2,Br,I,$([#6;+0;!$([#6;$([#6]~[#7,#8,#9])])])]': [0]
        }
    }

atom_radiuses = [periodic_table.GetRvdw(n) for n in range(1, 119)] + [0.0]

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 0)  # number of scalar features

rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 0)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 0)


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            # allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
        ])

    return torch.tensor(atom_features_list)


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_lig_graph(mol, complex_graph):
    lig_coords = torch.from_numpy(mol.GetConformer().GetPositions()).float()
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]] if bond.GetBondType() != BT.UNSPECIFIED else [0, 0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph['ligand'].x = atom_feats
    complex_graph['ligand'].pos = lig_coords
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_index = edge_index
    complex_graph['ligand', 'lig_bond', 'ligand'].edge_attr = edge_attr
    return


def generate_conformer(mol):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('[I] rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # else:
    #    AllChem.MMFFOptimizeMolecule(mol_rdkit, confId=0)


def get_lig_graph_with_matching(mol_, complex_graph, popsize, maxiter, matching, keep_original, num_conformers, remove_hs):
    if matching:
        mol_maybe_noh = copy.deepcopy(mol_)
        if remove_hs:
            mol_maybe_noh = RemoveHs(mol_maybe_noh, sanitize=True)
        if keep_original:
            complex_graph['ligand'].orig_pos = mol_maybe_noh.GetConformer().GetPositions()

        rotable_bonds = get_torsion_angles(mol_maybe_noh)
        if not rotable_bonds: print("no_rotable_bonds but still using it")

        for i in range(num_conformers):
            mol_rdkit = copy.deepcopy(mol_)

            mol_rdkit.RemoveAllConformers()
            mol_rdkit = AllChem.AddHs(mol_rdkit)
            generate_conformer(mol_rdkit)
            if remove_hs:
                mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
            mol = copy.deepcopy(mol_maybe_noh)
            if rotable_bonds:
                optimize_rotatable_bonds(mol_rdkit, mol, rotable_bonds, popsize=popsize, maxiter=maxiter)
            mol.AddConformer(mol_rdkit.GetConformer())
            rms_list = []
            AllChem.AlignMolConformers(mol, RMSlist=rms_list)
            mol_rdkit.RemoveAllConformers()
            mol_rdkit.AddConformer(mol.GetConformers()[1])

            if i == 0:
                complex_graph.rmsd_matching = rms_list[0]
                get_lig_graph(mol_rdkit, complex_graph)
                # complex_graph['compound_pair'].x = extract_pair_distribution(mol_rdkit, use_LAS_constrains, 
                #                                                              remove_hs=remove_hs, sanitize=True)
                # print("First Time Calculating ...")
            else:
                if torch.is_tensor(complex_graph['ligand'].pos):
                    complex_graph['ligand'].pos = [complex_graph['ligand'].pos]
                complex_graph['ligand'].pos.append(torch.from_numpy(mol_rdkit.GetConformer().GetPositions()).float())

    else:  # no matching
        complex_graph.rmsd_matching = 0
        if remove_hs: mol_ = RemoveHs(mol_)
        if keep_original:
            complex_graph['ligand'].orig_pos = mol_.GetConformer().GetPositions()
        get_lig_graph(mol_, complex_graph)

    edge_mask, mask_rotate = get_transformation_mask(complex_graph)
    complex_graph['ligand'].edge_mask = torch.tensor(edge_mask)
    complex_graph['ligand'].mask_rotate = mask_rotate
    complex_graph.no_hydrogen = remove_hs

    return


def generate_ligand_phore_feat(mol, data=None, remove_hs=True):
    """
    Ligand pharmacophore type, norm & angle values
    data['ligand'].phorefp
    data['ligand'].norm
    data['ligand'].norm_angle1
    data['ligand'].norm_angle2
    """
    mol = RemoveHs(mol) if remove_hs else mol
    # rdPartialCharges.ComputeGasteigerCharges(mol)
    coords = mol.GetConformer().GetPositions()
    lig_phorefps = [] # [N, NUM_PHORETYPE]
    lig_norms = [] # [N, NUM_PHORETYPE, 3]
    lig_norm_angles1 = [] # [N, NUM_PHORETYPE]
    lig_norm_angles2 = [] # [N, NUM_PHORETYPE]

    mol = analyze_phorefp(mol, remove_hs=remove_hs)

    for atom in mol.GetAtoms():
        phorefp, norm, angles1, angles2 = fetch_phorefeature(atom, coords)
        lig_phorefps.append(phorefp)
        lig_norms.append(norm)
        lig_norm_angles1.append(angles1)
        lig_norm_angles2.append(angles2)
    
    if data is not None:
        data['ligand'].ph = torch.tensor([int(mol.GetProp(f"_{fp}")) if mol.HasProp(f"_{fp}")  else 0 for fp in PHORETYPES])
        data['ligand'].phorefp = torch.tensor(lig_phorefps).float()
        data['ligand'].norm = torch.tensor(np.array(lig_norms)).float().reshape(len(lig_norms), -1)
        data['ligand'].norm_angle1 = torch.tensor(lig_norm_angles1).float()
        data['ligand'].norm_angle2 = torch.tensor(lig_norm_angles2).float()

    return lig_phorefps, lig_norms, lig_norm_angles1, lig_norm_angles2


def analyze_phorefp(mol, remove_hs=True):
    _mol = Chem.AddHs(mol)
    hy_check(_mol)
    ha_check(_mol)
    if remove_hs: 
        _mol = Chem.RemoveHs(_mol)
    [phore_check(_mol, phoretype) for phoretype in PHORETYPES if phoretype not in ['HY', 'HA']]
    return _mol


def fetch_phorefeature(atom, coords):
    phorefp = check_atom_phoretype(atom)
    norm, angles1, angles2 = calculate_phore_norms(atom, phorefp, coords)
    return phorefp, norm, angles1, angles2


def check_atom_phoretype(atom):
    phorefp = [0] * NUM_PHORETYPE
    prop_dict = atom.GetPropsAsDict()
    for i in range(NUM_PHORETYPE):
        phorefp[i] = 1 if PHORETYPES[i] in prop_dict and prop_dict[PHORETYPES[i]] == True else 0
    return phorefp


def phore_check(mol, phoretype='MB', setTrue=True, debug=DEBUG):
    idxes = []
    if phoretype in PHORE_SMARTS:
        count = 0
        for smarts, loc in PHORE_SMARTS[phoretype].items():
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            count += len(matches) * len(loc)
            for match in matches:
                for l in loc:
                    idxes.append(match[l])
                    # idxes.append(l)
                    if setTrue:
                        at = mol.GetAtomWithIdx(match[l])
                        at.SetBoolProp(phoretype, True)
                        if debug:
                            print(smarts, at.GetSymbol()+str(at.GetIdx()), f"Set as {phoretype}")
        idxes = list(set(idxes))
        if phoretype != 'AR':
            if phoretype == 'CR':
                mol.SetProp('_CR', str(int(mol.GetProp("_AR"))+int(mol.GetProp("_PO"))))
            elif phoretype == 'NE':
                mol.SetProp(f"_{phoretype}", str(count))
            else:
                mol.SetProp(f"_{phoretype}", str(len(idxes)))
        else:
            mol.SetProp(f"_{phoretype}", str(rdescriptors.CalcNumAromaticRings(mol)))

    # print(idxes)
    return idxes


def ha_check(mol, debug=True):
    idxes = phore_check(mol, 'HA', setTrue=True)
    # print(f"HA: index {idxes}")
    # for idx in idxes:
    #     at = mol.GetAtomWithIdx(idx)
    #     print(calAccSurf(at))
    #     if calAccSurf(at) >= 0.02:
    #         at.SetBoolProp('HA', True)
    #         if debug:
    #             print(at.GetSymbol()+str(at.GetIdx()), f"set as HA")


def hy_check(mol, follow_ancphore=False):
    if follow_ancphore:
        labelLipoAtoms(mol)
        atomSet = set([])
        for at in mol.GetAtoms():
            at.SetBoolProp('HY', False)
            atomSet.add(at.GetIdx())
            if at.GetAtomicNum() != 1:
                t = at.GetDoubleProp('pcharge')
                if float_eq(t, 0):
                    at.SetDoubleProp('pcharge', calAccSurf(at, 'HY') * t)
        
        # Rings smaller than 7 atoms
        for ring in Chem.GetSSSR(mol):
            if len(ring) < 7:
                lipoSum = 0
                for at_idx in ring:
                    lipoSum += mol.GetAtomWithIdx(at_idx).GetDoubleProp('pcharge')
                    if at_idx in atomSet:
                        atomSet.remove(at_idx)
                if lipoSum > 9.87:
                    for at_idx in ring:
                        mol.GetAtomWithIdx(at_idx).SetBoolProp('HY', True)

        # Atoms with three or more bonds
        for at_idx in atomSet:
            at = mol.GetAtomWithIdx(at_idx)
            collected_idx = [at_idx]
            if at.GetTotalNumHs() > 2:
                lipoSum = at.GetDoubleProp('pcharge')
                for bond in at.GetBonds():
                    neib = bond.GetOtherAtom(at)
                    if neib.GetTotalNumHs() == 1 and at.GetAtomicNum() != 1:
                        lipoSum += neib.GetDoubleProp('pcharge')
                        collected_idx.append(neib.GetIdx())
                if lipoSum > 9.87:
                    for at_idx in collected_idx:
                        mol.GetAtomWithIdx(at_idx).SetBoolProp('HY', True)
    else:
        phore_check(mol, 'HY', setTrue=True)


def calAccSurf(atom, mode='HA'):
    mol = atom.GetOwningMol()
    coords = mol.GetConformer().GetPositions()
    coord = coords[atom.GetIdx()]
    radius = periodic_table.GetRvdw(atom.GetAtomicNum())
    if mode == 'HA':
        radius = 1.8 
    elif mode == 'HY':
        radius = periodic_table.GetRvdw(atom.GetAtomicNum())


    arclength = 1.0 / np.sqrt(np.sqrt(3.0) * 2.0)
    dphi = arclength / radius
    nlayer = int(np.pi / dphi) + 1
    phi = 0.0
    sphere = []
    for i in range(nlayer):
        rsinphi = radius * np.sin(phi)
        z = radius * np.cos(phi)
        dtheta = 2 * np.pi if rsinphi == 0 else arclength / rsinphi
        tmpNbrPoints = int(2 * np.pi / dtheta)
        if tmpNbrPoints <= 0:
            tmpNbrPoints = 1
        dtheta = np.pi * 2.0 / tmpNbrPoints
        theta = 0 if i % 2 else np.pi
        for j in range(tmpNbrPoints):
            sphere.append(np.array([rsinphi*np.cos(theta)+coord[0], rsinphi*np.sin(theta)+coord[1], z+coord[2]]))
            theta += dtheta
            if theta > np.pi * 2:
                theta -= np.pi * 2
        phi += dphi
    
    aList = []
    if mode == 'HA':
        aList = [at for at in mol.GetAtoms()\
                    if np.sum(np.square(coords[at.GetIdx()] - coord)) \
                        <= np.square(3.0 + periodic_table.GetRvdw(at.GetAtomicNum()))\
                    and at.GetIdx() != atom.GetIdx()]
    elif mode == 'HY':
        aList = [at for at in mol.GetAtoms()\
                    if np.sum(np.square(coords[at.GetIdx()] - coord)) \
                        <= np.square(periodic_table.GetRvdw(atom.GetAtomicNum()) + \
                                     periodic_table.GetRvdw(at.GetAtomicNum()) + 2.8)\
                    and at.GetIdx() != atom.GetIdx()]
    
    delta = 1 if mode != 'HY' else 1.4 / radius

    nbrAccSurfPoints = 0
    prob_r = 1.2 if mode != 'HY' else 1.4
    isAccessible = True
    for s in sphere:
        p = s if mode != 'HY' else (s - coord) * delta + s
        for at in aList:
            distSq = np.sum(np.square(coords[at.GetIdx()] - p))
            r = periodic_table.GetRvdw(at.GetAtomicNum())
            sumSq = np.square(r+prob_r)
            if distSq <= sumSq:
                isAccessible = False
                break
        if isAccessible:
            nbrAccSurfPoints += 1
    if mode == 'HA':
        return float(nbrAccSurfPoints / len(sphere))
    elif mode == 'HY':
        return float(nbrAccSurfPoints / len(sphere) * 4 * np.pi * radius * radius)


def labelLipoAtoms(m):
    # pcharges = [1.0] * len(m.GetAtoms())
    for at in m.GetAtoms():
        at.SetDoubleProp("pcharge", 1.0)

    for at in m.GetAtoms():
        at_num = at.GetAtomicNum()
        if at_num == 1:
            at.SetDoubleProp("pcharge", 0.0)

        elif at_num == 7:
            at.SetDoubleProp("pcharge", 0.0)
            if not at.GetIsAromatic():
                labelLipoNeighbors(at, 0.25)
                if at.GetTotalNumHs() != 0:
                    for bond in at.GetBonds():
                        neib = bond.GetOtherAtom(at)
                        neib.SetDoubleProp('pcharge', 0.0)
                        labelLipoNeighbors(neib, 0.0)

        elif at_num == 8:
            at.SetDoubleProp("pcharge", 0.0)
            if not at.GetIsAromatic():
                labelLipoNeighbors(at, 0.25)
                for bond in at.GetBonds():
                    neib = bond.GetOtherAtom(at)
                    if neib.GetAtomicNum() == 1:
                        for bond1 in at.GetBonds():
                            nneib = bond1.GetOtherAtom(at)
                            nneib.SetDoubleProp('pcharge', 0.0)
                            labelLipoNeighbors(nneib, 0.0)
                    if bond.GetBondType().name == "DOUBLE":
                        neib.SetDoubleProp('pcharge', 0.0)
                        for bond1 in neib.GetBonds():
                            nneib = bond1.GetOtherAtom(neib)
                            if nneib.GetIdx() == at.GetIdx():
                                continue
                            nneib.SetDoubleProp('pcharge', 0.0)
                            labelLipoNeighbors(nneib, 0.6)

        elif at_num == 16:
            for bond in at.GetBonds():
                neib = bond.GetOtherAtom(at)
                if neib.GetAtomicNum() == 1:
                    at.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.0)
                if bond.GetBondType().name == "DOUBLE":
                    at.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.6)
            
            if at.GetTotalNumHs() > 2:
                at.SetDoubleProp('pcharge', 0.0)
                for bond in at.GetBonds():
                    neib = bond.GetOtherBonds(at)
                    neib.SetDoubleProp('pcharge', 0.0)
                    labelLipoNeighbors(at, 0.6)

        if at.GetFormalCharge() != 0:
            for bond in at.GetBonds():
                neib = bond.GetOtherAtom(at)
                neib.SetDoubleProp('pcharge', 0.0)
                labelLipoNeighbors(neib, 0.0)
    
    for at in m.GetAtoms():
        value = at.GetDoubleProp('pcharge')
        if (float_eq(value, 0.36) or value < 0.25) and not float_eq(value, 0.15):
            at.SetDoubleProp('pcharge', 0.0)


def float_eq(a, b, epsilon=1e-6):
    return abs(a - b) <= epsilon
            

def labelLipoNeighbors(atom, value):
    for bond in atom.GetBonds():
        neib = bond.GetOtherAtom(atom)
        neib.SetDoubleProp('pcharge', value * neib.GetDoubleProp('pcharge'))


def calculate_phore_norms(atom, possible_phore_fp, coords):
    norm = [np.array([0, 0, 0])] * NUM_PHORETYPE
    angle1 = [0] * NUM_PHORETYPE
    angle2 = [0] * NUM_PHORETYPE

    # get atom coordinate and neighbor coordinates.
    atom_coord = coords[atom.GetIdx()]
    neibs = atom.GetNeighbors()
    neib_coords = [coords[at.GetIdx()] for at in neibs]
    num_root = len(neib_coords)
    root_coord = np.mean(neib_coords, axis=0)
    # print(atom.GetSymbol()+str(atom.GetIdx()), possible_phore_fp)

    # loop over the pharmacophore fingerprint to calculate pre-defined norms and angles.
    for idx, phoretype in enumerate(possible_phore_fp):
        if phoretype == 0:
            pass
        else:
            curr_phore = PHORETYPES[idx]
            if curr_phore == 'AR':
            # if curr_phore == 'AR' or  (curr_phore == 'CR' and possible_phore_fp[2] != 0):
                # print(neib_coords)
                two_neibs = random.sample(neib_coords, 2)
                curr_norm = np.cross(two_neibs[0] - atom_coord, two_neibs[1] - atom_coord)
                curr_norm = curr_norm / (np.linalg.norm(curr_norm) + 1e-12)
                norm[idx] = curr_norm
                angle1[idx] = 0.0
                angle2[idx] = PI

            else:
                curr_norm = atom_coord - root_coord
                curr_norm = curr_norm / (np.linalg.norm(curr_norm) + 1e-12)
                norm[idx] = curr_norm

                if curr_phore == 'MB':
                    if num_root == 1:
                        angle1[idx] = PI / 3.0
                        angle2[idx] = PI / 3.0
                    if num_root >= 2:
                        angle1[idx] = 0.0
                        angle2[idx] = 0.0

                elif curr_phore == 'HA':
                    if num_root == 1:
                        angle1[idx] = PI / 3.0
                        angle2[idx] = PI / 3.0
                    if num_root >= 2:
                        angle1[idx] = 0.0
                        angle2[idx] = 0.0

                elif curr_phore == 'HD':
                    if num_root == 1:
                        angle1[idx] = PI / 3.0
                        angle2[idx] = PI / 3.0
                    if num_root >= 2:
                        angle1[idx] = 0.0
                        angle2[idx] = 0.0

                elif curr_phore == 'XB':
                    angle1[idx] = 0.0
                    angle2[idx] = 0.0

    return norm, angle1, angle2


def write_mol_with_coords(mol, new_coords, path, skip_h=True):
    conf = mol.GetConformer()
    idx = 0
    for i in range(mol.GetNumAtoms()):
        # print(f"{mol.GetAtomWithIdx(i).GetSymbol()}")
        if skip_h and mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            continue
        x,y,z = new_coords.astype(np.double)[idx]
        idx += 1
        conf.SetAtomPosition(i,Point3D(x,y,z))
    if path.endswith(".sdf"):
        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()
    elif path.endswith(".mol"):
        Chem.MolToMolFile(mol, path)


def write_mol_with_multi_coords(mol, multi_new_coords, path, name, marker="", properties=None):
    w = Chem.SDWriter(path)
    if properties is not None:
        w.SetProps(list(properties.keys()))

    _mol = copy.deepcopy(mol)
    
    for idx, new_coords in enumerate(multi_new_coords):
        _mol.SetProp("_Name", f"{name}_{marker}_{idx}")
        if properties is not None:
            for k, v in properties.items():
                _mol.SetProp(f"{k}", f"{v[idx]}")

        conf = _mol.GetConformer()
        for i in range(_mol.GetNumAtoms()):
        # print(f"{mol.GetAtomWithIdx(i).GetSymbol()}")
            x,y,z = new_coords.astype(np.double)[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        w.write(_mol)
    w.close()


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.mol'):
        mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print(f"[E] RDKit was unable to read the molecule. {e}")
        return None

    return mol


def read_sdf_or_mol2(sdf_fileName, mol2_fileName):

    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            problem = False
        except Exception as e:
            problem = True

    return mol, problem


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float().reshape(-1, 16)
    return pair_dis_distribution


def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask


#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def extract_pair_distribution(mol, has_LAS_mask=False, remove_hs=True, sanitize=True, coords=None):
    _mol = Chem.RemoveHs(mol, sanitize=sanitize) if remove_hs else copy.deepcopy(mol)
    coords = _mol.GetConformer().GetPositions() if coords is None else coords
    LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(_mol) if has_LAS_mask else None
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask)
    
    return pair_dis_distribution


def lipinski_wt_limit(m):
    return Descriptors.MolWt(m) <= 500


def lipinski_logp_limit(m):
    return Descriptors.MolLogP(m) <= 5


def lipinski_hba_limit(m):
    return rdescriptors.CalcNumLipinskiHBA(m) <= 10


def lipinski_hbd_limit(m):
    return rdescriptors.CalcNumLipinskiHBD(m) <= 5


def lipinski_rt_limit(m):
    return rdescriptors.CalcNumRotatableBonds(m) <= 10


def lipinski_violations(m):
    return 5 - lipinski_wt_limit(m) - lipinski_logp_limit(m) - lipinski_hba_limit(m) - lipinski_hbd_limit(m) - lipinski_rt_limit(m)


def lipinski_rule_analysis(mol):
    wt = lipinski_wt_limit(mol)
    logp = lipinski_logp_limit(mol)
    hba = lipinski_hba_limit(mol)
    hbd = lipinski_hbd_limit(mol)
    rtb = lipinski_rt_limit(mol)
    violations = 5 - wt - logp - hba - hbd - rtb
    return violations