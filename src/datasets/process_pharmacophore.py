from collections import namedtuple
import copy
import os
import random
import time
from rdkit import Chem

import numpy as np
from scipy import linalg
import scipy.spatial as spa
import scipy
from scipy.special import softmax
import torch
import pandas as pd

from datasets.process_mols import generate_ligand_phore_feat, analyze_phorefp

ANCPHORE = os.path.join(os.path.dirname(__file__), '../../programs/AncPhore')
DEBUG = False

## Predefined NamedTuples
Phore = namedtuple('Phore', ['id', 'features', 'exclusion_volumes', 'clusters'])
PhoreFeature = namedtuple('PhoreFeature', ['type', 'alpha', 'weight', 'factor', 'coordinate', 
                                           'has_norm', 'norm', 'label', 'anchor_weight'])
Coordinate = namedtuple('Coordinate', ['x', 'y', 'z'])

## Pharmacophore properties
allowable_features_phore = {
    'possible_phore_type_list': ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX'],
    'possible_has_norm': [True, False],
    'possible_is_exlusion_volume': [True, False]
}

phore_feature_dims = (list(map(len, [
    allowable_features_phore['possible_phore_type_list'],
    allowable_features_phore['possible_has_norm'],
    allowable_features_phore['possible_is_exlusion_volume'],
])), 2)


"""
pre-defined weight for each pharmacophore type.
    1.5 //MECH
    1.2  // HD
    1.2  // HA
    1.0  // AR
    0.5  // HY
    1.5  // PO
    1.5  // NE
    1.0  // CV
    1.0  // EX
    1.0  // UNDEF
    1.0  // XB halogen bonding
    1.0  // CR cation-π interaction
"""
phore_pre_weight = [1.5, 1.2, 1.0, 1.5, 1.2, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0]


"""
pre-defined alpha (related with the radius of pharmacophores) for each type
    1.0,  // MB
    1.0,  // HD
    1.0,  // HA
    0.7,  // AR 
    0.7,  // HY
    1.0,  // PO
    1.0,  // NE
    1.0,  // CV
    0.837,// EX
    1.0,  // UNDEF
    1.0,  // XB
    0.7   // CR: AR is 0.7; PO is 1.0
"""
phore_pre_alpha = [1.0, 1.0, 0.7, 1.0, 1.0, 0.7, 1.0, 1.0, 0.7, 1.0, 0.837]
pi = 3.1415926


def parse_phore(phore_file=None, name=None, data_path=None, 
                skip_wrong_lines=True, verbose=False, epsilon=1e-6, skip_ex=False, cvs=False):
    """
    Parses a pharmacophore file and returns a list of pharmacophore objects.

    Args:
        phore_file (str, optional): Path to the pharmacophore file. If `name` and `data_path` are provided, this is ignored.
        name (str, optional): Name of the pharmacophore. Used to construct the file path if `data_path` is provided.
        data_path (str, optional): Base path to the data directory. Used with `name` to construct the file path.
        skip_wrong_lines (bool, optional): Whether to skip lines that cannot be parsed correctly. Defaults to True.
        verbose (bool, optional): Whether to print warnings if no pharmacophores are read. Defaults to False.
        epsilon (float, optional): Tolerance for clustering pharmacophore features. Defaults to 1e-6.
        skip_ex (bool, optional): Whether to skip exclusion volumes. Defaults to False.
        cvs (bool, optional): Whether to consider full set of CV (covalent) features. Defaults to False.

    Returns:
        list: A list of pharmacophore objects parsed from the file.

    Raises:
        FileNotFoundError: If the specified pharmacophore file is not found.

    Notes:
        - If both `name` and `data_path` are provided, the `phore_file` is constructed as `os.path.join(data_path, f"{name}/{name}_complex.phore")`.
        - The function reads the pharmacophore file line by line, parsing each line into pharmacophore features.
        - If `skip_wrong_lines` is False, the function will stop parsing upon encountering an incorrect line.
        - If `verbose` is True and no pharmacophores are read, a warning message is printed.
    """
    if name is not None and data_path is not None:
        phore_file = os.path.join(data_path, f"{name}/{name}_complex.phore")

    phores = []
    if phore_file is not None and os.path.exists(phore_file):
        with open(phore_file, 'r') as f:
            started, finished, correct = False, False, True
            id = ""
            phore_feats = []
            exclusion_volumes = []
            clusters = {}
            while True:
                record = f.readline().strip()
                if record:
                    if not started:
                        id = record
                        started = True
                    else:
                        phore_feat = parse_phore_line(record, skip_wrong_lines, cvs) if correct else False
                        if phore_feat is None:
                            finished = True
                        elif phore_feat == False:
                            correct = False
                        else:
                            if phore_feat.type != 'EX':
                                phore_feats.append(phore_feat)
                            else:
                                if not skip_ex:
                                    exclusion_volumes.append(phore_feat)
                            add_phore_to_cluster(phore_feat, clusters, epsilon)
                    if finished:
                        if len(phore_feats) and correct:
                            phore = Phore(id, copy.deepcopy(phore_feats), 
                                          copy.deepcopy(exclusion_volumes), copy.deepcopy(clusters))
                            phores.append(phore)
                        phore_feats = []
                        exclusion_volumes = []
                        clusters = {}
                        started, finished = False, False
                else:
                    break
    else:
        raise FileNotFoundError(f"The specified pharmacophore file (*.phore) is not found: `{phore_file}`")
    if verbose:
        if len(phores) == 0:
            print(f"[W] No pharmacophores read from the phore file `{phore_file}`")

    return phores


def add_phore_to_cluster(phore_feat, clusters, epsilon=1e-6):
    """
    Adds a pharmacophore feature to the appropriate cluster based on its coordinates.

    Args:
    phore_feat (object): The pharmacophore feature to be added. It must have a 'coordinate' attribute with 'x', 'y', and 'z' properties.
    clusters (dict): A dictionary where keys are coordinates (with 'x', 'y', and 'z' properties) and values are lists of pharmacophore features.
    epsilon (float, optional): The maximum distance between coordinates to consider them as the same cluster. Default is 1e-6.

    Returns:
    dict: The updated clusters dictionary with the new pharmacophore feature added to the appropriate cluster.
    """
    if phore_feat.coordinate in clusters:
        clusters[phore_feat.coordinate].append(phore_feat)
    else:
        if len(clusters) == 0:
            clusters[phore_feat.coordinate] = [phore_feat]
        else:
            flag = False
            for stored_coord in clusters:
                curr_coord = np.array([phore_feat.coordinate.x, phore_feat.coordinate.y, 
                                       phore_feat.coordinate.z])
                stored_coord = np.array([stored_coord.x, stored_coord.y, stored_coord.z])
                if np.sqrt(np.sum((stored_coord - curr_coord) ** 2)) <= epsilon:
                    clusters[stored_coord].append(phore_feat)
                    flag = True
                    break
            if not flag:
                clusters[phore_feat.coordinate] = [phore_feat]
    return clusters


def extract_random_phore_from_origin(phore, up_num=8, low_num=4, 
                                     sample_num=10, max_rounds=100, **kwargs):
    """
    Extracts random pharmacophores from the given origin pharmacophore.

    Args:
    phore (Phore): The origin pharmacophore from which to extract random pharmacophores.
    up_num (int, optional): The upper limit for the number of clusters to sample. Default is 8.
    low_num (int, optional): The lower limit for the number of clusters to sample. Default is 4.
    sample_num (int, optional): The number of random pharmacophores to extract. Default is 10.
    max_rounds (int, optional): The maximum number of rounds to attempt extraction. Default is 100.
    **kwargs: Additional keyword arguments.

    Returns:
    list: A list of extracted random pharmacophores.
    """
    phores = []
    collection = []
    coords = phore.clusters.keys()
    _round = 0
    while sample_num != 0:
        _round += 1
        num = min(random.choice(list(range(low_num, up_num))), len(coords))
        clusters = random.sample(coords, num)
        ex, feat = [], []
        for cluster in clusters:
            selected = random.choice(phore.clusters[cluster])
            if selected.type == "EX":
                ex.append(selected)
            else:
                feat.append(selected)
        collect = set(ex+feat)
        if collect not in collection:
            phores.append(Phore(f"{phore.id}_{sample_num}", 
                                copy.deepcopy(feat), copy.deepcopy(ex), {}))
            collection.append(collect)
            sample_num -= 1
        if _round >= max_rounds:
            break
    return phores


def generate_random_exclusion_volume(phore, ligand, remove_hs=True, low=3.0, up=5.0, 
                                     ex_dis=0.8, theta=15.0, num_ex=5, mode='radius', near_phore=True, cutoff=2.0, strict=True,
                                     use_non_phore=False, rounds=100, debug=DEBUG, only_surface_ex=False, **kwargs):
    """
    Generate random exclusion volumes for a given pharmacophore and ligand.
    Args:
    phore (Phore): The pharmacophore object.
    ligand (Mol): The ligand molecule.
    remove_hs (bool, optional): Whether to remove hydrogen atoms from the ligand. Default is True.
    low (float, optional): The lower bound for exclusion volume generation. Default is 3.0.
    up (float, optional): The upper bound for exclusion volume generation. Default is 5.0.
    ex_dis (float, optional): The exclusion distance. Default is 0.8.
    theta (float, optional): The angle for exclusion volume generation. Default is 15.0.
    num_ex (int, optional): The number of exclusion volumes to generate. Default is 5.
    mode (str, optional): The mode for exclusion volume generation. Default is 'radius'.
    near_phore (bool, optional): Whether to check if the exclusion volume is near the pharmacophore. Default is True.
    cutoff (float, optional): The cutoff distance for checking proximity to the pharmacophore. Default is 2.0.
    strict (bool, optional): Whether to use strict checking for proximity to the pharmacophore. Default is True.
    use_non_phore (bool, optional): Whether to use non-pharmacophore features for exclusion volume generation. Default is False.
    rounds (int, optional): The number of rounds for exclusion volume generation. Default is 100.
    debug (bool, optional): Whether to enable debug mode. Default is DEBUG.
    only_surface_ex (bool, optional): Whether to generate exclusion volumes only on the surface. Default is False.
    **kwargs: Additional keyword arguments.
    Returns:
    Phore: The updated pharmacophore object with generated exclusion volumes.
    """
    if remove_hs:
        ligand = Chem.RemoveHs(ligand)
    lig_coords = ligand.GetConformer().GetPositions()
    lig_phorefps, lig_norms, _, _ = generate_ligand_phore_feat(ligand, remove_hs=remove_hs)
    exclusion_volumes = np.empty((0, 3))
    if len(lig_coords) > 50:
        rounds = rounds // 2
    # print(f"lig_phorefps: {lig_phorefps}")
    # print(f"lig_norms: {lig_norms}")
    for idx, atom in enumerate(ligand.GetAtoms()):
        calculated = False
        if near_phore:
            if not check_nearby_phore(phore, lig_coords[idx], lig_phorefps[idx], cutoff=cutoff, strict=strict):
                continue
            
        for phore_idx, norm in enumerate(lig_norms[idx]):
            if lig_phorefps[idx][phore_idx] != 0:
                # ex_pos = lig_coords[idx] + norm * (low + up) / 2
                random_exs = generate_ex_and_detect_clash(lig_coords[idx], norm, phore, exclusion_volumes, lig_coords, 
                                             ex_dis=ex_dis, low=low, up=up, theta=theta, num_ex=num_ex, mode=mode)
                exclusion_volumes = np.concatenate([exclusion_volumes, random_exs], axis=0)
                calculated = True
        if not calculated and use_non_phore:
            if debug:
                print(f"Starting to generate EX for {atom.GetSymbol()}_{atom.GetIdx()}")
            neib_coords = np.array([lig_coords[neib.GetIdx()] for neib in atom.GetNeighbors()])
            curr_norm = lig_coords[atom.GetIdx()] - neib_coords.mean(axis=0)
            # ex_pos = lig_coords[idx] + curr_norm * low
            random_exs = generate_ex_and_detect_clash(lig_coords[idx], curr_norm, phore, exclusion_volumes, lig_coords, 
                                         ex_dis=ex_dis, low=low, up=up, theta=theta, num_ex=num_ex, mode=mode, rounds=rounds)
            exclusion_volumes = np.concatenate([exclusion_volumes, random_exs], axis=0)
            if debug:
                print(f"{len(random_exs)} generated for {atom.GetSymbol()}_{atom.GetIdx()}")
            
    exclude_idx = []
    if only_surface_ex:
        exclude_idx = filter_surface_ex(lig_coords, exclusion_volumes)
    exclusion_volumes = [PhoreFeature(type='EX', alpha=0.837, weight=0.5, factor=1, 
                                      coordinate=Coordinate(x=float(ex[0]), y=float(ex[1]), z=float(ex[2])),
                                      has_norm=0, norm=Coordinate(0, 0, 0), label='0', anchor_weight=1)
                         for idx, ex in enumerate(exclusion_volumes) if idx not in exclude_idx]

    phore = copy.deepcopy(phore)._replace(exclusion_volumes=exclusion_volumes+phore.exclusion_volumes)
    return phore


def check_nearby_phore(phore, atom_coord, lig_phorefp=None, cutoff=2, strict=True):
    """
    Check if there are pharmacophore features near a given atom coordinate.

    Args:
        phore (Pharmacophore): The pharmacophore object containing features.
        atom_coord (np.ndarray): The coordinates of the atom to check against.
        lig_phorefp (list, optional): A list representing the ligand pharmacophore fingerprint. Defaults to None.
        cutoff (float, optional): The distance cutoff to consider a feature as nearby. Defaults to 2.
        strict (bool, optional): If True, checks for specific feature types in the ligand pharmacophore fingerprint. Defaults to True.

    Returns:
        bool: True if there are nearby pharmacophore features, False otherwise.
    """
    matches = []
    nearby = False
    for feature in phore.features:
        coord = feature.coordinate
        coord = np.array([coord.x, coord.y, coord.z])
        if np.sqrt(np.sum((coord - atom_coord) ** 2)) < cutoff:
            if strict and isinstance(lig_phorefp, list):
                phorefp = np.array([int(x == feature.type) for x in ['MB', 'HD', 'AR', 'PO', 'HA', 'HY', 'NE', 'CV', 'CR', 'XB', 'EX']])
                match = phorefp * np.array(lig_phorefp)
                if sum(match) > 0:
                    matches.append(match)
                    nearby = True
            else:
                nearby = True
    if matches:
        lig_phorefp = [0 if i == 0 else 1 for i in sum(matches).tolist()]
        # print(f"matches: {matches}")
    return nearby


def generate_ex_and_detect_clash(at_pos, norm, phore, exclusion_volumes, lig_coords, ex_dis=0.8, 
                                 low=3.0, up=5.0, mode='radius', theta=15.0, num_ex=5, rounds=100):
    """
    Generate exclusion volumes and detect clashes with pharmacophore features.
    Args:
    at_pos (array-like): The position of the atom.
    norm (array-like): The normal vector.
    phore (object): The pharmacophore object.
    exclusion_volumes (list): List of exclusion volumes.
    lig_coords (array-like): Coordinates of the ligand.
    ex_dis (float, optional): Exclusion distance. Default is 0.8.
    low (float, optional): Lower bound for exclusion volume generation. Default is 3.0.
    up (float, optional): Upper bound for exclusion volume generation. Default is 5.0.
    mode (str, optional): Mode of exclusion volume generation. Options are 'radius', 'shell', 'aminoacid'. Default is 'radius'.
    theta (float, optional): Angle for shell mode. Default is 15.0.
    num_ex (int, optional): Number of exclusion volumes to generate. Default is 5.
    rounds (int, optional): Number of rounds for exclusion volume generation. Default is 100.
    Returns:
    list: List of generated exclusion volumes that do not clash with the pharmacophore features.
    """
    random_exs = []
    if mode == 'radius':
        center = at_pos + norm * (low + up) / 2
        radius = (up - low) / 2
        random_exs = generate_ex_by_radius(center, radius, exclusion_volumes=exclusion_volumes, 
                                           ex_dis=ex_dis, num_ex=num_ex, rounds=rounds)

    elif mode == 'shell':
        random_exs = generate_ex_by_shell(at_pos, norm, low=low, up=up, theta=theta, 
                                          num_ex=num_ex, rounds=rounds, ex_dis=ex_dis, 
                                          exclusion_volumes=exclusion_volumes)
    
    elif mode == 'aminoacid':
        random_exs = generate_ex_by_aminoacid()
    random_exs = exclude_clashed_ex(random_exs, phore, lig_coords, exclusion_volumes, ex_dis=ex_dis)
    return random_exs


def filter_surface_ex(ligand_coords, ex_coords, cutoff=30.0, cutoff_num=15, exclude_far=True, debug=False):
    """
    Filters out exclusion spheres based on their distance and angular relationship to ligand coordinates.

    Parameters:
    ligand_coords (ndarray): Array of ligand coordinates.
    ex_coords (ndarray): Array of exclusion spheres.
    cutoff (float, optional): Angular cutoff in degrees for filtering. Default is 30.0.
    cutoff_num (int, optional): Minimum number of occurrences for an extraneous coordinate to be removed. Default is 15.
    exclude_far (bool, optional): Whether to exclude coordinates that are too far from any ligand. Default is True.
    debug (bool, optional): If True, prints debug information. Default is False.

    Returns:
    list: List of indices of exclusion spheres to be removed.
    """
    dis_mat = spa.distance_matrix(ligand_coords, ex_coords)
    def stack_analysis(ex_index, lig_index, l_coords, e_coords, dis_mat, cutoff_angle=10.0):
        remove_list = []
        for idx1 in range(len(ex_index)):
            for idx2 in range(idx1+1, len(ex_index)):
                i = ex_index[idx1]
                j = ex_index[idx2]
                # if i in remove_list or j in remove_list:
                #     continue
                vec_1 = e_coords[i] - l_coords[lig_index]
                vec_2 = e_coords[j] - l_coords[lig_index]
                len_vec_1 = dis_mat[lig_index, i]
                len_vec_2 = dis_mat[lig_index, j]
                # assert sum(vec_1 ** 2)**0.5  - len_vec_1 < 1e-12
                angle = np.rad2deg(np.arccos(vec_1.dot(vec_2) / len_vec_1 / len_vec_2))
                delta_len = len_vec_2 - len_vec_1
                if  angle <= cutoff_angle and delta_len >= 1.0:
                    remove_list.append(j)
                # print(f"EX{ex_index[idx2]} --> EX{ex_index[idx1]} :: {angle}")
        return remove_list 
    sorted_index = dis_mat.argsort()
    total_list = []
    mask_d = np.sort(dis_mat, axis=1) <= 7.0
    for i in range(len(sorted_index)):
        nearby_ex = sorted_index[i][mask_d[i]]
        if len(nearby_ex) >= 2:
            # print(f"len(nearby_ex) = {nearby_ex}")
            total_list.extend(stack_analysis(nearby_ex, i, ligand_coords, ex_coords, cutoff_angle=cutoff, dis_mat=dis_mat))
    too_far = np.arange(len(ex_coords))[np.sort(dis_mat, axis=0)[0, :] > 6.0].tolist()
    # total_list.extend(too_far)
    remove_list = []
    counts = {}
    if total_list:
        counts = pd.Series(total_list).value_counts().to_dict()
        remove_list = [int(k) for k, v in counts.items() if int(v) >= cutoff_num]
    if exclude_far:
        remove_list.extend(too_far)
    remove_list = [k for k in remove_list if k not in sorted_index[:, 0]]
    remove_list = list(set(remove_list))
    if debug:
        print(f"Angle: {cutoff}, Number: {cutoff_num}, Too Far: {too_far}, Remove: {remove_list} -->", counts)
    return remove_list


def write_phore_to_file(phore, path, name=None, overwrite=False):
    """
    Writes the pharmacophore object to a file in a specific format.

    Args:
    phore (Pharmacophore): The pharmacophore object to be written to the file.
    path (str): The directory path or file path where the pharmacophore file will be saved.
    name (str, optional): The name of the pharmacophore file. If None, the pharmacophore's id will be used. Default is None.
    overwrite (bool, optional): If True, overwrite the existing file if it exists. Default is False.

    Returns:
    str: The filename of the written pharmacophore file.
    """
    name = name if name is not None else phore.id
    filename = os.path.join(path, f"{name}.phore") if os.path.isdir(path) else path
    if not os.path.exists(filename) or overwrite:
        with open(filename, 'w') as f:
            f.write(f"{name}\n")
            for feat in phore.features:
                out_string = [feat.type, feat.alpha, feat.weight, feat.factor, 
                            feat.coordinate.x, feat.coordinate.y, feat.coordinate.z, 
                            int(feat.has_norm), feat.norm.x, feat.norm.y, feat.norm.z, 
                            feat.label, feat.anchor_weight]
                f.write("\t".join([f"{x:.3f}" if isinstance(x, float) else str(x) for x in out_string ]) + '\n')

            for ex in phore.exclusion_volumes:
                out_string = [ex.type, ex.alpha, ex.weight, ex.factor, 
                              ex.coordinate.x, ex.coordinate.y, ex.coordinate.z, 
                              int(ex.has_norm), ex.norm.x, ex.norm.y, ex.norm.z, 
                              ex.label, ex.anchor_weight]
                f.write("\t".join([f"{x:.3f}" if isinstance(x, float) else str(x) for x in out_string ]) + '\n')
            f.write("$$$$\n")
    return filename


def generate_ex_by_radius(center, radius, exclusion_volumes=None, 
                          ex_dis=0.8, num_ex=5, rounds=100, debug=DEBUG):
    """
    Generate exclusion volumes by radius around a center point.
    Args:
    center (array-like): The center point around which exclusion volumes are generated.
    radius (float): The radius within which exclusion volumes are generated.
    exclusion_volumes (array-like, optional): Existing exclusion volumes to avoid overlap with.
    ex_dis (float, optional): Minimum distance between exclusion volumes. Default is 0.8.
    num_ex (int, optional): Number of exclusion volumes to generate. Default is 5.
    rounds (int, optional): Maximum number of rounds to attempt generating exclusion volumes. Default is 100.
    debug (bool, optional): If True, print debug information. Default is DEBUG.
    Returns:
    numpy.ndarray: Array of generated exclusion volumes.
    """
    random_exs = np.empty((0, 3))
    _not_max_round = True
    _not_max_num_ex = True
    n = 0
    
    st = time.time()
    while _not_max_num_ex and _not_max_round:
        curr_norm = np.random.randn(3)
        curr_ex = center + curr_norm * radius
        if len(random_exs) == 0:
            curr_ex = curr_ex.reshape(-1, 3)
        else:
            curr_ex = exclude_clashed_ex([curr_ex], exclusion_volumes=random_exs, ex_dis=ex_dis)
        if exclusion_volumes is not None:
            curr_ex = exclude_clashed_ex(curr_ex, exclusion_volumes=exclusion_volumes, ex_dis=ex_dis)
        
        random_exs = np.concatenate([random_exs, curr_ex.reshape(-1, 3)], axis=0)
        # print(f'{n} rounds sampled.')
        n += 1
        if rounds == n:
            _not_max_round = False
        if len(random_exs) == num_ex:
            _not_max_num_ex = False
    if debug and not _not_max_round and _not_max_num_ex:
        print("[W] Max round reached. Not enough exclusion spheres added.")
    if debug and not _not_max_num_ex:
        print(f"[I] {num_ex} exclusion spheres added within {n} rounds {time.time()-st}.")

    return random_exs


def generate_ex_by_shell(at_pos, norm, exclusion_volumes=None, low=3, up=5, 
                         ex_dis=0.8, theta=np.pi/12, num_ex=5, rounds=100, debug=DEBUG):
    """
    Generate exclusion volumes by randomly sampling points around a given position within a specified shell.
    Args:
    at_pos (array-like): The central position around which exclusion volumes are generated.
    norm (array-like): The normal vector used for generating perpendicular vectors.
    exclusion_volumes (array-like, optional): Existing exclusion volumes to avoid clashes with. Default is None.
    low (float, optional): The lower bound of the distance range from the central position. Default is 3.
    up (float, optional): The upper bound of the distance range from the central position. Default is 5.
    ex_dis (float, optional): The minimum distance to maintain between exclusion volumes. Default is 0.8.
    theta (float, optional): The maximum angle for random rotation in radians. Default is np.pi/12.
    num_ex (int, optional): The number of exclusion volumes to generate. Default is 5.
    rounds (int, optional): The maximum number of rounds to attempt generating exclusion volumes. Default is 100.
    debug (bool, optional): If True, print debug information. Default is DEBUG.
    Returns:
    numpy.ndarray: An array of generated exclusion volumes.
    """
    random_exs = np.empty((0, 3))

    _not_max_rounds = True
    _not_max_num_ex = True
    n = 0
    st = time.time()
    
    while _not_max_rounds or _not_max_num_ex:
        _norm = generate_perpendicular_vector(norm)
        angle = np.random.uniform(0, theta)
        rotation = spa.transform.Rotation.from_rotvec(_norm * angle)
        # rotmat = axis_angle_to_rotate_matrix(_norm, angle)
        curr_ex = rotation.apply(norm) * np.random.uniform(low, up) + at_pos
        if len(random_exs) == 0:
            curr_ex = curr_ex.reshape(-1, 3)
        else:
            curr_ex = exclude_clashed_ex([curr_ex], exclusion_volumes=random_exs, ex_dis=ex_dis)
            
        if exclusion_volumes is not None:
            curr_ex = exclude_clashed_ex(curr_ex, exclusion_volumes=exclusion_volumes, ex_dis=ex_dis)
        
        random_exs = np.concatenate([random_exs, curr_ex.reshape(-1, 3)], axis=0)
        if debug:
            print(len(random_exs), 'EX generated')

        n += 1
        if rounds == n:
            _not_max_rounds = False
        if len(random_exs) == num_ex:
            _not_max_num_ex = False

    if debug and not _not_max_rounds and _not_max_num_ex:
        print("[W] Max round reached. Not enough exclusion spheres added.")
    if debug and not _not_max_num_ex:
        print(f"[I] {num_ex} exclusion spheres added within {n} rounds {time.time()-st:.3f} seconds.")

    return random_exs


def generate_ex_by_aminoacid():
    random_exs = []
    # return random_exs
    raise NotImplementedError("Exclusion sphere generation by amino acids is currently not implemented yet.")


def exclude_clashed_ex(random_exs, phore=None, lig_coords=None, 
                       exclusion_volumes=None, low=3.0, ex_dis=0.8, debug=DEBUG):
    """
    Exclude exclusion spheres that clash with given pharmacophore features, ligand coordinates, 
    or exclusion volumes.
    Parameters:
    random_exs (list or np.ndarray): List or array of random points to be filtered.
    phore (object, optional): Pharmacophore object containing features with coordinates.
    lig_coords (list or np.ndarray, optional): List or array of ligand coordinates.
    exclusion_volumes (list or np.ndarray, optional): List or array of exclusion volume coordinates.
    low (float, optional): Distance threshold for clashing with pharmacophore or ligand coordinates. Default is 3.0.
    ex_dis (float, optional): Distance threshold for clashing with exclusion volumes. Default is 0.8.
    debug (bool, optional): If True, prints the number of points abandoned due to clashes. Default is DEBUG.
    Returns:
    np.ndarray: Array of points that do not clash with the given coordinates.
    """
    random_exs = np.array(random_exs).reshape(-1, 3)
    num_ex = len(random_exs)
    # print(f'phore: {phore}')
    if phore is not None:
        phore_coords = np.array([[feat.coordinate.x, feat.coordinate.y, feat.coordinate.z] \
                                 for feat in phore.features])
        random_exs = ex_not_clashed(random_exs, phore_coords, low, return_axis=0)
    if lig_coords is not None:
        random_exs = ex_not_clashed(random_exs, lig_coords, distance=low, return_axis=0)
    
    if exclusion_volumes is not None:
        exclusion_volumes = np.array(exclusion_volumes).reshape(-1, 3)
        random_exs = ex_not_clashed(random_exs, exclusion_volumes, distance=ex_dis, return_axis=0)
        if debug:
            print(num_ex - len(random_exs), "points abandoned.")

    return random_exs


def ex_not_clashed(points1, points2, distance, return_axis=0):
    """
    Check if points in two sets do not clash within a specified distance.

    This function computes the distance matrix between two sets of points and 
    checks if all distances are greater than the specified distance. It returns 
    the set of points that do not clash based on the specified return axis.

    Args:
        points1 (array-like): The first set of points.
        points2 (array-like): The second set of points.
        distance (float): The minimum distance to check for clashes.
        return_axis (int, optional): The axis to return the points from. 
                                     0 to return points1, 1 to return points2. 
                                     Default is 0.

    Returns:
        array-like: The set of points from the specified axis that do not clash 
                    with the points from the other set.
    """
    return [points1, points2][return_axis][np.all(spa.distance_matrix(points1, points2) > distance, axis=[1, 0][return_axis])]


def get_phore_graph(phore, graph, consider_ex=True, neighbor_cutoff=5.0, ex_connected=True, debug=False):
    """
    Constructs a pharmacophore graph from the given pharmacophore and graph.
    Args:
        phore (Pharmacophore): The pharmacophore object containing features and exclusion volumes.
        graph (HeteroData): The graph to which the pharmacophore graph will be added.
        consider_ex (bool, optional): Whether to consider exclusion volumes. Defaults to True.
        neighbor_cutoff (float, optional): The distance cutoff for considering neighbors. Defaults to 5.0.
        ex_connected (bool, optional): Whether exclusion volumes should be connected. Defaults to True.
        debug (bool, optional): If True, prints debug information. Defaults to False.
    """
    ex_start_index = 0
    if not consider_ex:
        phore = phore._replace(exclusion_volumes=[])
    ex_start_index = len(phore.features)
    neighbor_cutoff = neighbor_cutoff if neighbor_cutoff is not None else float("inf")
    phore_feats = phore.features + phore.exclusion_volumes
    num_phores = len(phore_feats)
    phore_coords = np.array([[point.coordinate.x, point.coordinate.y, point.coordinate.z]
                    for point in phore_feats])
    phore_norms = np.array([[point.norm.x - point.coordinate.x, point.norm.y - point.coordinate.y, point.norm.z - point.coordinate.z] 
                   if point.has_norm else [0, 0, 0] for point in phore_feats])
    _norm = np.linalg.norm(phore_norms, axis=1)
    _norm[_norm==0] = 1
    phore_norms = phore_norms / _norm.reshape(-1, 1)

    distances = spa.distance.cdist(phore_coords, phore_coords)
    src_list = []
    dst_list = []
    # mean_norm_list = []
    valid_distances = []
    for i in range(num_phores):
        try:
            flag = False
            dst = []
            if i < ex_start_index:
                dst = list(range(0, ex_start_index))
                dst.remove(i)
            else:
                dst = [x for x in list(np.where(distances[i, :] < neighbor_cutoff)[0])]
                dst.remove(i)
                if not ex_connected:
                    dst = [x for x in dst if x >= ex_start_index]
                
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2] # choose second because first is i itself
                # dst = list(np.argsort(distances[i, ex_start_index:-1]))[1:2] # choose second because first is i itself
            if debug:
                type_mark = '* ' if phore_feats[i].type != 'EX' else '! '
                print(f"{type_mark}dst[{i}]: {dst}")
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            # valid_dist_np = distances[i, dst]
            # valid_distances.extend(valid_dist_np.tolist())
            # sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            # weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)

        except Exception as e:
            # print('valid_dist_np:', valid_dist_np)
            print('dst:', dst)
            print('flag:', flag)
            print("phore id:", phore.id)
            print(f"phore: {phore}")
            raise(e)
        # assert 1 - 1e-2 < weights[0].sum() < 1.01
        # diff_vecs = phore_coords[src, :] - phore_coords[dst, :]  # (neigh_num, 3)
        # mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        # denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        # mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        # mean_norm_list.append(mean_vec_ratio_norm)
    # mu_r_norm = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    # graph['phore'].mu_r_norm = mu_r_norm
    graph['phore'].x = phore_featurizer(phore_feats)
    graph['phore'].pos = torch.from_numpy(phore_coords).float()
    graph['phore'].norm = torch.from_numpy(phore_norms).float()
    graph['phore', 'phore_contact', 'phore'].edge_index = torch.from_numpy(np.asarray([src_list, dst_list]))
    graph['phore', 'phore_contact', 'phore'].edge_attr = torch.from_numpy(np.asarray(valid_distances).reshape(len(src_list), -1)).float()

    return


def phore_featurizer(phore):
    """
    Converts a list of pharmacophore features into a tensor representation.

    Args:
        phore (list): A list of pharmacophore features. Each feature is expected to have the attributes:
            - type (str): The type of the pharmacophore feature.
            - has_norm (bool): Indicates if the feature has a norm.
            - alpha (float): The alpha value of the feature.
            - weight (float): The weight of the feature.

    Returns:
        torch.Tensor: A tensor representation of the pharmacophore features, where each feature is represented
        as a list containing:
            - Index of the feature type in the allowable features list.
            - Index indicating if the feature is an exclusion volume.
            - Index indicating if the feature has a norm.
            - The alpha value of the feature.
            - The weight of the feature.
    """
    phore_feat_list = []
    for idx, phore_feat in enumerate(phore):
        phore_feat_list.append([
            safe_index(allowable_features_phore['possible_phore_type_list'], phore_feat.type),
            safe_index(allowable_features_phore['possible_is_exlusion_volume'], phore_feat.type == 'EX'),
            safe_index(allowable_features_phore['possible_has_norm'], phore_feat.has_norm),
            phore_feat.alpha, 
            phore_feat.weight,
            # phore_feat.factor,
            # phore_feat.anchor_weight
        ])
    return torch.tensor(phore_feat_list)


def parse_phore_line(record, skip_wrong_lines=False, cvs=False):
    """
    Parses a line of pharmacophore data and returns a PhoreFeature object.

    Args:
        record (str): A tab-separated string representing a pharmacophore feature.
        skip_wrong_lines (bool, optional): If True, skips lines that cannot be parsed. Defaults to False.
        cvs (bool, optional): If True, uses the full pharmacophore type string. If False, uses the first two characters. Defaults to False.

    Returns:
        PhoreFeature: An object representing the parsed pharmacophore feature.
        None: If the record is "$$$$".
        False: If the line cannot be parsed and skip_wrong_lines is True.

    Raises:
        SyntaxError: If the line cannot be parsed and skip_wrong_lines is False.

    Note:
        The record string is expected to have the following tab-separated fields:
        phore_type, alpha, weight, factor, x, y, z, has_norm, norm_x, norm_y, norm_z, label, anchor_weight.
    """
    if record == "$$$$":
        return None
    else:
        try:
            phore_type, alpha, weight, factor, x, y, z, \
                has_norm, norm_x, norm_y, norm_z, label, anchor_weight = record.split("\t")
            phore_type = phore_type if cvs else phore_type[:2]
            coordinate = Coordinate(float(x), float(y), float(z))
            norm = Coordinate(float(norm_x), float(norm_y), float(norm_z))
            has_norm = bool(int(has_norm))
            alpha, weight, factor, anchor_weight = float(alpha), float(weight), float(factor), float(anchor_weight)
            return PhoreFeature(phore_type, alpha, weight, factor, coordinate, has_norm, norm, label, anchor_weight)
        except:
            print(f"[E]: Failed to parse the line:\n {record}")
            if not skip_wrong_lines:
                raise SyntaxError("Invalid phore feature syntax from the specified phore file.")
            else:
                return False


def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def test_phore():
    from torch_geometric.data import Dataset, HeteroData
    phore_file = '../../test/2x8z_complex.phore'
    phores = parse_phore(phore_file=phore_file)
    # if len(phores):
    #     print("Phore ID:\n", phores[0].id)
    #     print("Phore Features:\n", phores[0].features)
    #     print("Phore Features:\n", phores[0].exclusion_volumes)
    phore = phores[0]
    graph = HeteroData()
    get_phore_graph(phore, graph, neighbor_cutoff=8)
    print(graph)


def axis_angle_to_rotate_matrix(axis, radian):
    """
    Converts an axis-angle representation to a rotation matrix.

    Args:
    axis (numpy.ndarray): A 3-element array representing the axis of rotation.
    radian (float): The angle of rotation in radians.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    return linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))


def generate_perpendicular_vector(v, norm=True, epsilon=1e-12):
    """
    Generate a vector that is perpendicular to the given vector `v`.

    Args:
    v (array-like): A 3-dimensional vector to which the generated vector will be perpendicular.
    norm (bool, optional): If True, the generated vector will be normalized. Default is True.
    epsilon (float, optional): A small value to avoid division by zero during normalization. Default is 1e-12.

    Returns:
    numpy.ndarray: A 3-dimensional vector that is perpendicular to `v`.
    """
    a, b = np.random.uniform(0.1, 1,size=(2))
    if v[2] != 0:
        c = - (a * v[0] + b * v[1]) / v[2] 
    else:
        assert not (v[0] == 0 and v[1] == 0)
        a = -v[1]
        b = v[0]
        c = 0
    vec = np.array([a, b, c])
    if norm:
        vec = vec / (np.linalg.norm(vec, axis=-1) + epsilon)
    return vec


def generate_complex_phore(ligand_file, protein_file, pdb_id, tmp_dir='../../data/ChEMBL/', ancphore_path=ANCPHORE):
    """
    Generates a pharmacophore file using the AncPhore program from complex structure.
    Args:
        ligand_file (str): Path to the ligand file.
        protein_file (str): Path to the protein file.
        pdb_id (str): PDB identifier for the complex.
        tmp_dir (str, optional): Temporary directory to store output files. Defaults to '../../data/ChEMBL/'.
        ancphore_path (str, optional): Path to the AncPhore executable. Defaults to ANCPHORE.
    Returns:
        str: Content of the generated complex pharmacophore file.
    Raises:
        AssertionError: If the AncPhore program is not found at the specified path.
        Exception: If there is an error during the execution of the AncPhore command.
    """
    assert os.path.exists(ancphore_path), "[E] AncPhore Program Not Found."
    
    out_file = os.path.join(tmp_dir, f"complex_phores/{pdb_id}_complex.phore")
    log_file = os.path.join(tmp_dir, f"logs/{pdb_id}_complex.log")
    command = f"{ancphore_path} -l {ligand_file} -p {protein_file} --refphore {out_file} > {log_file} 2>&1"
    if not os.path.exists(out_file):
        try:
            os.system(command)
        except Exception as e:
            print(e)
    content = ""
    if os.path.exists(out_file):
        content = "".join(open(out_file, 'r').readlines())
    return content


def parse_score_file(score_file, return_all=False, fitness=1):
    """
    Parses a score file and extracts specific fitness scores or all relevant data.

    Args:
        score_file (str): Path to the score file to be parsed.
        return_all (bool, optional): If True, returns all relevant data columns. Defaults to False.
        fitness (int, optional): Specifies which fitness score to extract. Defaults to 1.
        - 1 : Database ID
        - 2 : Mol Energy
        - 3 : Reference ID
        - 4 : Number of database pharmacophore features (N_db)
        - 5 : Volume of database pharmacophore (V_db)
        - 6 : Volume of reference pharmacophore (V_ref)
        - 7 : Volume of overlap between reference and database pharmacophore models (V_overlap)
        - 8 : Percentage of matched pairs (n/N, d <= r)
        - 9 : Volume of overlap between reference exclusion spheres and ligand atoms (V_exOverlap)
        - 10: The percentage of the overlap anchor feature volumes out of total anchor features (V_overlapAnchor/V_anchor)
        - 11: The percentage of the overlap feature volumes out of total features (V_overlap/V_ref)
        - 12: The percentage of the overlap exclusion volumes (max(V_exOverlap/epsilon, 1))
        - 13: Fitness score with customized weights (overlap_coeff, percent_coeff, anchor_coeff)
        - 14: PhScore 1 (overlap_coeff=1.0, percent_coeff=0.0, anchor_coeff=0.0)
        - 15: PhScore 2 (overlap_coeff=0.5, percent_coeff=0.5, anchor_coeff=0.0)
        - 16: PhScore 3 (overlap_coeff=0.5, percent_coeff=0.0, anchor_coeff=0.5)
        - 17: PhScore 4 (overlap_coeff=0.3333, percent_coeff=0.3333, anchor_coeff=0.3333)

    Returns:
        list: A list of floats representing the extracted fitness scores or relevant data columns.
              Returns None if an error occurs during parsing.

    Raises:
        Exception: If there is an error in parsing the score file, an exception is caught and an error message is printed.
    """
    index = {1:-4, 2:-3, 3: -2, 4:-1, 5:-5, 6:-6}
    try:
        if not return_all:
            return [float(line.strip().split("\t")[index[fitness]]) for line in open(score_file, 'r').readlines()]
        else:
            return [[float(x) for x in line.strip().split("\t")[-6:-1]] for line in open(score_file, 'r').readlines()]
        # return [float(line.strip().split("\t")[-2]) / 1.2 for line in open(score_file, 'r').readlines()]
    except Exception as e:
        print(f"[E] Failed to parse the score file {score_file}.", e)
        return None


def calc_phore_fitting(ligand_file, phore_file, score_file, dbphore_file, log_file, 
                       overwrite=False, return_all=False, exVolume_cutoff=500,
                       overlap_coeff=-1, percent_coeff=-1, anchor_coeff=-1,
                       ancphore_path=ANCPHORE, 
                       target_fishing=False, fitness=1
                       ):
    """
    Calculate the pharmacophore fitting score for a given ligand.
    Args:
    ligand_file (str): Path to the ligand file.
    phore_file (str): Path to the reference pharmacophore file.
    score_file (str): Path to the output score file.
    dbphore_file (str): Path to the database pharmacophore file.
    log_file (str): Path to the log file.
    overwrite (bool): Whether to overwrite the existing score file. Default is False.
    return_all (bool): Whether to return all scores. Default is False.
    exVolume_cutoff (int): Excluded volume cutoff value. Default is 500.
    overlap_coeff (float): Overlap coefficient. Default is -1 (not used).
    percent_coeff (float): Percent coefficient. Default is -1 (not used).
    anchor_coeff (float): Anchor coefficient. Default is -1 (not used).
    ancphore_path (str): Path to the AncPhore program.
    target_fishing (bool): Whether to use target fishing mode. Default is False.
    fitness (int): Fitness value. Default is 1.
    Returns:
    scores (list or None): Parsed scores from the score file, or None if the score file is not generated.
    """
    ligand_file = os.path.abspath(ligand_file)
    phore_file   = os.path.abspath(phore_file)
    score_file   = os.path.abspath(score_file)
    dbphore_file = os.path.abspath(dbphore_file)
    log_file     = os.path.abspath(log_file)
    
    name = os.path.basename(ligand_file).split(".")[0]
    flag = False
    if not os.path.exists(ligand_file):
        flag = True
        print(f"[E] Failed to calculate the fitting score of ligand `{name}`.\nThe ligand file `{ligand_file}` doesn't exist.")
    if not os.path.exists(phore_file):
        flag = True
        print(f"[E] Failed to calculate the fitting score of ligand `{name}`.\nThe ligand file `{phore_file}` doesn't exist.")
    if not os.path.exists(ancphore_path):
        flag = True
        print(f"[E] Invalid path to AncPhore program: `{ancphore_path}`")
    scores = None
    fitness = 5 if target_fishing else fitness
    if not flag and (not os.path.exists(score_file) or overwrite):
        ancphore_path = os.path.dirname(ancphore_path)
        cutoff_flag = "" if exVolume_cutoff == 500 else f"--exvolume_cutoff {exVolume_cutoff}"
        coeff_flag = ""
        if overlap_coeff != -1:
            coeff_flag += f"--overlap_coeff {overlap_coeff} "
        if percent_coeff != -1:
            coeff_flag += f"--percent_coeff {percent_coeff} "
        if anchor_coeff != -1:
            coeff_flag += f"--anchor_coeff {anchor_coeff} "

        command = f"cd {ancphore_path} && timeout 200s ./AncPhore -d {ligand_file} --refphore {phore_file} --scores {score_file} usedMultiConformerFile formodel {cutoff_flag} {coeff_flag}> {log_file} 2>&1 && cd - > /dev/null"
        # command = f"cd {ancphore_path} && timeout 200s ./AncPhore -d {ligand_file} --refphore {phore_file} --dbphore {dbphore_file} --scores {score_file} usedMultiConformerFile formodel {cutoff_flag} {coeff_flag}> {log_file} 2>&1 && cd - > /dev/null"
        try:
            status_code = os.system(command)
            # print(f'Command: {command}')
        except Exception as e:
            print(f"[E] Failed to calculate the fitting score of ligand `{name}`.", e)
        
    if os.path.exists(score_file):
        scores = parse_score_file(score_file, return_all=return_all, fitness=fitness)
        # print(f"{name}: {scores}")
    else:
        print(f'[E] No score file generated for {name} and {os.path.basename(phore_file)}')

    return scores


if __name__ == '__main__':
    pass

