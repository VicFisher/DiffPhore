import torch, copy
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

"""
    Preprocessing and computation for torsional updates to conformers
"""


def get_transformation_mask(pyg_data):
    """
    Generates a transformation mask for the given PyG data.

    This function creates a mask that identifies which edges in the graph 
    should be rotated based on the connectivity of the graph after removing 
    each edge. It uses NetworkX to analyze the connectivity of the graph.

    Args:
    -----------
    pyg_data (torch_geometric.data.Data): The input data containing the graph information in PyG format.

    Returns:
    --------
    mask_edges (numpy.ndarray): A boolean array indicating which edges should be considered for rotation.
    mask_rotate (numpy.ndarray): A boolean matrix where each row corresponds to an edge and each column 
        corresponds to a node. The value is True if the node is part of the 
        subgraph that should be rotated when the corresponding edge is removed.
    """
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False, norm=None):
    """
    Modify the torsion angles of a conformer based on the provided updates.

    Args:
    pos (torch.Tensor or np.ndarray): The positions of the atoms in the conformer.
    edge_index (torch.Tensor): The edge indices representing the bonds between atoms.
    mask_rotate (torch.Tensor): A mask indicating which parts of the conformer should be rotated.
    torsion_updates (torch.Tensor): The updates to the torsion angles for each edge.
    as_numpy (bool, optional): If True, return the positions and normals as numpy arrays. Default is False.
    norm (torch.Tensor or np.ndarray, optional): The normals of the atoms in the conformer. Default is None.

    Returns:
    tuple: A tuple containing the updated positions and normals (if provided) of the conformer.
    """
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    if norm is not None:
        if type(norm) != np.ndarray: norm = norm.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        # print(f"rot_vec.type: {type(rot_vec)}")
        # print(f"rot_vec.shape: {rot_vec.shape}")
        # print(f"torsion_updates.type: {type(torsion_updates)}")
        # print(f"torsion_updates.shape: {torsion_updates.shape}")

        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
        if norm is not None:
            norm[:, mask_rotate[idx_edge]] = (norm[:, mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
            # norm[:, mask_rotate[idx_edge]] = (norm[:, mask_rotate[idx_edge]] - norm[:, None, v]) @ rot_mat.T + norm[:, None, v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    if not as_numpy: norm = torch.from_numpy(norm.astype(np.float32)) if norm is not None else None
    return pos, norm


def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    """
    Perturbs the torsion angles of a batch of molecular conformers.

    Args:
    data (Data or object with attributes `pos`, `edge_index`, `edge_mask`, and `mask_rotate`): 
        The input data containing molecular conformers and their associated information.
    torsion_updates (array-like): 
        The updates to be applied to the torsion angles.
    split (bool, optional): 
        If True, returns a list of perturbed positions for each conformer. 
        If False, returns a single array with all perturbed positions. Default is False.
    return_updates (bool, optional): 
        If True, returns the list of torsion updates applied to each conformer. Default is False.

    Returns:
    pos_new (array-like or list of array-like): 
        The perturbed positions of the molecular conformers. The format depends on the `split` parameter.
    torsion_update_list (list of array-like, optional): 
        The list of torsion updates applied to each conformer. Only returned if `return_updates` is True.
    """
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new
