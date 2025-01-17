"""
Diffusion-related operations taken from DiffDock
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates, keep_update=False):
    """
    Modify the conformer of a ligand based on translation, rotation, and torsion updates.
    Args:
        data (torch_geometric.data.HeteroData): The ligand data should include:
            - 'ligand'.pos (torch.Tensor): Positions of the ligand atoms.
            - 'ligand'.norm (torch.Tensor, optional): Normals of the ligand atoms.
            - 'ligand'.x (torch.Tensor): Features of the ligand atoms.
            - 'ligand'.edge_index (torch.Tensor): Edge indices of the ligand graph.
            - 'ligand'.edge_mask (torch.Tensor): Mask for edges to be considered for torsion updates.
            - 'ligand'.mask_rotate (np.ndarray or torch.Tensor): Mask for atoms to be rotated.
        tr_update (torch.Tensor): Translation update vector.
        rot_update (torch.Tensor): Rotation update vector in axis-angle representation.
        torsion_updates (torch.Tensor or None): Torsion angle updates.
        keep_update (bool, optional): Whether to keep track of the updates. Default is False.
    Returns:
        Updated data with modified ligand positions and normals.
    """
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    lig_norm = data['ligand'].norm.reshape(-1, data['ligand'].x.shape[0], 3) + data['ligand'].pos.unsqueeze(0) \
        if hasattr(data['ligand'], 'norm') else None

    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    # print(f"rot_update.dtype: {rot_update.dtype}")
    # print(f"data['ligand'].pos: {data['ligand'].pos.dtype}")
    # print(f"lig_center.dtype: {lig_center.dtype}")
    # print(f"tr_update.dtype: {tr_update.dtype}")
    # print(f"rot_mat.dtype: {rot_mat.dtype}")
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center
    rigid_new_norm = (lig_norm - lig_center) @ rot_mat.T + tr_update + lig_center if lig_norm is not None else None


    if torsion_updates is not None:
        flexible_new_pos, flexible_new_norm = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) \
                                                               else data['ligand'].mask_rotate[0],
                                                           torsion_updates,
                                                           norm=rigid_new_norm)

        flexible_new_pos = flexible_new_pos.to(rigid_new_pos.device)
        flexible_new_norm = flexible_new_norm.to(rigid_new_pos.device) if flexible_new_norm is not None else None
        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        aligned_flexible_norm = flexible_new_norm @ R.T + t.T - aligned_flexible_pos if flexible_new_norm is not None else None

        data['ligand'].pos = aligned_flexible_pos
        if aligned_flexible_norm is not None:
            data['ligand'].norm = aligned_flexible_norm.reshape(data['ligand'].x.shape[0], -1)
    else:
        data['ligand'].pos = rigid_new_pos
        data['ligand'].norm = (rigid_new_norm - rigid_new_pos).reshape(data['ligand'].x.shape[0], -1) \
            if rigid_new_norm is not None else None
    
    if keep_update:
        data.rvs_rot_update = data.rvs_rot_update + [rot_update.numpy()] if hasattr(data, 'rvs_rot_update') else [rot_update.numpy()]
        data.rvs_tr_update = data.rvs_tr_update + [tr_update.numpy()] if hasattr(data, 'rvs_tr_update') else [tr_update.numpy()]
        tor = torsion_updates if torsion_updates is not None else None
        data.rvs_tor_update = data.rvs_tor_update + [tor] if hasattr(data, 'rvs_tor_update') else [tor]
        data.docked_poses = data.docked_poses + [data['ligand'].pos.cpu().numpy()] if hasattr(data, 'docked_poses') \
            else [data['ligand'].pos.cpu().numpy()]

    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    """
    Returns a function that generates timestep embeddings based on the specified type.

    Args:
    embedding_type (str): The type of embedding to use. Supported types are 'sinusoidal' and 'fourier'.
    embedding_dim (int): The dimensionality of the embedding.
    embedding_scale (int, optional): The scale factor for the embedding. Default is 10000.

    Returns:
    function: A function that takes a timestep and returns the corresponding embedding.

    Raises:
    NotImplementedError: If the specified embedding_type is not supported.
    """
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    """
    Generates a time schedule for diffusion processes.

    Args:
    inference_steps (int): The number of inference steps.

    Returns:
    numpy.ndarray: A linearly spaced array of time steps from 1 to 0, excluding the endpoint.
    """
    return np.linspace(1, 0, inference_steps + 1)[:-1]


def set_time(complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms, device):
    """
    Sets the time attributes for the nodes in the ligand, receptor, and optionally atom graphs within the complex_graphs dictionary.

    Args:
    complex_graphs (torch_geometric.data.HeteroData): HeteroData graphs for 'ligand', 'receptor', and optionally 'atom'.
    t_tr (float): The translation time to be set for the nodes.
    t_rot (float): The rotation time to be set for the nodes.
    t_tor (float): The torsion time to be set for the nodes.
    batchsize (int): The batch size for the complex time attributes.
    all_atoms (bool): A flag indicating whether to set the time attributes for the 'atom' graph.
    device (torch.device): The device to which the tensors should be moved.

    Returns:
    None
    """
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                               'rot': t_rot * torch.ones(batchsize).to(device),
                               'tor': t_tor * torch.ones(batchsize).to(device)}
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)}
        
def set_time_phore(graphs, t_tr, t_rot, t_tor, batchsize, device):
    """
    Sets the time attributes for the nodes in the 'ligand' and 'phore' graphs, 
    as well as the complex time attributes for the entire batch.

    Args:
    graphs (dict): A dictionary containing 'ligand' and 'phore' graphs.
    t_tr (float): The translation time value to be set for the nodes.
    t_rot (float): The rotation time value to be set for the nodes.
    t_tor (float): The torsion time value to be set for the nodes.
    batchsize (int): The size of the batch.
    device (torch.device): The device (CPU or GPU) to which the tensors will be moved.

    Returns:
    None
    """
    graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(graphs['ligand'].num_nodes).to(device)}
    graphs['phore'].node_t = {
        'tr': t_tr * torch.ones(graphs['phore'].num_nodes).to(device),
        'rot': t_rot * torch.ones(graphs['phore'].num_nodes).to(device),
        'tor': t_tor * torch.ones(graphs['phore'].num_nodes).to(device)}
    graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                        'rot': t_rot * torch.ones(batchsize).to(device),
                        'tor': t_tor * torch.ones(batchsize).to(device)}