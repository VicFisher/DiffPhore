from argparse import Namespace
from functools import partial
import math
import os

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter, scatter_mean, scatter_add, scatter_max
import numpy as np
from e3nn.nn import BatchNorm
import yaml

from utils import so3, torus
from datasets.process_mols import lig_feature_dims
from datasets.process_pharmacophore import phore_feature_dims
from models.e3phore import OuterProductModule, Trioformer


class AtomEncoder(torch.nn.Module):
    """
    AtomEncoder is a neural network module designed to encode atomic features into embeddings.

    Args:
        emb_dim (int): The dimension of the embedding space.
        feature_dims (tuple): A tuple where the first element is a list containing the number of categories for each categorical feature, 
                              and the second element is the number of scalar features.
        sigma_embed_dim (int): The dimension of the sigma embedding.

    Attributes:
        atom_embedding_list (torch.nn.ModuleList): A list of embedding layers for each categorical feature.
        num_categorical_features (int): The number of categorical features.
        num_scalar_features (int): The number of scalar features plus the sigma embedding dimension.
        linear (torch.nn.Linear): A linear layer to process scalar features if they exist.

    Methods:
        forward(x):
            Encodes the input features into an embedding.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            Returns:
                torch.Tensor: The resulting embedding of shape (batch_size, emb_dim).
    """

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim):
        # first element of feature_dims tuple is a list with the length of each categorical feature 
        # and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)


    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features, f"Invalid input shape: {x.shape[1]} != {self.num_categorical_features} + {self.num_scalar_features}"
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])

        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    """
    A tensor product convolutional layer for processing geometric data.
    Args:
        in_irreps (o3.Irreps): Input irreducible representations.
        sh_irreps (o3.Irreps): Spherical harmonics irreducible representations.
        out_irreps (o3.Irreps): Output irreducible representations.
        n_edge_features (int): Number of edge features.
        residual (bool, optional): If True, use residual connections. Default is True.
        batch_norm (bool, optional): If True, apply batch normalization. Default is True.
        dropout (float, optional): Dropout rate. Default is 0.0.
        hidden_features (int, optional): Number of hidden features in the fully connected layer. If None, defaults to n_edge_features.
    Attributes:
        in_irreps (o3.Irreps): Input irreducible representations.
        out_irreps (o3.Irreps): Output irreducible representations.
        sh_irreps (o3.Irreps): Spherical harmonics irreducible representations.
        residual (bool): If True, use residual connections.
        tp (o3.FullyConnectedTensorProduct): Fully connected tensor product layer.
        fc (torch.nn.Sequential): Fully connected layer.
        batch_norm (BatchNorm or None): Batch normalization layer if batch_norm is True, else None.
    Methods:
        forward(node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
            Forward pass of the tensor product convolutional layer.
            Args:
                node_attr (torch.Tensor): Node attributes.
                edge_index (torch.Tensor): Edge indices.
                edge_attr (torch.Tensor): Edge attributes.
                edge_sh (torch.Tensor): Spherical harmonics of edges.
                out_nodes (int, optional): Number of output nodes. If None, defaults to the number of input nodes.
                reduce (str, optional): Reduction method for scatter operation. Default is 'mean'.
            Returns:
                torch.Tensor: Output node attributes after applying the tensor product convolution.
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features
        # print('in_irreps', in_irreps)
        # print('sh_irreps', sh_irreps)
        # print('out_irreps', out_irreps)
        

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        # print("tp.weight_numel =", tp.weight_numel)
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        # print('tp.shape =', tp.shape)
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class TensorProductScoreModel(torch.nn.Module):
    """
    The major workflow of DiffPhore.

    Methods:
        forward(data):
            Forward pass of the model.
        get_trtheta_score(data, lig_node_attr):
            Computes translational and rotational score vectors.
        build_center_conv_graph(data):
            Builds the filter and edges for the convolution generating translational and rotational scores.
        build_bond_conv_graph(data):
            Builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes.
    """
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5.0, phore_max_radius=5.0, cross_max_distance=25.0, consider_norm=False,
                 center_max_distance=30.0, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, confidence_mode=False,
                 confidence_dropout=0.0, confidence_no_batchnorm=False, num_confidence_outputs=1,
                 num_phoretype=11, auto_phorefp=True, use_phore_rule=True, cross_distance_transition=False,
                 phore_direction_transition=False, phoretype_match_transition=False,
                 angle_match=True, new=True, ex_factor=-2.0, phoretype_match=True, 
                 boarder=False, clash_tolerance=0.4, clash_cutoff=[1, 2, 3, 4, 5], by_radius=False,
                 use_phore_match_feat=False, use_att=False, trioformer_layer=1, update_by_att=False,
                 contrastive_model=None, contrastive_node=False, atom_weight='softmax', 
                 dist_for_fitscore=False, angle_for_fitscore=False, type_for_fitscore=False, 
                 norm_by_ph=False, sigmoid_for_fitscore=False, readout='mean', as_exp=False, scaler=1.0, 
                 multiple=False, **kwargs):
        super(TensorProductScoreModel, self).__init__()

        self.encoder = LigPhoreEncoder(
            t_to_sigma=t_to_sigma, device=device, timestep_emb_func=timestep_emb_func, in_lig_edge_features=in_lig_edge_features, 
            sigma_embed_dim=sigma_embed_dim, sh_lmax=sh_lmax, ns=ns, nv=nv, num_conv_layers=num_conv_layers, lig_max_radius=lig_max_radius, 
            phore_max_radius=phore_max_radius, cross_max_distance=cross_max_distance, consider_norm=consider_norm,
            center_max_distance=center_max_distance, distance_embed_dim=distance_embed_dim, cross_distance_embed_dim=cross_distance_embed_dim, 
            no_torsion=no_torsion, scale_by_sigma=scale_by_sigma, use_second_order_repr=use_second_order_repr, batch_norm=batch_norm,
            dropout=dropout, confidence_mode=confidence_mode, confidence_dropout=confidence_dropout, 
            confidence_no_batchnorm=confidence_no_batchnorm, num_confidence_outputs=num_confidence_outputs,
            num_phoretype=num_phoretype, auto_phorefp=auto_phorefp, use_phore_rule=use_phore_rule, cross_distance_transition=cross_distance_transition,
            phore_direction_transition=phore_direction_transition, phoretype_match_transition=phoretype_match_transition,
            angle_match=angle_match, new=new, ex_factor=ex_factor, phoretype_match=phoretype_match, 
            boarder=boarder, clash_tolerance=clash_tolerance, clash_cutoff=clash_cutoff, by_radius=by_radius,
            use_phore_match_feat=use_phore_match_feat, use_att=use_att, trioformer_layer=trioformer_layer, update_by_att=update_by_att,
            contrastive_model=contrastive_model, contrastive_node=contrastive_node, atom_weight=atom_weight, 
            angle_for_fitscore=angle_for_fitscore, type_for_fitscore=type_for_fitscore, as_exp=as_exp, 
            scaler=scaler, multiple=multiple, **kwargs
        )
        
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.phore_max_radius = phore_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers
        self.consider_norm = consider_norm
        self.num_phoretype = num_phoretype
        # self.phore_weight = torch.tensor(phore_weight).float().to(self.device)
        self.auto_phorefp = auto_phorefp
        self.use_phore_rule = use_phore_rule
        self.angle_match = angle_match
        self.new = new
        self.ex_factor = ex_factor
        self.phoretype_match = phoretype_match
        self.boarder = boarder
        self.clash_tolerance = clash_tolerance
        self.clash_cutoff = clash_cutoff if isinstance(clash_cutoff, list) else [clash_cutoff]
        self.by_radius = by_radius
        self.use_phore_match_feat = use_phore_match_feat
        self.use_att = use_att
        self.trioformer_layer = trioformer_layer
        self.atom_weight = atom_weight
        self.norm_by_ph = norm_by_ph
        self.dist_for_fitscore = dist_for_fitscore
        self.angle_for_fitscore = angle_for_fitscore
        self.type_for_fitscore = type_for_fitscore
        self.sigmoid_for_fitscore = sigmoid_for_fitscore
        self.readout = readout
        self.as_exp = as_exp
        self.scaler = scaler
        self.multiple = multiple


        # center of mass translation and rotation components
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns)
        )

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.encoder.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
        self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

        if not no_torsion:
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=self.encoder.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f'{ns}x0o + {ns}x0e',
                n_edge_features=3 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )


    def forward(self, data):
        """
        Perform a forward pass through the model.

        Args:
            data (HeteroDataBatch): Input data containing information about the ligand and pharmacophore.

        Returns:
            tuple: A tuple containing the predicted translation score (tr_pred), 
                   rotation score (rot_pred), and torsion score (tor_pred).
        """
        ## Ligand-Pharmacophore Interaction Encoder
        lig_node_attr, phore_node_attr = self.encoder(data)

        ## Estimate the ligand's scores
        tr_pred, rot_pred, tor_pred = self.get_trtheta_score(data, lig_node_attr)
        return tr_pred, rot_pred, tor_pred


    def get_trtheta_score(self, data, lig_node_attr):
        """
        Compute the translational, rotational, and torsional score vectors for a given molecular complex.

        Args:
            data (Data): Input data containing molecule and pharmacophore information.
            lig_node_attr (Tensor): Node attributes of the ligand.

        Returns:
            tuple: A tuple containing:
                - tr_pred (Tensor): Predicted translational score vector.
                - rot_pred (Tensor): Predicted rotational score vector.
                - tor_pred (Tensor): Predicted torsional score vector (empty if no torsion or no edges).

        Notes:
            - The function first computes the translational and rotational score vectors.
            - It then normalizes these vectors and scales them by sigma if required.
            - If torsion is not considered or there are no edges, an empty tensor is returned for torsional score.
            - Otherwise, the torsional score vector is computed and scaled by sigma if required.
        """
        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: 
            tor_pred = torch.empty(0)
            if tr_pred.device.type == 'cuda':
                tor_pred.to(tr_pred.device)
            return tr_pred, rot_pred, tor_pred

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred


    def build_center_conv_graph(self, data):
        """
        Builds the filter and edges for the convolution generating translational and rotational scores.

        Args:
            data (HeteroDataBatch): A batch of data containing the ligand and pharmacophore data. It should have the following keys:
                - 'ligand': An object with the following attributes:
                    - batch (Tensor): A tensor containing the batch indices of the ligand nodes.
                    - x (Tensor): A tensor containing the features of the ligand nodes.
                    - pos (Tensor): A tensor containing the positions of the ligand nodes.
                    - node_sigma_emb (Tensor): A tensor containing the sigma embeddings of the ligand nodes.

        Returns:
            tuple: A tuple containing the following elements:
                - edge_index (Tensor): A tensor containing the edge indices.
                - edge_attr (Tensor): A tensor containing the edge attributes.
                - edge_sh (Tensor): A tensor containing the spherical harmonics of the edge vectors.
        """
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, 
                                         normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh


    def build_bond_conv_graph(self, data):
        """
        Builds the graph for the convolution between the center of the rotatable bonds and the neighboring nodes.

        Args:
            data (Data): A data object containing the ligand information.

        Returns:
            tuple: A tuple containing the following elements:
                - bonds (Tensor): The indices of the bonds in the ligand.
                - edge_index (Tensor): The edge indices for the graph convolution.
                - edge_attr (Tensor): The edge attributes after encoding and embedding.
                - edge_sh (Tensor): The spherical harmonics of the edge vectors.
        """

        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, 
                            batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.encoder.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, 
                                         normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh


class LigPhoreEncoder(torch.nn.Module):
    """
    LigPhoreEncoder is a neural network module designed for encoding ligand and pharmacophore graphs 
    and performing message passing between them. It supports various configurations and features 
    such as attention mechanisms, contrastive learning, and different types of embeddings.
    Methods:
        forward(data):
            Forward pass of the model.
        build_lig_conv_graph(data):
            Builds the ligand graph edges and initial node and edge features.
        build_phore_conv_graph(data):
            Builds the receptor initial node and edge embeddings.
        _build_phoretype_cross_conv_graph(data, z_ij=None, **kwargs):
            Builds the cross pharmacophore type and normal direction alignment between ligand and pharmacophores.
        boarder_analyze(data):
            Analyzes the border for exclusion volumes.
        get_geometric_attention(data, lig_node_feat, phore_node_feat):
            Computes geometric attention between ligand and pharmacophore nodes.
    """
    def __init__(self, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5.0, phore_max_radius=5.0, cross_max_distance=25.0, consider_norm=False,
                 center_max_distance=30.0, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True, 
                 dropout=0.0,  num_phoretype=11, auto_phorefp=True, use_phore_rule=True, cross_distance_transition=False,
                 phore_direction_transition=False, phoretype_match_transition=False,
                 angle_match=True, new=True, ex_factor=-2.0, phoretype_match=True, 
                 boarder=False, clash_tolerance=0.4, clash_cutoff=[1, 2, 3, 4, 5], by_radius=False,
                 use_phore_match_feat=False, use_att=False, trioformer_layer=1, update_by_att=False,
                 contrastive_model=None, contrastive_node=False, atom_weight='softmax', confidence_mode=False, 
                 angle_for_fitscore=False, type_for_fitscore=False, scaler=1.0, multiple=False, **kwargs):
        super(LigPhoreEncoder, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.phore_max_radius = phore_max_radius
        self.cross_max_distance = cross_max_distance
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers
        self.consider_norm = consider_norm
        self.num_phoretype = num_phoretype
        self.auto_phorefp = auto_phorefp
        self.use_phore_rule = use_phore_rule
        self.angle_match = angle_match
        self.new = new
        self.ex_factor = ex_factor
        self.phoretype_match = phoretype_match
        self.boarder = boarder
        self.clash_tolerance = clash_tolerance
        self.clash_cutoff = clash_cutoff if isinstance(clash_cutoff, list) else [clash_cutoff]
        self.by_radius = by_radius
        self.use_phore_match_feat = use_phore_match_feat
        self.use_att = use_att
        self.trioformer_layer = trioformer_layer
        self.contrastive = {}
        self.contrastive['model'] = contrastive_model
        self.contrastive['ns'] = contrastive_model.ns if contrastive_model is not None else 0
        if contrastive_model is not None and contrastive_model.num_conv_layers >=3:
            self.contrastive['ns'] *= 2
        self.atom_weight = atom_weight
        self.contrastive_node = contrastive_node
        self.angle_for_fitscore = angle_for_fitscore
        self.type_for_fitscore = type_for_fitscore
        self.scaler = scaler
        self.multiple = multiple

        if self.contrastive['model'] is not None:
            self.lig_con_embeding = nn.Linear(self.contrastive['ns'], ns)
            self.phore_con_embeding = nn.Linear(self.contrastive['ns'], ns)
        else:
            self.lig_con_embeding = None
            self.phore_con_embeding = None
        # print(f'self.by_radius = {self.by_radius}')
        # print(f'self.clash_cutoff = {self.clash_cutoff}')
        # print(f'self.boarder = {self.boarder}')

        self.boarder_embedding = AtomEncoder(emb_dim=ns, feature_dims=([2] * len(self.clash_cutoff), 1), sigma_embed_dim=0) if self.boarder else None
        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),
                                                nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.phore_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=phore_feature_dims, 
                                                sigma_embed_dim=sigma_embed_dim)
        self.phore_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), 
                                                  nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        cross_edge_dim = sigma_embed_dim + cross_distance_embed_dim
        if self.use_phore_match_feat:
            cross_edge_dim += 33
        
        if self.use_att:
            cross_edge_dim += ns
        
        self.cross_edge_embedding = nn.Sequential(nn.Linear(cross_edge_dim, ns), 
                                                  nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.phore_distance_expansion = GaussianSmearing(0.0, phore_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
    
        self.cross_distance_transition = nn.Sequential(nn.Linear(cross_distance_embed_dim, int(cross_distance_embed_dim / 2)), nn.ReLU(), 
                                                       nn.Dropout(dropout), nn.Linear(int(cross_distance_embed_dim / 2), 1), nn.Softplus())\
                                                           if new and phoretype_match and cross_distance_transition else None
        # self.cross_distance_transition = nn.Sequential(nn.Linear(cross_distance_embed_dim, int(cross_distance_embed_dim / 2)), nn.ReLU(), 
        #                                                nn.Dropout(dropout), nn.Linear(int(cross_distance_embed_dim / 2), 1))\
        #                                                    if new and phoretype_match and cross_distance_transition else None
        # self.phore_direction_transition = phore_direction_transition

        if self.use_att:
            self.OPM = OuterProductModule(inp_dim=ns, inp_dim2=ns, c=ns//2, out_dim=ns, bias=False)
            self.linear_att_l = nn.Linear(ns, ns, bias=False)
            self.linear_att_p = nn.Linear(ns, ns, bias=False)
            if self.trioformer_layer > 1:
                self.trioformer = nn.ModuleList([Trioformer(inp_dim=ns, c=ns*2, num_heads=4, bias=True, c_opm=ns//2, gatt_head=8, dropout=0) 
                                                 for i in range(self.trioformer_layer)])
            else:
                self.trioformer = Trioformer(inp_dim=ns, c=ns*2, num_heads=4, bias=True, c_opm=ns//2, gatt_head=8, dropout=0)
            self.mlp_att = nn.Sequential(nn.Linear(ns, ns*2), nn.LeakyReLU(), 
                                         nn.Dropout(dropout), nn.Linear(ns*2, 1), nn.LeakyReLU())

        self.phore_direction_transition = nn.Sequential(nn.Linear(1, self.num_phoretype), nn.LeakyReLU(), nn.Dropout(dropout),
                                                        nn.Linear(self.num_phoretype, 1), nn.LeakyReLU()) \
                                                            if new and phoretype_match and phore_direction_transition else None
        # self.phoretype_match_transition = nn.Sequential(nn.Linear(self.num_phoretype * 3, self.num_phoretype), nn.ReLU(), nn.Dropout(dropout), 
        #                                                 nn.Linear(self.num_phoretype, 1)) \
        self.phoretype_match_transition = nn.Sequential(nn.Linear(self.num_phoretype * 3, self.num_phoretype), nn.ReLU(), nn.Dropout(dropout), 
                                                        nn.Linear(self.num_phoretype, 1), nn.Softplus()) \
                                                            if new and phoretype_match and phoretype_match_transition else None
        # self.phoretype_match_transition = nn.Sequential(nn.Linear(self.num_phoretype * 3, self.num_phoretype), nn.ReLU(), nn.Dropout(dropout), 
        #                                                 nn.Linear(self.num_phoretype, 1), nn.ReLU()) \
        #                                                     if new and phoretype_match and phoretype_match_transition else None
        self.softmax = nn.Softmax(dim=0) if new else None
        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        lig_conv_layers, phore_conv_layers, lig_to_phore_conv_layers, phore_to_lig_conv_layers = [], [], [], []
        lig_to_phore_norm_conv_layers, phore_to_lig_norm_conv_layers = [], []
        lig_phorefp_layers = []
        
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            # print(f"parameters: {parameters}")
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            phore_layer = TensorProductConvLayer(**parameters)
            phore_conv_layers.append(phore_layer)
            lig_to_phore_layer = TensorProductConvLayer(**parameters)
            lig_to_phore_conv_layers.append(lig_to_phore_layer)
            phore_to_lig_layer = TensorProductConvLayer(**parameters)
            phore_to_lig_conv_layers.append(phore_to_lig_layer)
            if self.consider_norm:
                lig_to_phore_norm_conv_layer = TensorProductConvLayer(**parameters)
                phore_to_lig_norm_conv_layer = TensorProductConvLayer(**parameters)
                lig_to_phore_norm_conv_layers.append(lig_to_phore_norm_conv_layer)
                phore_to_lig_norm_conv_layers.append(phore_to_lig_norm_conv_layer)
            
            if self.auto_phorefp:
                lig_phorefp_layer = TensorProductConvLayer(**parameters)
                lig_phorefp_layers.append(lig_phorefp_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.phore_conv_layers = nn.ModuleList(phore_conv_layers)
        self.lig_to_phore_conv_layers = nn.ModuleList(lig_to_phore_conv_layers)
        self.phore_to_lig_conv_layers = nn.ModuleList(phore_to_lig_conv_layers)
        if self.consider_norm:
            self.lig_to_phore_norm_conv_layers = nn.ModuleList(lig_to_phore_norm_conv_layers)
            self.phore_to_lig_norm_conv_layers = nn.ModuleList(phore_to_lig_norm_conv_layers)
        if self.auto_phorefp:
            self.lig_phorefp_layers = nn.ModuleList(lig_phorefp_layers)
            self.mlp_phorefp = nn.Sequential(
                nn.Linear(self.lig_phorefp_layers[-1].tp.weight_numel, self.num_phoretype),
                nn.Sigmoid(), nn.Dropout(dropout), nn.Linear(self.num_phoretype, self.num_phoretype)
            )


    def forward(self, data):
        ## LMP Representation: [G_l, G_p, G_lp]
        # build ligand graph (G_l)
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)

        if self.boarder and self.boarder_embedding is not None:
            lig_node_boarder = self.boarder_analyze(data)
            # print(f"lig_node_attr.shape = {lig_node_attr.shape}")
            # print(f"lig_node_boarder.shape = {lig_node_boarder.shape}")
            lig_node_attr += self.boarder_embedding(lig_node_boarder)

        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build phore graph (G_p)
        phore_node_attr, phore_edge_index, phore_edge_attr, phore_edge_sh = self.build_phore_conv_graph(data)
        phore_src, phore_dst = phore_edge_index
        phore_node_attr = self.phore_node_embedding(phore_node_attr)
        phore_edge_attr = self.phore_edge_embedding(phore_edge_attr)

        # build bipartite graph (G_lp)
        cross_conv_graph_build_func = self._build_phoretype_cross_conv_graph
        if self.use_att:
            lig_node_attr, phore_node_attr, z_ij, _ = self.get_geometric_attention(data, lig_node_attr, phore_node_attr)
            cross_conv_graph_build_func = partial(cross_conv_graph_build_func, z_ij=z_ij)

        cross_edge_index, cross_edge_attr, cross_edge_sh, cross_edge_norm_sh = cross_conv_graph_build_func(data)
        cross_lig, cross_phore = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        ## Message Passing
        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # inter graph message passing
            phore_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], phore_node_attr[cross_phore, :self.ns]], -1)
            lig_inter_update = self.phore_to_lig_conv_layers[l](phore_node_attr, cross_edge_index, phore_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0])
            lig_inter_update_norm, phore_inter_update_norm = 0, 0
            phore_intra_update, phore_inter_update = 0, 0
            if self.consider_norm:
                lig_inter_update_norm = self.phore_to_lig_norm_conv_layers[l](phore_node_attr, cross_edge_index, phore_to_lig_edge_attr_, cross_edge_norm_sh,
                                                              out_nodes=lig_node_attr.shape[0])

            if l != len(self.lig_conv_layers) - 1:
                phore_edge_attr_ = torch.cat([phore_edge_attr, phore_node_attr[phore_src, :self.ns], phore_node_attr[phore_dst, :self.ns]], -1)
                phore_intra_update = self.phore_conv_layers[l](phore_node_attr, phore_edge_index, phore_edge_attr_, phore_edge_sh)

                lig_to_phore_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], phore_node_attr[cross_phore, :self.ns]], -1)
                phore_inter_update = self.lig_to_phore_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_phore_edge_attr_,
                                                                  cross_edge_sh, out_nodes=phore_node_attr.shape[0])
                if self.consider_norm:
                    phore_inter_update_norm = self.lig_to_phore_norm_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_phore_edge_attr_,
                                                                  cross_edge_norm_sh, out_nodes=phore_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update + lig_inter_update_norm

            if l != len(self.lig_conv_layers) - 1:
                phore_node_attr = F.pad(phore_node_attr, (0, phore_intra_update.shape[-1] - phore_node_attr.shape[-1]))
                phore_node_attr = phore_node_attr + phore_intra_update + phore_inter_update + phore_inter_update_norm
    
        return lig_node_attr, phore_node_attr


    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


    def build_phore_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        data['phore'].node_sigma_emb = self.timestep_emb_func(data['phore'].node_t['tr']) # tr rot and tor noise is all the same
        node_attr = torch.cat([data['phore'].x, data['phore'].node_sigma_emb], 1)

        edge_index = data['phore', 'phore'].edge_index
        src, dst = edge_index
        edge_vec = data['phore'].pos[dst.long()] - data['phore'].pos[src.long()]

        edge_length_emb = self.phore_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['phore'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


    def _build_phoretype_cross_conv_graph(self, data, z_ij=None, **kwargs):
        # builds the cross pharmacophore type and normal direction alignment 
        # between ligand and pharmacophores

        # mask for exclusion volumes
        lig_phorefp = data['ligand'].phorefp if not self.auto_phorefp else data['ligand'].auto_phorefp
        # mask_lig_fp = torch.sum(lig_phorefp, dim=-1) != 0
        # if len(mask_lig_fp) == 0:
        #     mask_lig_fp = torch.ones_like(data['ligand'].batch).bool().to(data['ligand'].batch.device)
        mask_phore_ex = data['phore'].phoretype[:, -1] == 1

        edge_index_1, batch_1 = fully_connect_two_graphs(data['ligand'].batch, data['phore'].batch, 
                                                mask_1=None, mask_2=torch.bitwise_not(mask_phore_ex),
                                                return_batch=True)
        edge_index_2, batch_2 = fully_connect_two_graphs(data['ligand'].batch, data['phore'].batch,
                                                mask_1=None, mask_2=mask_phore_ex, 
                                                return_batch=True)

        edge_index = torch.cat([edge_index_1, edge_index_2], axis=1) 
        perm = my_sort_edge_index(edge_index)
        edge_index = edge_index[:, perm]
        batch = torch.cat([batch_1, batch_2], axis=0)[perm] 
        src, dst = edge_index
        
        edge_vec = data['phore'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        rotate_norm = data['phore'].norm[dst.long()] if not self.angle_match else 0
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)

        if self.phoretype_match or self.angle_match:
            aggreement_1 = data['phore'].phoretype[edge_index_1[1]] * lig_phorefp[edge_index_1[0]]
            aggreement_2 = torch.zeros_like(data['phore'].phoretype[edge_index_2[1]]).to(data['phore'].x.device)

            aggreement = torch.cat([aggreement_1, aggreement_2], axis=0)[perm, :]

            if self.phoretype_match:
                phoretype_attr = torch.cat([aggreement, data['phore'].phoretype[dst.long()], lig_phorefp[src.long()]], axis=-1)
                # Wrong calculation of phoretype_attr: perm can't be employed twice.
                # phoretype_attr = torch.cat([aggreement, data['phore'].phoretype[dst.long()], lig_phorefp[src.long()]], axis=-1)[perm, :] # type: ignore
                if self.new:
                    total_weight = 1.0
                    if self.cross_distance_transition:
                        distance = self.cross_distance_transition(edge_length_emb)
                        total_weight = distance * total_weight
                        # print(f"distance: {distance}")
                        
                    if self.phoretype_match_transition:
                        feat_match = self.phoretype_match_transition(phoretype_attr)
                        # print(f"feat_match: {feat_match}")
                        total_weight = feat_match * total_weight
                    
                    total_weight = total_weight * getattr(self, 'scaler', 1.0)
                    
                    if self.phore_direction_transition:
                        direction = torch.pow(-1, (self.phore_direction_transition(total_weight) < 0).float())
                        # print(f"direction: {direction}")
                        edge_vec = edge_vec * direction

                    atom_weight = 1.0
                    if self.atom_weight == 'softmax':
                        atom_weight = self.softmax(total_weight) if not isinstance(total_weight, float) \
                            and self.softmax is not None else 1.0
                    elif self.atom_weight == 'sigmoid':
                        atom_weight = torch.sigmoid(total_weight) if not isinstance(total_weight, float) else 1.0
                    elif self.atom_weight == 'atomwise':
                        ## calcuate the edge-wise weight of pharmacophore-derived driving force for each atom-pharmacophore pair.
                        ## Suppose atom A1, A2 has connections with all pharmacophores B ({b1, b2, b3, b4}), here the weight is calculated for the 8 edges.
                        ## E.g., A1b1=0.05, A1b2=0.2, A1b3=0.025, A1b4=0.225, A2b1=0.05, A2b2=0.2, A2b3=0.025, A2b4=0.225
                        # print(f"total_weight0: {total_weight}")
                        atom_weight = total_weight.exp() / scatter(total_weight.exp(), batch, dim=0, reduce='sum')[batch] \
                            if not isinstance(total_weight, float) else 1.0
                        # total_weight = total_weight.exp() / scatter(total_weight.exp(), src.long(), dim=0, reduce='sum')[src.long()] \
                        #     if not isinstance(total_weight, float) else total_weight
                        # print(f"batch: {batch}")
                        # print("total_weight1:", )
                    elif self.atom_weight == 'phore':
                        ## Calcuate the weight of pharmacophore-derived driving force for all pharmacophore connecting to the same atom.
                        ## Atom A -> PH/EX P1,P2,P3,E4. To calculate the weight for P1,P2,P3,E4 whose sum is 1.
                        # print(f"total_weight0: {total_weight}")
                        atom_weight = total_weight.exp() / scatter(total_weight.exp(), src.long(), dim=0, reduce='sum')[src.long()] \
                            if not isinstance(total_weight, float) else 1.0
                        
                        # print("SRC:", src)
                        # print("total_weight1:", total_weight)
                    #print('atom_weight:', atom_weight)
                    total_weight = total_weight * atom_weight + 1e-12 if getattr(self, 'multiple', True) else atom_weight
                    #print("total_weight1:", total_weight)

                    edge_vec = edge_vec * total_weight
                    # if self.phore_direction_transition:
                    #     direction = self.phore_direction_transition(phoretype_direction) > 0
                    #     print("direction:", direction)
                    #     edge_vec = edge_vec * direction.float()
                else:
                    phoretype_direction_1 = torch.pow(-1, torch.sum(aggreement_1, dim=-1).unsqueeze(-1) - 1)
                    phoretype_direction_2 = self.ex_factor * torch.ones_like(edge_index_2[1]).unsqueeze(-1).to(data['phore'].x.device)
                    # phoretype_direction = torch.cat([phoretype_direction_1, phoretype_direction_2], axis=0)
                    phoretype_direction = torch.cat([phoretype_direction_1, phoretype_direction_2], axis=0)[perm, :] # type: ignore
                    edge_vec = edge_vec * phoretype_direction

                if self.use_phore_match_feat:
                    # print(f'edge_attr.shape: {edge_attr.shape}')
                    edge_attr = torch.cat([edge_attr, phoretype_attr], axis=-1)

            if self.use_att and z_ij is not None:
                edge_attr = torch.cat([edge_attr, z_ij], axis=-1)
                edge_vec = self.mlp_att(z_ij) * edge_vec

                    # print(f'phoretype_attr.shape: {phoretype_attr.shape}')
            # total_weight = self.phore_direction_transition(phoretype_direction)


            # pharmacophore norm match
            # select matching pharmacophore norm corresponding to pharmacophore type from pre-calculated norms
            if self.angle_match:
                lig_norm = torch.sum(aggreement.unsqueeze(-1) * \
                                    data['ligand'].norm[src.long()].reshape(-1, self.num_phoretype, 3), dim=1)
                rotate_norm = torch.clip(torch.cross(lig_norm, data['phore'].norm[dst.long()]), 1e-12) \
                    * torch.sum(aggreement, dim=-1, keepdim=True)
                rotate_norm = F.normalize(rotate_norm)
                curr_angle = angle_vectors(lig_norm, data['phore'].norm[dst.long()]).unsqueeze(-1)
                lig_norm_angle1 = torch.sum(aggreement * data['ligand'].norm_angle1[src.long()], dim=-1, keepdim=True)
                lig_norm_angle2 = torch.sum(aggreement * data['ligand'].norm_angle2[src.long()], dim=-1, keepdim=True)
                norm_delta_angle = torch.concat([torch.abs(curr_angle - lig_norm_angle1), 
                                                torch.abs(curr_angle - lig_norm_angle2)], dim=-1)
                smaller_angle_index = torch.sort(norm_delta_angle, dim=-1).indices[:, 0].unsqueeze(-1)
                norm_real = torch.gather(torch.concat(
                    [curr_angle - lig_norm_angle1, curr_angle - lig_norm_angle2], dim=-1), dim=1, index=smaller_angle_index)

                rotate_norm = rotate_norm * norm_real

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, 
                                         normalize=True, normalization='component')
        edge_norm_sh = o3.spherical_harmonics(self.sh_irreps, rotate_norm, 
                                              normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh, edge_norm_sh


    def boarder_analyze(self, data):
        """
        Analyzes the border information between ligand atoms and their closest exclusion volumes.

        Args:
            data (dict): A dictionary containing the following keys:
                - 'ligand': A dictionary with keys:
                    - 'pos': Tensor of shape (N, 3) representing the positions of ligand atoms.
                    - 'batch': Tensor of shape (N,) representing the batch indices of ligand atoms.
                    - 'x': Tensor of shape (N, F) representing the features of ligand atoms.
                - 'phore': A dictionary with keys:
                    - 'pos': Tensor of shape (M, 3) representing the positions of phore atoms.
                    - 'batch': Tensor of shape (M,) representing the batch indices of phore atoms.
                    - 'phoretype': Tensor of shape (M, T) representing the types of phore atoms.

        Returns:
            Tensor: A tensor representing the border analysis results, with clashed information and minimum distances.
        """
        l_pos, l_mask = to_dense_batch(data['ligand'].pos, batch=data['ligand'].batch)
        p_pos, p_mask = to_dense_batch(data['phore'].pos, batch=data['phore'].batch)
        ex_mask, _ = to_dense_batch(data['phore'].phoretype[:, -1] == 1, batch=data['phore'].batch, fill_value=False)
        dis_mask = l_mask.unsqueeze(-1) * (p_mask.unsqueeze(1) * ex_mask.unsqueeze(1))
        dis_min, _ = torch.min(torch.cdist(l_pos, p_pos) + (1 - dis_mask.float()) * 1e9, dim=-1, keepdim=False)
        dis_min = dis_min[l_mask].unsqueeze(-1)

        ## Radius of Exlcusion Volume
        # alpha = K / radius ** 2,  K = 2.41798725037, alpha_ex = 0.837
        if self.by_radius: 
            from datasets.process_mols import atom_radiuses
            r_ex = (2.41798725037 / 0.837) ** 0.5
            r_atom = torch.tensor(atom_radiuses).to(data['ligand'].pos.device)[data['ligand'].x[:, 0].long()].unsqueeze(-1)
            clashed = dis_min - r_atom - r_ex <= self.clash_tolerance

        else:
            # clashed = []
            # for idx in range(len(self.clash_cutoff)):
            #     if idx == 0:
            #         c = dis_min <= self.clash_cutoff[idx]
            #     elif idx > 0 and idx < len(self.clash_cutoff) - 1:
            #         c = (dis_min > self.clash_cutoff[idx-1]) * (dis_min <= self.clash_cutoff[idx])
            #     else:
            #         c = dis_min > self.clash_cutoff[idx]
            #     clashed.append(c)
            clashed = dis_min.tile([1, len(self.clash_cutoff)]) <= torch.tensor(self.clash_cutoff).to(dis_min.device)
        boarder = torch.cat([clashed, dis_min], axis=-1) # type: ignore
        return boarder


    def get_geometric_attention(self, data, lig_node_feat, phore_node_feat):
        """
        Computes the geometric attention between ligand and pharmacophore node features.

        Args:
            data (HeteroDataBatch): The batch of HeteroGraph containing the ligand and pharmacophore data.
                - 'ligand': Contains the ligand data with attributes 'batch' and 'pos'.
                - 'phore': Contains the pharmacophore data with attributes 'batch' and 'pos'.
            lig_node_feat (torch.Tensor): Node features of the ligand.
            phore_node_feat (torch.Tensor): Node features of the pharmacophore.

        Returns:
            tuple: A tuple containing:
                - h_l (torch.Tensor): Updated ligand node features.
                - h_p (torch.Tensor): Updated pharmacophore node features.
                - z_ij (torch.Tensor): Attention scores between ligand and pharmacophore nodes.
                - weights (torch.Tensor): Weights from the trioformer layers.
        """
        h_l = self.linear_att_l(lig_node_feat)
        h_p = self.linear_att_p(phore_node_feat)
        # print(f"[I] Trioformer lig_node_feat.shape = {lig_node_feat.shape}")
        # print(f"[I] Trioformer phore_node_feat.shape = {phore_node_feat.shape}")
        h_l, mask_l = to_dense_batch(h_l, data['ligand'].batch)
        h_p, mask_p = to_dense_batch(h_p, data['phore'].batch)
        mask_z = mask_l.unsqueeze(-1) * mask_p.unsqueeze(-2)

        # print(f'[I] mask_l.shape = {mask_l.shape}')
        # print(f'[I] mask_p.shape = {mask_p.shape}')
        coord_l, _ = to_dense_batch(data['ligand'].pos, data['ligand'].batch)
        coord_p, _ = to_dense_batch(data['phore'].pos, data['phore'].batch)
        d_ik = torch.cdist(coord_l, coord_l) * (mask_l.unsqueeze(-1) * mask_l.unsqueeze(-2)).float()
        d_jk_ = torch.cdist(coord_p, coord_p) * (mask_p.unsqueeze(-1) * mask_p.unsqueeze(-2)).float()
        z_ij = self.OPM(h_l, h_p)
        if self.trioformer_layer > 1:
            for i in range(self.trioformer_layer):
                h_l, h_p, z_ij, weights = self.trioformer[i](h_l, h_p, z_ij, d_ik, d_jk_, mask_l, mask_p, return_weight=True)
        else:
            h_l, h_p, z_ij, weights = self.trioformer(h_l, h_p, z_ij, d_ik, d_jk_, mask_l, mask_p, return_weight=True)

        return h_l[mask_l], h_p[mask_p], z_ij[mask_z], weights


class GaussianSmearing(torch.nn.Module):
    """
    A PyTorch module for Gaussian smearing of edge distances.

    This module is used to embed edge distances into a higher-dimensional space
    using Gaussian functions. It creates a set of Gaussian functions with means
    evenly spaced between `start` and `stop`, and a fixed standard deviation
    determined by the spacing between the means.

    Args:
        start (float, optional): The starting value of the range for the Gaussian means. Default is 0.0.
        stop (float, optional): The ending value of the range for the Gaussian means. Default is 5.0.
        num_gaussians (int, optional): The number of Gaussian functions to create. Default is 50.

    Attributes:
        coeff (float): The coefficient used in the exponent of the Gaussian function.
        offset (torch.Tensor): A tensor containing the means of the Gaussian functions.

    Methods:
        forward(dist):
            Applies the Gaussian smearing to the input distances.

            Args:
                dist (torch.Tensor): A tensor of distances to be smeared.

            Returns:
                torch.Tensor: A tensor containing the smeared distances.
    """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def angle_vectors(a, b, dim=-1):
    """
    Calculate the angle between two vectors a and b along a specified dimension.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.
        dim (int, optional): The dimension along which to compute the angle. Default is -1.

    Returns:
        torch.Tensor: The angle between the vectors a and b in radians.
    """
    a_norm = a.norm(dim=dim, keepdim=True)
    b_norm = b.norm(dim=dim, keepdim=True)
    return 2 * torch.atan2(
        (a * b_norm - a_norm * b).norm(dim=dim),
        (a * b_norm + a_norm * b).norm(dim=dim)
    )


def fully_connect_two_graphs(batch_1, batch_2, mask_1=None, mask_2=None, return_batch=False):
    """
    Fully connects two graphs represented by batches of nodes, optionally using masks to filter nodes.

    Args:
        batch_1 (torch.Tensor): A tensor representing the first batch of nodes.
        batch_2 (torch.Tensor): A tensor representing the second batch of nodes.
        mask_1 (torch.Tensor, optional): A boolean mask tensor for the first batch. Defaults to None.
        mask_2 (torch.Tensor, optional): A boolean mask tensor for the second batch. Defaults to None.
        return_batch (bool, optional): If True, returns the batch indices along with the new indices. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the new indices connecting the two graphs.
        torch.Tensor (optional): A tensor containing the batch indices if return_batch is True.
    """
    mask_1 = torch.ones_like(batch_1).bool() if mask_1 is None else mask_1
    mask_2 = torch.ones_like(batch_2).bool() if mask_2 is None else mask_2
    index_1 = torch.arange(len(batch_1)).to(batch_1.device)
    index_2 = torch.arange(len(batch_2)).to(batch_2.device)
    masked_index_1 = index_1[mask_1]
    masked_index_2 = index_2[mask_2]
    masked_batch_1 = batch_1[mask_1]
    masked_batch_2 = batch_2[mask_2]
    new_index = []
    batch = []
    for i in torch.unique(masked_batch_1):
        _mask_1 = masked_batch_1 == i
        _mask_2 = masked_batch_2 == i
        _masked_index_1 = masked_index_1[_mask_1]
        _masked_index_2 = masked_index_2[_mask_2]
        len_1 = _masked_index_1.shape[0]
        len_2 = _masked_index_2.shape[0]
        new_index.append(torch.concat([_masked_index_1.unsqueeze(-1).tile([1, len_2]).reshape(1, -1), 
                                       _masked_index_2.unsqueeze(-1).tile([1, len_1]).T.reshape(1, -1)], 
                                      axis=0)) # type: ignore
        if return_batch:
            batch += [i] * (len_1 * len_2)
    new_index = torch.concat(new_index, axis=1).long() # type: ignore

    if return_batch:
        return new_index, torch.tensor(batch).long().to(batch_1.device)
    return new_index


def my_sort_edge_index(edge_index):
    """
    Sorts the edge indices based on a unique key generated from the edge indices.

    The function generates a unique key for each edge by multiplying the source node index by 
    the maximum node index plus one, and then adding the target node index. It then sorts the 
    edges based on these keys.

    Args:
        edge_index (torch.Tensor): A 2D tensor where each column represents an edge with 
                                   source and target node indices.

    Returns:
        torch.Tensor: A 1D tensor containing the indices that would sort the edge_index.
    """
    return (edge_index[0] * (int(edge_index.max()) + 1) + edge_index[1]).argsort()

