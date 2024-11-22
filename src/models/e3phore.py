import math
from functools import partial
from turtle import forward

import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean

from datasets.process_mols import lig_feature_dims
from datasets.process_pharmacophore import phore_feature_dims

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims):
        # first element of feature_dims tuple is a list with the length of each categorical feature 
        # and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)


    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:\
                                         self.num_categorical_features + self.num_scalar_features])
        return x_embedding


class MHAWithPairBias(nn.Module):
    def __init__(self, inp_dim=16, c=32, num_heads=4, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.att_head_size = c
        self.all_head_size = self.att_head_size * self.num_heads
        self.linear_q = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_b = nn.Linear(inp_dim, self.num_heads, bias=False) if bias else None
        self.layernorm = nn.LayerNorm(inp_dim)
        self.final_linear = nn.Linear(self.all_head_size, inp_dim)
        self.softmax = nn.Softmax(dim=-1)

    def reshape_last_dim(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.att_head_size))

    def forward(self, q, k, v, mask=None, bias=None, return_weight=False):
        B, Q, _ = q.shape
        q = self.reshape_last_dim(self.linear_q(q))
        k = self.reshape_last_dim(self.linear_k(k))
        v = self.reshape_last_dim(self.linear_v(v))

        logits = torch.einsum('bqhc,bkhc->bhqk', q, k) * (self.num_heads ** (-0.5))
        if mask is not None:
            logits = logits + ((mask.unsqueeze(1).float() * 1e9 - 1.))

        if bias is not None and self.linear_b is not None:
            bias = self.linear_b(bias).permute(0, 3, 1, 2) # [B, I, J, F] -> [B, I, J, H] -> [B, H, I, J]
            logits = logits + bias
        
        weights = self.softmax(logits)
        output = self.final_linear(torch.einsum('bhqk,bkhc->bqhc', weights, v).reshape(B, Q, self.all_head_size))
        output = self.layernorm(output)
        if return_weight:
            return output, weights
        else:
            return output, None


class OuterProductModule(nn.Module):
    def __init__(self, inp_dim, inp_dim2, c=16, out_dim=32, bias=False):
        super().__init__()
        self.layernorm_l = nn.LayerNorm(inp_dim)
        self.layernorm_p = nn.LayerNorm(inp_dim2)
        self.linear_l = nn.Linear(inp_dim, c, bias=bias)
        self.linear_p = nn.Linear(inp_dim2, c, bias=bias)
        self.linear_final = nn.Linear(1, out_dim)

    def forward(self, h_l, h_p):
        h_l = self.linear_l(self.layernorm_l(h_l))
        h_p = self.linear_p(self.layernorm_p(h_p))
        z_ij = self.linear_final(torch.mean(h_l.unsqueeze(2) * h_p.unsqueeze(1), dim=-1, keepdim=True))
        return z_ij


class GeometryConstraitUpdate(nn.Module):
    def __init__(self, inp_dim, c=32, num_heads=8):
        super().__init__()
        self.inp_dim = inp_dim
        self.num_heads = num_heads
        self.att_head_size = c
        self.all_head_size = self.att_head_size * self.num_heads

        self.layernorm = nn.LayerNorm(inp_dim)
        self.linear_q = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(inp_dim, self.all_head_size, bias=False)
        self.linear_b = nn.Linear(inp_dim, self.num_heads, bias=False)
        self.linear_d = nn.Linear(1, self.num_heads, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.g = nn.Linear(inp_dim, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, inp_dim)

    def reshape_last_dim(self, x):
        return x.view(x.size()[:-1] + (self.num_heads, self.att_head_size))

    def forward(self, z_ij, d_jk_, mask_z=None, return_weight=False):
        z_ij = self.layernorm(z_ij)
        q = self.reshape_last_dim(self.linear_q(z_ij)) * (self.num_heads ** (-0.5)) # [B, I, J, H, C]
        k = self.reshape_last_dim(self.linear_k(z_ij)) # [B, I, J, H, C]
        v = self.reshape_last_dim(self.linear_v(z_ij)) # [B, I, J, H, C]
        b = self.linear_b(z_ij).permute(0, 1, 3, 2).unsqueeze(-1) # [B, I, J, H] -> [B, I, H, J, 1]
        d = self.linear_d(d_jk_).permute(0, 3, 1, 2).unsqueeze(1) # [B, J, J, H] -> [B, 1, H, J, J]
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + b + d # [B, I, H, J, J]
        if mask_z is not None:
            att_mask = 1e9 * (mask_z.unsqueeze(-2).unsqueeze(-2).float() - 1.0) # [B, I, 1, 1, J]
            logits = logits + att_mask
        weights = self.softmax(logits) # [B, I, H, J, J]
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v) # [B, I, J, H, C]

        g = self.reshape_last_dim(self.g(z_ij)).sigmoid()
        output = g * weighted_avg
        output = output.reshape(output.size()[:-2] + (self.all_head_size, ))
        output = self.final_linear(output) 
        if mask_z is not None:
            output = output * mask_z.unsqueeze(-1)

        if return_weight:
            return output, weights
        else:
            return output, None


class Trioformer(nn.Module):
    def __init__(self, inp_dim=16, c=32, num_heads=4, bias=True, c_opm=8, gatt_head=8, dropout=0.):
        super().__init__()
        ## Protein-ligand node embedding update with multi-head cross-attention
        self.mha_l = MHAWithPairBias(inp_dim=inp_dim, c=c, num_heads=num_heads, bias=bias)
        self.mha_p = MHAWithPairBias(inp_dim=inp_dim, c=c, num_heads=num_heads, bias=bias)

        ## Node level transitions
        self.transition_l = nn.Sequential(nn.Linear(inp_dim, inp_dim*2, bias=False), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(inp_dim*2, inp_dim, bias=False))
        self.transition_p = nn.Sequential(nn.Linear(inp_dim, inp_dim*2, bias=False), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(inp_dim*2, inp_dim, bias=False))

        ## Pair embeddings update with Outer Product Module (OPM)
        self.opm = OuterProductModule(inp_dim=inp_dim, inp_dim2=inp_dim, c=c_opm, out_dim=inp_dim)

        ## Geometry-aware pair update
        self.gapu_l = GeometryConstraitUpdate(inp_dim=inp_dim, c=c, num_heads=gatt_head)
        self.gapu_p = GeometryConstraitUpdate(inp_dim=inp_dim, c=c, num_heads=gatt_head)

    def forward(self, h_l, h_p, z_ij, d_ik, d_jk_, mask_l=None, mask_p=None, return_weight=True):
        # print(f'[I] Trioformer mask_l.shape = {mask_l.shape}')
        # print(f'[I] Trioformer mask_p.shape = {mask_p.shape}')
        mask_z = mask_l.unsqueeze(-1) * mask_p.unsqueeze(-2) if mask_l is not None and mask_p is not None else None
        
        h_l_update, att_weight_p2l = self.mha_l(h_l, h_p, h_p, mask_z, z_ij, return_weight=return_weight)
        h_p_update, att_weight_l2p = self.mha_p(h_p, h_l, h_l, 
                                                mask_z.permute(0, 2, 1), z_ij.permute(0, 2, 1, 3), 
                                                return_weight=return_weight)
        h_l = h_l + h_l_update
        h_p = h_p + h_p_update
        h_l = h_l + self.transition_l(h_l)
        h_p = h_p + self.transition_p(h_p)

        z_ij = z_ij + self.opm(h_l, h_p)

        z_ij_update_l, att_weight_geo_l = self.gapu_l(z_ij.permute(0, 2, 1, 3), 
                                                      d_ik.unsqueeze(-1), mask_z.permute(0, 2, 1), 
                                                      return_weight=return_weight)
        z_ij_update_p, att_weight_geo_p = self.gapu_p(z_ij, d_jk_.unsqueeze(-1), mask_z, 
                                                      return_weight=return_weight)
        z_ij = z_ij + z_ij_update_l.permute(0, 2, 1 ,3) + z_ij_update_p

        if return_weight:
            weights = {
                'att_weight_p2l': att_weight_p2l, 'att_weight_l2p': att_weight_l2p,
                'att_weight_geo_l': att_weight_geo_l, 'att_weight_geo_p': att_weight_geo_p
                }
            return h_l, h_p, z_ij, weights
        else:
            return h_l, h_p, z_ij, None


class CoordRefine(nn.Module):
    def __init__(self, h_l_dim, h_p_dim, z_dim, hidden_dim=128, n_layers=20, dropout=0.1,
                 edge_dim_l=None, edge_dim_lp=None, **kwargs):
        super().__init__()
        self.phi_m1s = nn.ModuleList([])
        self.phi_m2s = nn.ModuleList([])
        self.varphi_ms = nn.ModuleList([])
        self.phi_x1s = nn.ModuleList([])
        self.phi_x2s = nn.ModuleList([])
        self.n_layers = n_layers
        self.edge_dim_l = edge_dim_l
        self.edge_dim_lp = edge_dim_lp

        ij_dim = h_l_dim + h_p_dim + z_dim + 1
        ik_dim = h_l_dim * 2 + 1
        if edge_dim_lp is not None:
            ij_dim += edge_dim_lp
        if edge_dim_l is not None:
            ik_dim += edge_dim_l
        
        for i in range(n_layers):
            self.phi_m1s.append(nn.Sequential(nn.Linear(ij_dim, hidden_dim), nn.ReLU(), 
                                              nn.Linear(hidden_dim, h_l_dim), nn.Dropout(dropout), nn.ReLU()))
            self.phi_m2s.append(nn.Sequential(nn.Linear(ij_dim, hidden_dim), nn.ReLU(), 
                                              nn.Linear(hidden_dim, h_p_dim), nn.Dropout(dropout), nn.ReLU()))
            self.varphi_ms.append(nn.Sequential(nn.Linear(ik_dim, hidden_dim), nn.ReLU(), 
                                                nn.Linear(hidden_dim, h_l_dim), nn.Dropout(dropout), nn.ReLU()))

            self.phi_x1s.append(nn.Sequential(nn.Linear(h_l_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1), 
                                              nn.Dropout(dropout), nn.LeakyReLU()))
            self.phi_x2s.append(nn.Sequential(nn.Linear(h_l_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1), 
                                              nn.Dropout(dropout), nn.LeakyReLU()))

    def forward(self, h_l, h_p, x_l, x_p, edge_index_lp, edge_index_l, z_ij,
                edge_attr_l=None, edge_attr_lp=None, **kwargs):
        src, dst = edge_index_lp
        src_l, dst_l = edge_index_l
        for i in range(self.n_layers):
            # Calculate distance
            vec_x_ij = x_p[dst] - x_l[src]
            vec_x_ik = x_l[dst_l] - x_l[src_l]
            d_ij = torch.norm(vec_x_ij, dim=-1)
            d_ik = torch.norm(vec_x_ik, dim=-1)

            # Message passing
            if self.edge_dim_lp is not None:
                _m_ij = torch.cat([h_l[src], h_p[dst], z_ij, d_ij, edge_attr_lp], dim=-1)
            else:
                _m_ij = torch.cat([h_l[src], h_p[dst], z_ij, d_ij], dim=-1)
            
            if self.edge_dim_l is not None:
                _m_ik = torch.cat([h_l[src_l], h_l[dst_l], d_ik, edge_attr_l], dim=-1)
            else:
                _m_ik = torch.cat([h_l[src_l], h_l[dst_l], d_ik], dim=-1)

            m_ij = self.phi_m1s[i](_m_ij)
            m_ji = self.phi_m2s[i](_m_ij)
            m_ik = self.varphi_ms[i](_m_ik)

            # Update the node features
            h_l = h_l + scatter_sum(m_ij, src, dim=0) + scatter_sum(m_ik, src_l, dim=0)
            h_p = h_p + scatter_sum(m_ji, dst, dim=0)

            # Update the coordinates
            vec_x_ij = vec_x_ij / (d_ij.unsqueeze(-1) + 1e-10)
            vec_x_ik = vec_x_ik / (d_ik.unsqueeze(-1) + 1e-10)
            delta_x_ij = scatter_sum(self.phi_x1s[i](m_ij) * vec_x_ij, src, dim=0)
            delta_x_ik = scatter_sum(self.phi_x2s[i](m_ik) * vec_x_ik, src_l, dim=0)
            x_l = x_l + delta_x_ij + delta_x_ik

        return h_l, x_l, h_p, x_p


class FeatureEmbedding(nn.Module):
    def __init__(self, lig_feature_dims, phore_feature_dims, in_lig_edge_features=4, hidden_dim=16, dropout=0.0):
        super().__init__()
        self.lig_node_embedding = AtomEncoder(emb_dim=hidden_dim, feature_dims=lig_feature_dims)
        self.phore_node_embedding = AtomEncoder(emb_dim=hidden_dim, feature_dims=phore_feature_dims)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features, hidden_dim),
                                        nn.ReLU(), nn.Dropout(dropout),nn.Linear(hidden_dim, hidden_dim))
        self.phore_edge_embedding = nn.Sequential(nn.Linear(1, hidden_dim), 
                                                  nn.ReLU(), nn.Dropout(dropout),nn.Linear(hidden_dim, hidden_dim))

    def forward(self, data):
        lig_node_attr = self.lig_node_embedding(data['ligand'].x)
        lig_edge_attr = self.lig_edge_embedding(data['ligand', 'ligand'].edge_attr)
        phore_node_attr = self.phore_node_embedding(data['phore'].x)
        phore_edge_attr = self.phore_edge_embedding(data['phore', 'phore'].edge_attr)
        return lig_node_attr, lig_edge_attr, phore_node_attr, phore_edge_attr


class E3Phore(nn.Module):
    def __init__(self, in_lig_edge_features=4, hidden_dim=16, dropout=0.0, n_trioformer_blocks=8, 
                 c=32, num_heads=4, bias=True, c_opm=8, gatt_head=8):
        super().__init__()
        self.input_embedding = FeatureEmbedding(in_lig_edge_features=in_lig_edge_features, hidden_dim=hidden_dim, dropout=dropout)
        self.n_trioformer_blocks = n_trioformer_blocks
        self.trioformer = nn.ModuleList([Trioformer(hidden_dim, c=c, num_heads=num_heads, bias=bias, dropout=dropout,
                                                    c_opm=c_opm, gatt_head=gatt_head) for i in range(n_trioformer_blocks)])
        self.coordrf = CoordRefine()

    def forward(self, data):
        h_l, edge_attr_l, h_p, edge_attr_p = self.input_embedding(data)
        for i in range(self.n_trioformer_blocks):
            h_l, h_p, z_ij, weights = self.trioformer[i](data)
        pred = self.coordrf(data)
        return pred

