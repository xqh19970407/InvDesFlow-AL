import math
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean, scatter_max
import numpy as np
from e3nn.nn import BatchNorm
from torch_geometric.data import Batch
import time
from e3nn.o3 import Linear


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, sigma_embed_dim=0, lm_embedding_type= None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        

    def forward(self, x):
        
        x = x.to(self.atom_embedding_list[0].weight.device)
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
        return x_embedding



class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index.to(edge_attr.device)
        
        
        tp = self.tp(node_attr[edge_dst], edge_sh.to(edge_attr.device), self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class GaussianSmearing(torch.nn.Module):
    
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1).to(self.offset.device) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


    
def relu(x):
    return torch.nn.functional.relu(x)

class SuperconGNN(torch.nn.Module):
    def __init__(self, ns=128, nv=10, sh_lmax=2,
                 num_conv_layers=6, 
                 material_max_radius=5,
                 in_material_edge_features=2, 
                 distance_embed_dim=32, 
                 batch_norm=True,
                 dropout=0.0,
                 use_second_order_repr=True,
                 use_three_order_repr = False,
                 material_feature_dims=([100], 6), 
                 residual=True,
                 ):
        super(SuperconGNN, self).__init__()
        self.material_max_radius = material_max_radius
        self.in_material_edge_features = in_material_edge_features
        self.material_distance_expansion = GaussianSmearing(0.0, material_max_radius, distance_embed_dim)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns = ns
        self.material_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=material_feature_dims)
        self.material_edge_embedding = nn.Sequential(nn.Linear(in_material_edge_features  + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        
        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
            if use_three_order_repr:
                irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o + {nv}x3o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o + {nv}x3o + {nv}x3e',
                ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        
        material_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            material_parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': residual,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            material_layer = TensorProductConvLayer(**material_parameters)
            material_conv_layers.append(material_layer)
        self.material_conv_layers = nn.ModuleList(material_conv_layers)
        
        self.linear = Linear(
            irreps_in=self.material_conv_layers[-1].out_irreps, irreps_out='1x0e'
        )
            
    def forward(self, data):
        if type(data)==list:
            data = Batch.from_data_list(data)
       
        material_node_attr, material_edge_index, material_edge_attr, material_edge_sh = self.build_material_conv_graph(data)
        material_src, material_dst = material_edge_index
        material_node_attr = self.material_node_embedding(material_node_attr)
        material_edge_attr = self.material_edge_embedding(material_edge_attr)
        
       
        for i in range(len(self.material_conv_layers)):
            material_edge_attr_ = torch.cat([material_edge_attr, material_node_attr[material_src, :self.ns], material_node_attr[material_dst, :self.ns]], -1)
            material_intra_update = self.material_conv_layers[i](material_node_attr, material_edge_index, material_edge_attr_, material_edge_sh) 
            material_node_attr = F.pad(material_node_attr, (0, material_intra_update.shape[-1] - material_node_attr.shape[-1]))
            material_node_attr =  material_node_attr + relu(  material_intra_update )
            
        atom_supercon_tc =  relu( self.linear(material_node_attr) )
        supercon_tc = scatter(atom_supercon_tc, data['material'].batch.to(atom_supercon_tc.device), dim=0, reduce='mean')
        return supercon_tc
    
    
    def build_material_conv_graph(self, data):
              
        
        radius_edges = radius_graph(data['material'].pos, self.material_max_radius, data['material'].batch) # 建立虚拟边,batch
        edge_index = torch.cat([data['material', 'material'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['material', 'material'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_material_edge_features, device=data['material'].x.device) # 将半径边concat进来
        ], 0)
        
        node_attr = data['material'].x
        src, dst = edge_index
        edge_vec = data['material'].pos[dst.long()] - data['material'].pos[src.long()]
        edge_length_emb = self.material_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr.to(edge_length_emb.device), edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        
        return node_attr, edge_index, edge_attr, edge_sh
    
