import json
from pymatgen.core.structure import Structure
from torch_geometric.data import HeteroData, Dataset
import torch
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
import numpy as np
from pymatgen.core.lattice import Lattice
import torch.nn.functional as F
import os
import pickle
import copy
CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)



def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def load_graph(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

import csv
def read_csv(file_path):
    data_list = []
    
    with open(file_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        
        for row in reader:
            data_list.append(row)
    
    return data_list
class CrystDataset(Dataset):
    def __init__(self, 
                 cif=False,
                 graph=False,
                 csv=False,
                 json_file_path='./data/mp.2018.6.1.json', 
                 graph_file_path='./data/process_6w_graph',
                 csv_file_path='./data/materials36/data_materials.csv',
                 cif_dir = './data/diffcsp_gen_data',
                 datatype='train',
                 pre_load=True,
                 size=None
                 ):
        super().__init__()
        # train/val/test 60000
        self.json_file_path = json_file_path
        self.graph_file_path = graph_file_path
        self.datatype = datatype
        self.graph = graph
        self.pre_load = pre_load
        self.csv = csv
        self.cif = cif
        if graph:
            if 'jid' in graph_file_path:
                graph_path = [os.path.join(graph_file_path,name) for name in [name for name in os.listdir(graph_file_path)]]
                # if self.datatype=='train':
                self.graph_path = graph_path
                if pre_load:
                    print('preload')
                    graph_list = [load_graph(path=path) for path in self.graph_path]
                    self.graph_list = copy.deepcopy(graph_list)
                    append_num = 0
                    if self.datatype=='train':
                        
                        for g in graph_list:
                            # if 'H' in g['name']:
                            append_num += 1
                            if 10 < g.Tc <= 30:
                                self.graph_list += [g] * 20
                            elif 30 < g.Tc :
                                self.graph_list += [g] * 30
                    print(f'{datatype} size ', len(self.graph_list), append_num)

            elif 'supercon' in graph_file_path:
                graph_path = [os.path.join(graph_file_path,name) for name in [name for name in os.listdir(graph_file_path)]]
                if self.datatype=='train' or 'test':
                    self.graph_path = graph_path
                if pre_load:
                    print('preload')
                    graph_list = [load_graph(path=path) for path in self.graph_path]
                    self.graph_list = copy.deepcopy(graph_list)
                    append_num = 0
                    for g in graph_list:
                        if 'H' in g['name']:
                            append_num += 1
                            if 20 < g.Tc <= 30:
                                self.graph_list += [g] * 20
                            elif 30 < g.Tc :
                                self.graph_list += [g] * 50
                    print(f'{datatype} size ', len(self.graph_list), append_num)
                            
                    
            elif '6w' in graph_file_path:
                graph_path = [os.path.join(graph_file_path,name) for name in sorted ([name for name in os.listdir(graph_file_path)])]
                if self.datatype=='train':
                    self.graph_path = graph_path[:60000]
                elif self.datatype=='val':
                    self.graph_path = graph_path[60000:64500]
                elif self.datatype=='test':
                    self.graph_path = graph_path[64500:]
                elif self.datatype=='all':
                    self.graph_path = graph_path
                else:
                    self.graph_path = graph_path[:size]
                print(f'{datatype} size ', len(self.graph_path))
                if pre_load:
                    print('preload')
                    self.graph_list = [load_graph(path=path) for path in self.graph_path]
            elif '38w' in graph_file_path:
                val_ids_path = '/home/xiaoqi/FormationEnergy/data/gnome_data/val_id.txt'
                with open(val_ids_path, 'r') as file:
                    lines = file.readlines()
                    val_ids = [line.rstrip('\n') for line in lines]
                file.close()
                val_ids = set(val_ids)
                graph_file_names = os.listdir(graph_file_path)
                print('total data size ', len(graph_file_names))
                print('val data size ', len(val_ids))
                self.graph_path = [os.path.join(graph_file_path,name) for name in sorted ([name for name in graph_file_names if name.split('_')[0] not in val_ids])]
                print(f'{datatype} size ', len(self.graph_path))
                if pre_load:
                    print('preload')
                    self.graph_list = [load_graph(path=path) for path in self.graph_path]
        elif csv:
            self.cif_str_list = read_csv(file_path=csv_file_path)[1:]
            print(f'{datatype} size ', len(self.cif_str_list))
        
        elif cif:
            self.cif_path_list = [os.path.join(cif_dir, file) for file in os.listdir(cif_dir)]
            print(f'{datatype} size ', len(self.cif_path_list))
        else:
            self.data_list = self.preprocess()
            print(f'{datatype} size ', len(self.data_list))
        
    def preprocess(self):
        # 打开JSON文件
        with open(self.json_file_path, 'r') as file:
            # 读取文件内容
            data = json.load(file)
        if self.datatype=='train':
            return data[:60000]
        elif self.datatype=='val':
            return data[60000:64500]
        elif self.datatype=='test':
            return data[64500:]
        elif self.datatype=='all':
            return data
            

    def len(self):
        if self.graph:
            return len(self.graph_list)
        elif self.csv:
            return len(self.cif_str_list)
        elif self.cif:
            return len(self.cif_path_list)
        else:
            return len(self.data_list)

    def get(self, index):
        if self.graph:
            if self.pre_load:
                return self.graph_list[index]
            return  load_graph(self.graph_path[index])
        elif self.csv:
            cif_str = self.cif_str_list[index][8]
                
            canonical_crystal = self.build_crystal(crystal_str=cif_str)
            frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = self.build_crystal_graph(canonical_crystal)
            edge_type = [0 for i in range(len(edge_indices))]
            complex_graph = HeteroData()# 定义异构图
            complex_graph['name'] = canonical_crystal.formula
            complex_graph['material'].pos = torch.from_numpy(canonical_crystal.cart_coords).float()
            complex_graph['material'].frac_pos = torch.from_numpy(frac_coords).float()
            complex_graph['material', 'material_bond', 'material'].edge_index = torch.tensor(edge_indices.transpose(), dtype=torch.long) 
            complex_graph['material', 'material_bond', 'material'].edge_attr = F.one_hot(torch.tensor(edge_type, dtype=torch.long), num_classes=2).to(torch.float)
            complex_graph['material'].x = self.get_atom_feature(atom_types, lengths, angles, num_atoms)
            complex_graph.formation_energy = float (self.cif_str_list[index][3])
            return complex_graph
        elif self.cif:
            with open(self.cif_path_list[index], "r") as cif_file:
                cif_str = cif_file.read()
            canonical_crystal = self.build_crystal(crystal_str=cif_str)
            frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = self.build_crystal_graph(canonical_crystal)
            edge_type = [0 for i in range(len(edge_indices))]
            complex_graph = HeteroData()# 定义异构图
            complex_graph['name'] = canonical_crystal.formula
            complex_graph['file_name'] = self.cif_path_list[index]
            complex_graph['material'].pos = torch.from_numpy(canonical_crystal.cart_coords).float()
            complex_graph['material'].frac_pos = torch.from_numpy(frac_coords).float()
            complex_graph['material', 'material_bond', 'material'].edge_index = torch.tensor(edge_indices.transpose(), dtype=torch.long) 
            complex_graph['material', 'material_bond', 'material'].edge_attr = F.one_hot(torch.tensor(edge_type, dtype=torch.long), num_classes=2).to(torch.float)
            complex_graph['material'].x = self.get_atom_feature(atom_types, lengths, angles, num_atoms)
            complex_graph.formation_energy = None
            return complex_graph
        
        else:
            try:
                cif_str = self.data_list[index]['structure']
                
                canonical_crystal = self.build_crystal(crystal_str=cif_str)
                frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = self.build_crystal_graph(canonical_crystal)
                edge_type = [0 for i in range(len(edge_indices))]
                complex_graph = HeteroData()# 定义异构图
                complex_graph['name'] = canonical_crystal.formula
                complex_graph['material'].pos = torch.from_numpy(canonical_crystal.cart_coords).float()
                complex_graph['material'].frac_pos = torch.from_numpy(frac_coords).float()
                complex_graph['material', 'material_bond', 'material'].edge_index = torch.tensor(edge_indices.transpose(), dtype=torch.long) 
                complex_graph['material', 'material_bond', 'material'].edge_attr = F.one_hot(torch.tensor(edge_type, dtype=torch.long), num_classes=2).to(torch.float)
                complex_graph['material'].x = self.get_atom_feature(atom_types, lengths, angles, num_atoms)
                complex_graph.formation_energy = self.data_list[index]['formation_energy_per_atom']
                return complex_graph
            except Exception as e:
                print(e)
                return None
    
    def get_atom_feature(self, atom_types, lengths, angles, num_atoms):
        scalars = torch.zeros([num_atoms, 6])
        for i in range(num_atoms):
            scalars[i, 0:3] = torch.from_numpy (lengths)
            scalars[i, 3:6] = torch.from_numpy (angles)
        x = torch.cat([ torch.tensor(atom_types).view([-1,1]), scalars], dim=1)
        return x
    
    def build_crystal_graph(self, crystal, graph_method='crystalnn'):
        
        if graph_method == 'crystalnn':
            try:
                crystal_graph = StructureGraph.with_local_env_strategy(
                    crystal, CrystalNN)
            except:
                crystalNN_tmp = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10)
                crystal_graph = StructureGraph.with_local_env_strategy(
                    crystal, crystalNN_tmp) 
        elif graph_method == 'none':
            pass
        else:
            raise NotImplementedError

        frac_coords = crystal.frac_coords
        atom_types = crystal.atomic_numbers
        lattice_parameters = crystal.lattice.parameters
        lengths = lattice_parameters[:3]
        angles = lattice_parameters[3:]

        assert np.allclose(crystal.lattice.matrix,
                        lattice_params_to_matrix(*lengths, *angles))

        edge_indices, to_jimages = [], []
        if graph_method != 'none':
            for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
                edge_indices.append([j, i])
                to_jimages.append(to_jimage)
                edge_indices.append([i, j])
                to_jimages.append(tuple(-tj for tj in to_jimage))

        atom_types = np.array(atom_types)
        lengths, angles = np.array(lengths), np.array(angles)
        edge_indices = np.array(edge_indices)
        to_jimages = np.array(to_jimages)
        num_atoms = atom_types.shape[0]

        return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms
    
    def build_crystal(self, crystal_str, niggli=True, primitive=False):
        """Build crystal from cif string."""
        crystal = Structure.from_str(crystal_str, fmt='cif')

        if primitive:
            crystal = crystal.get_primitive_structure()

        if niggli:
            crystal = crystal.get_reduced_structure()

        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        # match is gaurantteed because cif only uses lattice params & frac_coords
        # assert canonical_crystal.matches(crystal)
        return canonical_crystal
        

def collate_fn(data_list):
    graph_list = []
    for graph in data_list:
        if graph is not None:
            graph_list.append(graph)
    return graph_list


class DataListLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset,
                 batch_size, shuffle, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)




