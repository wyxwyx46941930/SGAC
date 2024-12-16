import pickle
import torch
from torch_geometric.data import Data, InMemoryDataset
import random


def convert_to_tudata(protein_name, protein_data):
    node_features = torch.tensor(protein_data['node_features'], dtype=torch.float)
    edge_index = torch.tensor(protein_data['graph'], dtype=torch.long)
    label = torch.tensor([protein_data['label']], dtype=torch.long)
    
    data = Data(x=node_features, edge_index=edge_index, y=label)
    data.name = protein_name  
    return data

class ProteinTUDataset(InMemoryDataset):
    def __init__(self, root, data_list):
        self.data_list = data_list
        super().__init__(root, transform=None, pre_transform=None)
        self.data, self.slices = self.collate(data_list)

    @property
    def processed_file_names(self):
        return []

    def process(self):
        pass  

    def get_protein_names(self):
        return [data.name for data in self.data_list]