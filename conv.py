import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch import nn
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = torch.nn.Linear(4, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        else:
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=None))
        return out

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(3, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_attr is not None:
            return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)
        else:
            return self.propagate(edge_index, x=x, edge_attr=None, norm=norm) + F.relu(
                x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr is None:
            return norm.view(-1, 1) * F.relu(x_j)
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class SAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.neigh_linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_encoder = torch.nn.Linear(3, emb_dim)  # Assuming edge_attr has dimension 3
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Linearly transform node features
        x = self.linear(x)

        # Encode edge attributes if available
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)

        # Propagate messages
        if edge_attr is not None:
            out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        else:
            out = self.propagate(edge_index, x=x)

        # Apply linear transformation to the concatenated embeddings
        out = self.neigh_linear(out) + F.relu(x + self.root_emb.weight)
        
        return out

    def message(self, x_j, edge_attr=None):
        # Apply edge embedding if edge_attr is provided
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        return F.relu(x_j)

    def update(self, aggr_out):
        # Apply the neighbor linear transformation and return
        return aggr_out

class GNN_node(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.0, residual = False, gnn_type = 'gin', feat_dim = None):

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.residual = residual
        self.fc = nn.Linear(feat_dim, emb_dim, bias=False)
        self.node_encoder = torch.nn.Embedding(feat_dim, emb_dim) 
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index  = batched_data.x, batched_data.edge_index
        tmp = self.fc(x)

        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr=None)
            h = self.batch_norms[layer](h)


            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]
            h_list.append(h)
        
        node_representation = h_list[-1]
        vice_node_representation = h_list[1]

        return node_representation, vice_node_representation