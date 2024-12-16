import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from conv import GNN_node

class GNN(torch.nn.Module):

    def __init__(self, num_layer = 2, emb_dim = 300,
                    gnn_type = 'gin', residual = False, drop_ratio = 0.5, graph_pooling = "sum", feat_dim = None):
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        self.gnn_node = GNN_node(num_layer, emb_dim, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, feat_dim=feat_dim)

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        
    def forward(self, batched_data):
        h_node, vice_node_representation = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        vice_h_graph = self.pool(vice_node_representation, batched_data.batch)

        return h_graph, vice_h_graph