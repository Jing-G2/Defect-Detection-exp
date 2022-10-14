import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, num_feature, num_label):
        super(GAT, self).__init__()
        self.GAT1 = GATConv(num_feature, 16, heads=16, concat=True, dropout=0.6)
        self.GAT2 = GATConv(256, 256, dropout=0.6)
        self.lin = Linear(256, num_label)

    def forward(self, x, edge_index, batch):
        x = self.GAT1(x, edge_index)
        x = F.relu(x)
        x = self.GAT2(x, edge_index)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x


# class GAT(torch.nn.Module):

#     def __init__(self, in_channels, out_channels, hidden_channels=256):
#         super(GAT, self).__init__()

#         self.num_layers = 2

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GATConv(in_channels, hidden_channels, dropout=0.6))
#         self.convs.append(GATConv(hidden_channels, out_channels, dropout=0.6))

#     def forward(self, x, adjs):
#         # `train_loader` computes the k-hop neighborhood of a batch of nodes,
#         # and returns, for each layer, a bipartite graph object, holding the
#         # bipartite edges `edge_index`, the index `e_id` of the original edges,
#         # and the size/shape `size` of the bipartite graph.
#         # Target nodes are also included in the source nodes so that one can
#         # easily apply skip-connections or add self-loops.
#         for i, (edge_index, _, size) in enumerate(adjs):
#             x_target = x[:size[1]]  # Target nodes are always placed first.
#             x = self.convs[i]((x, x_target), edge_index)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x.log_softmax(dim=-1)

#     def inference(self, x_all, subgraph_loader, device):
#         # pbar = tqdm(total=x_all.size(0) * self.num_layers)
#         # pbar.set_description('Evaluating')

#         # Compute representations of nodes layer by layer, using *all*
#         # available edges. This leads to faster computation in contrast to
#         # immediately computing the final representations of each batch.
#         for i in range(self.num_layers):
#             xs = []
#             for batch_size, n_id, adj in subgraph_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 x_target = x[:size[1]]
#                 x = self.convs[i]((x, x_target), edge_index)
#                 if i != self.num_layers - 1:
#                     x = F.relu(x)
#                 xs.append(x.cpu())

#                 # pbar.update(batch_size)

#             x_all = torch.cat(xs, dim=0)

#         # pbar.close()

#         return x_all