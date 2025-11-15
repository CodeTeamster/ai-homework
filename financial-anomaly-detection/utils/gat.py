from torch_geometric.nn import GATConv


import torch.nn as nn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, n_heads=1, dropout=0.5):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=n_heads, dropout=dropout))
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_channels * n_heads, hidden_channels, heads=n_heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * n_heads, out_channels, heads=n_heads, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

