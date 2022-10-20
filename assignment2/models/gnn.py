import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, TransformerConv

class NaiveGCN(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            dropout
    ):
        super(NaiveGCN, self).__init__()
        self.dropout = dropout

        self.linear1 = torch.nn.Linear(in_channels, 2*hidden_channels)

        self.linear4 = torch.nn.Linear(hidden_channels, 2*hidden_channels)
        self.norm3 = torch.nn.GroupNorm(16, 2*hidden_channels)
        self.linear5 = torch.nn.Linear(2*hidden_channels, hidden_channels)
        self.norm4 = torch.nn.GroupNorm(16, hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = torch.nn.GroupNorm(16, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = torch.nn.GroupNorm(16, hidden_channels)

        self.linear2 = torch.nn.Linear(2*hidden_channels, hidden_channels)
        self.linear3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # pre net
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)

        # split
        x1, x2 = torch.split(x, x.size()[1]//2, dim=1)

        # linear net
        res = x1
        x1 = self.linear4(x1)
        x1 = F.relu(x1)
        x1 = self.norm3(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.linear5(x1)
        x1 = F.relu(x1)
        x1 = self.norm4(x1)
        x1 = x1 + res

        # res gcn
        res = x2
        x2 = self.conv1(x2, edge_index)
        x2 = F.relu(x2)
        x2 = self.norm1(x2)
        x2 = x2 + res
        res = x2
        x2 = self.conv2(x2, edge_index)
        x2 = F.relu(x2)
        x2 = self.norm2(x2)
        x2 = res + x2

        # predict
        x = torch.cat([x1, x2], dim=1)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear3(x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class GraphSAGE(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            aggr
    ):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout

        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = SAGEConv(hidden_channels, hidden_channels, aggr)
        self.norm1 = torch.nn.GroupNorm(4, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr)
        self.norm2 = torch.nn.GroupNorm(4, hidden_channels)
        # self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr, normalize=True)
        # self.norm3 = torch.nn.GroupNorm(16, hidden_channels)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        # self.norm4 = torch.nn.GroupNorm(16, hidden_channels)

        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.linear3 = torch.nn.Linear(hidden_channels//2, out_channels)
        # self.linear4 = torch.nn.Linear(hidden_channels//4, hidden_channels//4)
        # self.linear5 = torch.nn.Linear(hidden_channels//4, out_channels)

    def forward(self, data, node_id):
        x, edge_index = data.x, data.edge_stores[0].edge_index

        # pre net
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)

        # res sage
        res = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        # x = x + res
        # res = x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = x + res
        # res = x
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.norm3(x)
        # # x = x + res
        # # res = x
        # x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # x = self.norm4(x)
        # x = x + res

        # predict
        # res = x
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)
        x = self.linear3(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(x)
        # x = self.linear4(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(x)
        # x = self.linear5(x)
        # x = x + res

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class GAT(nn.Module):
    # FIXME: CUDA OOM
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            heads,
            negative_slope,
            dropout
    ):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads, negative_slope)
        self.norm1 = torch.nn.GroupNorm(16, heads*hidden_channels)
        self.linear3 = torch.nn.Linear(heads*hidden_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads, negative_slope)
        self.norm2 = torch.nn.GroupNorm(16, heads*hidden_channels)
        self.linear4 = torch.nn.Linear(heads*hidden_channels, hidden_channels)

        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # pre net
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)

        # res sage
        res = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = x + res
        res = x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = x + res

        # predict
        x = self.linear2(x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class MixModel(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            heads,
            negative_slope,
            dropout,
            aggr
    ):
        super(MixModel, self).__init__()
        self.dropout = dropout

        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads, negative_slope)
        self.norm1 = torch.nn.GroupNorm(16, heads * hidden_channels)
        self.linear3 = torch.nn.Linear(heads * hidden_channels, 2*hidden_channels)
        self.conv2 = SAGEConv(2*hidden_channels, hidden_channels, aggr)
        self.norm2 = torch.nn.GroupNorm(16, hidden_channels)

        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels//2)
        self.linear4 = torch.nn.Linear(hidden_channels//2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # pre net
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(x)

        # res sage
        res = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.linear3(x)
        x = F.relu(x)
        # x = x + res
        # res = x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = x + res

        # predict
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear4(x)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()