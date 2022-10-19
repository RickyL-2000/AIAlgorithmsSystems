import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
        x2 = self.norm1(x2)
        x2 = F.relu(x2)
        x2 = x2 + res
        res = x2
        x2 = self.conv2(x2, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
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


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, data):
        x, adj = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        self.out_att.reset_parameters()
