import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x = data.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

# 2946
# Epoch: 10, Loss: 0.0819, Train: 64.629, Valid: 64.883
# Epoch: 20, Loss: 0.1008, Train: 66.707, Valid: 66.826
# Epoch: 30, Loss: 0.0783, Train: 69.494, Valid: 68.852
# Epoch: 40, Loss: 0.0665, Train: 70.226, Valid: 69.722
# Epoch: 50, Loss: 0.0658, Train: 67.379, Valid: 67.184
# Epoch: 60, Loss: 0.0649, Train: 70.504, Valid: 69.947
# Epoch: 70, Loss: 0.0647, Train: 70.460, Valid: 69.957
# Epoch: 80, Loss: 0.0644, Train: 70.789, Valid: 70.187
# Epoch: 90, Loss: 0.0643, Train: 70.789, Valid: 70.221
# Epoch: 100, Loss: 0.0642, Train: 71.114, Valid: 70.456
# Epoch: 110, Loss: 0.0641, Train: 71.223, Valid: 70.553
# Epoch: 120, Loss: 0.0641, Train: 71.348, Valid: 70.648
# Epoch: 130, Loss: 0.0640, Train: 71.448, Valid: 70.729
# Epoch: 140, Loss: 0.0640, Train: 71.540, Valid: 70.795
# Epoch: 150, Loss: 0.0640, Train: 71.619, Valid: 70.856
# Epoch: 160, Loss: 0.0639, Train: 71.692, Valid: 70.913
# Epoch: 170, Loss: 0.0639, Train: 71.758, Valid: 70.966
# Epoch: 180, Loss: 0.0639, Train: 71.817, Valid: 71.014
# Epoch: 190, Loss: 0.0638, Train: 71.872, Valid: 71.054
# Epoch: 200, Loss: 0.0638, Train: 71.920, Valid: 71.089
