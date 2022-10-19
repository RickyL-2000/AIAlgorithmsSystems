# %%
from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from utils.utils import *

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToSparseTensor
import os

from hparams import hparams

import models
from models.mlp import MLP

#设置gpu设备
device = hparams['device']
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

if __name__ == '__main__':
    pass

# %%
model_name = hparams["model_name"]
exp_name = hparams["exp_name"]

# %%
path = './datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir = f'./results/{model_name}_{exp_name}/' #模型保存路径
mkdir(save_dir)
dataset_name = 'DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
# data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图
data.edge_index = data.edge_stores[0].edge_index

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']
# result_dir = prepare_folder(dataset_name,'mlp')

to_sparse_tensor = ToSparseTensor()
train_data = to_sparse_tensor(data.subgraph(train_idx))
train_data.edge_index = train_data.edge_stores[0].edge_index
valid_data = to_sparse_tensor(data.subgraph(split_idx['valid']))
valid_data.edge_index = valid_data.edge_stores[0].edge_index
test_data = to_sparse_tensor(data.subgraph(split_idx['test']))
test_data.edge_index = test_data.edge_stores[0].edge_index

# %%
para_dict = hparams[model_name]
# model_para = para_dict.copy()
# model_para.pop('lr')
# model_para.pop('weight_decay')
# model = MLP(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)
# print(f'Model MLP initialized')
model = models.build_model(model_name, hparams[model_name], device)

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

# %%
def train(model, data, train_idx, optimizer):
    # data.y is labels of shape (N, )
    model.train()

    optimizer.zero_grad()

    out = model(data)

    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, train_data, valid_data, split_idx, evaluator):
    # data.y is labels of shape (N, )
    with torch.no_grad():
        model.eval()

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            # node_id = split_idx[key]
            data = train_data if key == "train" else valid_data

            out = model(data)
            y_pred = out.exp()  # (N,num_classes)

            losses[key] = F.nll_loss(out, data.y).cpu().item()
            eval_results[key] = evaluator.eval(data.y, y_pred)[eval_metric]

    return eval_results, losses, y_pred

# %%
print(sum(p.numel() for p in model.parameters()))  # 模型总参数量

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['weight_decay'])
best_valid = 0
min_valid_loss = 1e8
train_data.to(device)
valid_data.to(device)

# 保存代码
mkdir(f"{save_dir}/code")
exe_cmd(f"cp -r models {save_dir}/code")
exe_cmd(f"cp hparams.py {save_dir}/code")
exe_cmd(f"cp train.py {save_dir}/code")

for epoch in range(1, hparams["epochs"] + 1):
    model.train()
    loss = train(model, train_data, train_idx, optimizer)
    model.eval()
    eval_results, losses, out = test(model, train_data, valid_data, split_idx, evaluator)
    train_eval, valid_eval = eval_results['train'], eval_results['valid']
    train_loss, valid_loss = losses['train'], losses['valid']

    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir + '/model.pt')  # 将表现最好的模型保存

    if epoch % hparams["log_steps"] == 0:   # 我们将AUC值乘上100，使其在0-100的区间内
        cur_log = f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_eval:.3f}, Valid: {100 * valid_eval:.3f}'
        print(cur_log)
        with open(f"{save_dir}/log.txt", "a") as f:
            f.write(cur_log + "\n")

# %%
# model.load_state_dict(torch.load(save_dir + '/model.pt'))  # 载入验证集上表现最好的模型
# def predict(data, node_id):
#     """
#     加载模型和模型预测
#     :param node_id: int, 需要进行预测节点的下标
#     :return: tensor, 类0以及类1的概率, torch.size[1,2]
#     """
#     # -------------------------- 实现模型预测部分的代码 ---------------------------
#     with torch.no_grad():
#         model.eval()
#         out = model(data.x[node_id])
#         y_pred = out.exp()  # (N,num_classes)
#
#     return y_pred

# %%
# dic={0:"正常用户",1:"欺诈用户"}
# node_idx = 0
# y_pred = predict(data, node_idx)
# print(y_pred)
# print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
#
# node_idx = 1
# y_pred = predict(data, node_idx)
# print(y_pred)
# print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}。')
