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
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图
# data.edge_index = data.edge_stores[0].edge_index

if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']
# result_dir = prepare_folder(dataset_name,'mlp')

# %%
para_dict = hparams[model_name]

model = models.build_model(model_name, hparams[model_name], device)

eval_metric = 'auc'  #使用AUC衡量指标
evaluator = Evaluator(eval_metric)

# %%
model.load_state_dict(torch.load(save_dir + '/model.pt'))  # 载入验证集上表现最好的模型
def predict(data, node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    data.to(device)
    with torch.no_grad():
        model.eval()
        out = model(data, node_id)
        y_pred = out.exp()  # (N,num_classes)

    return y_pred.cpu().numpy()

# %%
y_pred = predict(data, split_idx['test'])
# np.save(f"{save_dir}/output.npy", y_pred)

# %%
y_pred = np.load(f"{save_dir}/output.npy")

# %%
y_pred = y_pred[split_idx['test']]
