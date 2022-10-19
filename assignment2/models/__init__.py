from .gnn import *
from .mlp import *

def build_model(model_name, hparams, device):
    model_para = hparams.copy()
    if model_name == "mlp":
        model_para.pop('lr')
        model_para.pop('weight_decay')
        model = MLP(**model_para).to(device)
        print(f'Model MLP initialized')
        return model
    elif model_name == "naivegnn":
        model_para.pop('lr')
        model_para.pop('weight_decay')
        model = NaiveGCN(**model_para).to(device)
        print(f'Model NaiveGCN initialized')
        return model
    elif model_name == "gat":
        model_para.pop('lr')
        model_para.pop('weight_decay')
        model = GAT(**model_para).to(device)
        print(f'Model GAT initialized')
        return model
