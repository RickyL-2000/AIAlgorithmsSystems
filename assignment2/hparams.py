hparams = {
    "mlp": {
        'lr': 0.01,
        'num_layers': 2,
        'in_channels': 20,
        'hidden_channels': 128,
        'out_channels': 2,
        'dropout': 0.0,
        'batchnorm': False,
        'weight_decay': 5e-7
    },
    "naivegnn": {
        'lr': 0.001,
        'in_channels': 20,
        'hidden_channels': 64,
        'out_channels': 2,
        'dropout': 0.5,
        'weight_decay': 5e-7
    },
    "gat": {
        'lr': 0.001,
        'in_channels': 20,
        'hidden_channels': 64,
        'out_channels': 2,
        'heads': 2,
        'negative_slope': 0.1,
        'dropout': 0.5,
        'weight_decay': 5e-7
    },
    "sage": {
        'lr': 0.002,
        'in_channels': 20,
        'hidden_channels': 32,
        'out_channels': 2,
        'dropout': 0.5,
        'weight_decay': 5e-7,
        'aggr': 'mean'
    },
    "mix": {
        'lr': 0.005,
        'in_channels': 20,
        'hidden_channels': 64,
        'out_channels': 2,
        'heads': 2,
        'negative_slope': 0.1,
        'dropout': 0.5,
        'weight_decay': 5e-7,
        'aggr': 'mean'
    },
    "epochs": 1600,
    "log_steps": 10,
    "model_name": "sage",
    "exp_name": 13,
    "device": 2
}
