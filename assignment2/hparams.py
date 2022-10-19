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
        'lr': 0.01,
        'n_feat': 20,
        'n_hid': 32,
        'n_class': 2,
        'dropout': 0.2,
        'alpha': 0.01,
        'n_heads': 2,
        'weight_decay': 5e-7
    },
    "epochs": 800,
    "log_steps": 10,
    "model_name": "naivegnn",
    "exp_name": 14,
    "device": 2
}
