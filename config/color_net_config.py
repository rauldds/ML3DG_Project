config = {
    'experiment_name': 'Color_net_overfitting',
    'device': 'cpu',  # change this to cpu if you do not have a GPU (cuda:0)
    'is_overfit': True,
    'batch_size': 4,
    'resume': False,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 2,
    'print_every_n': 10,
    'validate_every_n': 25,
    'num_workers':2,
    'save_freq': 5
}