from src import train_grnet

config = {
    'experiment_name': 'GRNet_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': True,
    'batch_size': 32,
    'resume_ckpt': None,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'betas': (.9, .999),
    'max_epochs': 250,
    'print_every_n': 10,
    'validate_every_n': 25,
    'milestones':[50],
    'gamma': 0.5
}
train_grnet.main(config)