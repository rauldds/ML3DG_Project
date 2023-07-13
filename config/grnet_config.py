config = {
    'experiment_name': 'GRNet_comp_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU (cuda:0)
    'is_overfit': False,
    'batch_size': 50,
    'batch_size_val': 64,
    'resume': False,
    'learning_rate': 2e-3,
    'weight_decay': 1e-4,
    'betas': (.9, .999),
    'max_epochs': 50,
    'print_every_n': 5,
    'validate_every_n': 1000,
    'milestones':[50],
    'num_workers':8,
    'gamma': 0.5,
    'save_freq': 5
}
completion_layers = ["conv1", "conv2", "conv3", "conv4", "fc5", "fc6",
                     "dconv7","dconv8","dconv9", "dconv10"]
classification_layers = ["gridding_rev", "point_sampling",
                         "feature_sampling", "fc11", "fc12",
                         "fc13", "fc14", "fc15"]