config = {
    'experiment_name': 'GRNet_comp_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU (cuda:0)
    'is_overfit': False,
    'batch_size': 50,
    'batch_size_val': 64,
    'resume': False,
    'learning_rate': 0.002,
    'weight_decay': 1e-4,
    'betas': (.9, .999),
    'max_epochs': 300,
    'print_every_n': 10,
    'validate_every_n': 500,
    'milestones':[40, 80, 120, 200],
    'num_workers':8,
    'gamma': 0.5,
    'save_freq': 10
}
completion_layers = ["conv1", "conv2", "conv3", "conv4", "fc5", "fc6",
                     "dconv7","dconv8","dconv9", "dconv10"]
classification_layers = ["gridding_rev", "point_sampling",
                         "feature_sampling", "fc11", "fc12",
                         "fc13", "fc14", "fc15"]