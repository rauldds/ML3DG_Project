config = {
    'experiment_name': 'GRNet_overfitting',
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU
    'is_overfit': True,
    'batch_size': 8,
    'resume_ckpt': None,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'betas': (.9, .999),
    'max_epochs': 10,
    'print_every_n': 2,
    'validate_every_n': 25,
    'milestones':[50],
    'num_workers':2,
    'gamma': 0.5,
    'save_freq': 100
}
completion_layers = ["gridding", "conv1", "conv2", "conv3", "conv4", "fc5", "fc6",
                     "dconv7","dconv8","dconv9", "dconv10", "gridding_rev"]
classification_layers = ["point_sampling", "feature_sampling", "fc11", "fc12", "fc13", "fc14", "fc15"]