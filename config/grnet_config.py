config = {
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU (cuda:0)
    'is_overfit': False,
    'num_workers': 3,
    'save_freq': 10,
    'batch_size': 24,
    'batch_size_val': 24,
    'max_epochs': 21,
    'validate_every_n': 430,
    'learning_rate': 0.002,
    'weight_decay': 1e-4,
    'betas': (.9, .999),
    'milestones':[40, 80, 120, 200],
    'gamma': 0.5,
    "cls_net":{
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
    }
}
completion_layers = ["conv1", "conv2", "conv3", "conv4", "fc5", "fc6",
                     "dconv7","dconv8","dconv9", "dconv10"]
classification_layers = ["gridding_rev", "point_sampling",
                         "feature_sampling", "fc11", "fc12",
                         "fc13", "fc14", "fc15"]