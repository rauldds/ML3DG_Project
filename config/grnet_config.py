config = {
    'device': 'cuda:0',  # change this to cpu if you do not have a GPU (cuda:0)
    'is_overfit': False,
    'num_workers': 8,
    'save_freq': 10,
    'batch_size': 54,
    'batch_size_val': 54,
    'max_epochs': 80,
    'validate_every_n': 500,
    'learning_rate': 0.00001,
    'weight_decay': 1e-4,
    'betas': (.9, .999),
    'milestones':[100],
    'gamma': 0.95,
    'learning_rate_loss_weights': 0.0001,
    "cls_net":{
        'learning_rate': 0.00012,
        'weight_decay': 1e-4,
    }
}
completion_layers = ["conv1", "conv2", "conv3", "conv4", "fc5", "fc6",
                     "dconv7","dconv8","dconv9", "dconv10"]
classification_layers = ["gridding_rev", "point_sampling",
                         "feature_sampling", "fc11", "fc12",
                         "fc13", "fc14", "fc15"]