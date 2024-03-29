import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_name = 'CNN'
    # choose among 'numpy', 'tensorflow', or 'torch'
    config.lib = 'tensorflow'

    # train hyperparameters
    config.max_lr = 6e-4
    config.min_lr = 6e-5
    config.epochs = 200
    config.batch_size = 64
    config.steps_per_epoch = 10
    config.validation_freq = 10
    config.early_stop = False

    config.prior_duration = 90
    config.post_duration = 60

    # neural network hyperparameters
    config._lambda = 0.001
    config.dropout = 0.1

    return config