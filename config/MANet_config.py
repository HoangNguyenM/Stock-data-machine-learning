import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_name = 'MANet'
    # choose among 'numpy', 'tensorflow', or 'torch'
    config.lib = 'torch'

    # train hyperparameters
    config.max_lr = 6e-4
    config.min_lr = 6e-5
    config.epochs = 10
    config.batch_size = 1024
    config.validation_freq = 1
    config.early_stop = False

    ### neural network hyperparameters
    # moving average sizes
    config.min_kernel = 2
    config.max_kernel = 64
    config.dist = 2
    config.n_ma = (config.max_kernel - config.min_kernel) // config.dist + 1

    # transformer
    config.stride = 2
    config.block_size = 32
    config.n_head = 5
    config.n_layer = 4
    config.in_channels = 5 * config.block_size

    config.bias = True
    config._lambda = 1e-4
    config.dropout = 0.1

    # data hyperparameters
    config.prior_duration = (config.max_kernel + config.stride * (config.block_size - 1))
    config.post_duration = 60

    return config