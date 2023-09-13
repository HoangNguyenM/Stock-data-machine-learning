import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_name = 'WMANN'
    # choose among 'numpy', 'tensorflow', or 'torch'
    config.lib = 'torch'

    # train hyperparameters
    config.max_lr = 1e-3
    config.min_lr = 1e-5
    config.epochs = 10
    config.batch_size = 2048
    config.steps_per_epoch = 10
    config.validation_freq = 10
    config.early_stop = False

    config.prior_duration = 90
    config.post_duration = 60

    # neural network hyperparameters
    config.ma_output_size = 15
    config.stride_size = 1
    config.nnodes1 = 64
    config.nnodes2 = 64
    config.kernel_list = [2, 6, 3, 9, 5, 15, 8, 24, 12, 36, 18, 54, 27, 75]
    config.ma_combine = 2

    config._lambda = 1e-4
    config.dropout = 0.1

    return config