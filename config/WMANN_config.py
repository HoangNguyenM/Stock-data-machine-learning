import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_name = 'WMANN'
    # choose among 'numpy', 'tensorflow', or 'torch'
    config.lib = 'tensorflow'

    # train hyperparameters
    config.epochs = 100
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
    config.kernel_list = [3, 10, 5, 15, 7, 20, 10, 30, 15, 45, 25, 75, 40]
    config.ma_combine = 2

    config._lambda = 0.001
    config.dropout = 0.1

    return config