import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # choose among 'numpy', 'tensorflow', or 'torch'
    config.lib = 'torch'

    config.epochs = 10
    config.batch_size = 64
    config.steps_per_epoch = 1000
    config.validation_freq = 10
    config.early_stop = False

    #config.prior_duration = [60, 90, 120, 180]
    #config.post_duration = [30, 45, 60]
    config.prior_duration = [120]
    config.post_duration = [60]

    return config