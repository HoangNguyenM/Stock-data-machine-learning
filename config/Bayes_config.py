import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.n_initial = 300
    config.n_total = 500
    
    return config