from config import MANet_config, CNN_config

import ml_collections
from models import NN_model
import models.utils as mutils
from get_data import get_data
from utils.get_memory import get_memory_usage

import torch
import logging

def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

### create configurations
config = ml_collections.ConfigDict()
# config.ticker_list = ["META", "TSLA", "GOOGL", "AMZN", "GS", "NVDA", "AMD", "JPM", "NFLX", "MSFT"]
config.ticker_list = ["META"]
config.task = 'train'
config.model = 'MANet'
config.ticker = 'manual'
# data config
config.start_date = '2016-01-01'
config.end_date = '2021-12-31'
config.eval_start_date = '2022-01-01'
config.eval_end_date = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MANet_conf = MANet_config.get_config()
# CNN_conf = CNN_config.get_config()

### make the data
make_X_Y = mutils.make_X_Y('torch')

print(f"Building model for prior duration: {MANet_conf.prior_duration}, post duration: {MANet_conf.post_duration}")
MANet_model = NN_model.NN_model(MANet_conf)
# CNN_model = NN_model.NN_model(CNN_conf)
lr = MANet_conf.max_lr

MANet_model.model.to(device)

for ticker in config.ticker_list:
    # # update learning rate
    # for param_group in MANet_model.optimizer.param_groups:
    #     param_group['lr'] = lr
    # lr /= (MANet_conf.max_lr/MANet_conf.min_lr) ** (1/(len(config.ticker_list)-1))

    print(f"Start training for {ticker}")
    data_train = get_data(ticker, start_date=config.start_date, end_date=config.end_date).astype('float32').values
    data_test = get_data(ticker, start_date=config.eval_start_date, end_date=config.eval_end_date).astype('float32').values
    
    X_train, Y_train = make_X_Y(data_train, prior_duration=MANet_conf.prior_duration, post_duration=MANet_conf.post_duration)
    del data_train
    X_test, Y_test = make_X_Y(data_test, prior_duration=MANet_conf.prior_duration, post_duration=MANet_conf.post_duration)
    del data_test

    X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    get_memory_usage()

    MANet_model.train(X_train, Y_train, X_test, Y_test, 
            epochs=MANet_conf.epochs, batch_size=MANet_conf.batch_size, 
            validation_freq=MANet_conf.validation_freq, early_stop=MANet_conf.early_stop)
    
    del X_train, Y_train, X_test, Y_test