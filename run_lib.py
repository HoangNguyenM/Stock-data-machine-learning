import time
import numpy as np
import tensorflow as tf
from keras import backend as K

from models import NN_model
import models.utils as mutils
from Bayesian_optim import Bayes

from config import indicator_config
from get_data import get_data
from evaluate import run_strategy
from utils.download_data import download_data
from utils.get_memory import get_memory_usage

def download(config):
    ### download/update the data for all tickers ###
    for ticker in config.ticker_list:
        #if ticker not in SP100:
        download_data(ticker=ticker)

def train(config, model_config):
    ### train the model ###
    if config.model == 'Bayes':
        if config.short:
            savefile = f"checkpoints/{config.strategy}_full_{config.agent_num}.pkl"
        else:
            savefile = f"checkpoints/{config.strategy}_long_{config.agent_num}.pkl"
        ### run Bayesian optimization to find the best indicators ###
        Bayes(ticker_list=config.ticker_list, start_date=config.start_date, end_date=config.end_date, 
            strategy_name=config.strategy, short=config.short, n_jobs=config.n_jobs,
            n_initial=model_config.n_initial, n_total=model_config.n_total, savefile=savefile)
        
    elif config.model == 'WMANN' or config.model == 'CNN':
        # get functions list for lib used: numpy, tensorflow or torch
        make_X_Y = mutils.make_X_Y(model_config.lib)
        print(f"Building model for prior duration: {model_config.prior_duration}, post duration: {model_config.post_duration}")
        train_model = NN_model.NN_model(model_config)
        lr = model_config.max_lr

        for ticker in config.ticker_list:
            # update learning rate
            if model_config.lib == 'tensorflow' or model_config.lib == 'numpy': 
                K.set_value(train_model.model.optimizer.learning_rate, lr)
            elif model_config.lib == 'torch':
                for param_group in train_model.optimizer.param_groups:
                    param_group['lr'] = lr
            lr /= (model_config.max_lr/model_config.min_lr) ** (1/(len(config.ticker_list)-1))

            print(f"Start training for {ticker}")
            data_train = get_data(ticker, start_date=config.start_date, end_date=config.end_date).astype('float32').values
            data_test = get_data(ticker, start_date=config.eval_start_date, end_date=config.eval_end_date).astype('float32').values
            
            X_train, Y_train = make_X_Y(data_train, prior_duration=model_config.prior_duration, post_duration=model_config.post_duration)
            del data_train
            X_test, Y_test = make_X_Y(data_test, prior_duration=model_config.prior_duration, post_duration=model_config.post_duration)
            del data_test

            print(X_train.shape, Y_train.shape)
            print(X_test.shape, Y_test.shape)
            get_memory_usage()

            train_model.train(X_train, Y_train, X_test, Y_test, 
                    epochs=model_config.epochs, batch_size=model_config.batch_size, steps_per_epoch=model_config.steps_per_epoch, 
                    validation_freq=model_config.validation_freq, early_stop=model_config.early_stop)
            
            del X_train, Y_train, X_test, Y_test

            if model_config.lib == 'tensorflow':
                tf.keras.backend.clear_session()

def evaluate(config, model_config):
    ### run/test the strategy ###
    if config.model == 'Bayes':
        # get the best config and run the strategy
        learnables = indicator_config.get_learnables(strategy_name=config.strategy)
        best_config = indicator_config.get_best_config()
        
        for sub_config in best_config:
            print(f"Config to evaluate: {sub_config}")
            kwargs = {}
            for learnable in learnables:
                kwargs[learnable] = sub_config[learnable]

            start_time = time.time()
            run_strategy(ticker_list=config.ticker_list, start_date=config.start_date, end_date=config.end_date, 
                        strategy_name=config.strategy, short=config.short, n_jobs=config.n_jobs, **kwargs)
            end_time = time.time()
            print(f"The process took {end_time-start_time:.2f} seconds to run")

    elif config.model == 'WMANN' or config.model == 'CNN':
        # get functions list for lib used: numpy, tensorflow or torch
        make_X_Y = mutils.make_X_Y(model_config.lib)
        print(f"Building model for prior duration: {model_config.prior_duration}, post duration: {model_config.post_duration}")
        test_model = NN_model.NN_model(model_config)

        for ticker in config.ticker_list:
            print(f"Start evaluating for {ticker}")
            data_test = get_data(ticker, start_date=config.eval_start_date, end_date=config.eval_end_date).astype('float32').values

            X_test, Y_test = make_X_Y(data_test, prior_duration=model_config.prior_duration, post_duration=model_config.post_duration)
            del data_test

            print(X_test.shape, Y_test.shape)
            get_memory_usage()

            test_model.evaluate(X_test=X_test, Y_test=Y_test, batch_size=model_config.batch_size)
            del X_test, Y_test
            if model_config.lib == 'tensorflow':
                tf.keras.backend.clear_session()