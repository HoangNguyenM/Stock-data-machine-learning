import time
import numpy as np
import tensorflow as tf
import torch
import itertools
import ml_collections

from models import CNN, CNN_tf, CNN_torch
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
        
    elif config.model == 'CNN':
        # get functions list for lib used: numpy, tensorflow or torch
        func_dict = get_func_dict(lib=model_config.lib)

        for ticker in config.ticker_list:
            print(f"Start training for {ticker}")
            data_train = func_dict.get_tensor(get_data(ticker, start_date=config.start_date, end_date=config.end_date).values, dtype=func_dict.data_type)
            data_test = func_dict.get_tensor(get_data(ticker, start_date=config.eval_start_date, end_date=config.eval_end_date).values, dtype=func_dict.data_type)

            for prior, post in itertools.product(model_config.prior_duration, model_config.post_duration):
                print(f"Building model for prior duration: {prior}, post duration: {post}")
                X_train, Y_train, _, _ = func_dict.make_X_y(data_train, prior_duration=prior, post_duration=post)
                X_test, Y_test, _, _ = func_dict.make_X_y(data_test, prior_duration=prior, post_duration=post)

                print(X_train.shape, Y_train.shape)
                print(X_test.shape, Y_test.shape)
                get_memory_usage()

                CNN_model = func_dict.get_model(prior_duration=prior, post_duration=post)
                CNN_model.train(X_train, Y_train, X_test, Y_test, 
                        epochs=model_config.epochs, batch_size=model_config.batch_size, steps_per_epoch=model_config.steps_per_epoch, 
                        validation_freq=model_config.validation_freq, early_stop=model_config.early_stop)


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

    elif config.model == 'CNN':
        # get functions list for lib used: numpy, tensorflow or torch
        func_dict = get_func_dict(lib=model_config.lib)

        for ticker in config.ticker_list:
            print(f"Start evaluating for {ticker}")
            data_test = func_dict.get_tensor(get_data(ticker, start_date=config.eval_start_date, end_date=config.eval_end_date).values, dtype=func_dict.data_type)

            for prior, post in itertools.product(model_config.prior_duration, model_config.post_duration):
                print(f"Building model for prior duration: {prior}, post duration: {post}")
                X_test, Y_test, _, _ = func_dict.make_X_y(data_test, prior_duration=prior, post_duration=post)

                print(X_test.shape, Y_test.shape)
                get_memory_usage()

                CNN_model = func_dict.get_model(prior_duration=prior, post_duration=post)
                CNN_model.evaluate(X_test=X_test, Y_test=Y_test, batch_size=model_config.batch_size)

def get_func_dict(lib):
    func_dict = ml_collections.ConfigDict()

    # get functions based on lib used
    if lib == 'numpy':
        func_dict.get_tensor = np.array
        func_dict.data_type = np.float32
        func_dict.make_X_y = CNN.make_X_y
        func_dict.get_model = CNN.CNN
    elif lib == 'tensorflow':
        func_dict.get_tensor = tf.convert_to_tensor
        func_dict.data_type = tf.float32
        func_dict.make_X_y = CNN_tf.make_X_y
        func_dict.get_model = CNN.CNN
    elif lib == 'torch':
        func_dict.get_tensor = torch.tensor
        func_dict.data_type = torch.float32
        func_dict.make_X_y = CNN_torch.make_X_y
        func_dict.get_model = CNN_torch.CNN

    return func_dict