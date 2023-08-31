import os
from joblib import Parallel, delayed
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args

import get_data, strategies
from evaluate import run_strategy
from utils import get_memory
from config import indicator_config

def Bayes(ticker_list, savefile, start_date=None, end_date=None, strategy_name='Indicators', short=True, 
                 n_jobs=5, n_initial=100, n_total=200):

    strategy_list = []

    get_memory.get_memory_usage()

    # create a strategy object for each data set
    print('Importing data...')
    for ticker in ticker_list:
        data = get_data.get_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if strategy_name == 'Indicators':
            strategy_list.append(strategies.Strategy_Indicators(data=data, short=short))
        elif strategy_name == 'CNN':
            strategy_list.append(strategies.Strategy_CNN_ATR(data=data, short=short))

    # Define the bounds for each feature
    print('Creating variables space...')
    learnables = indicator_config.get_learnables(strategy_name=strategy_name)
    space_config = indicator_config.get_space_config()
    space = [space_config[learnable] for learnable in learnables]

    get_memory.get_memory_usage()

    # # make the negative profit function to minimize, the average of all data sets
    # @use_named_args(space)
    # def get_avg_neg_profit(**kwargs):
    #     return -np.mean([strategy.calculate_profit(**kwargs) for strategy in strategy_list])
    

    # make the negative profit function to minimize, the average of all data sets
    @use_named_args(space)
    def get_avg_neg_profit(**kwargs):
        def subjob(strategy):
            return strategy.calculate_profit(**kwargs)
        pool = Parallel(n_jobs = n_jobs, backend = 'loky', verbose = 0, pre_dispatch = 'all')
        result = pool(delayed(subjob)(strategy) for strategy in strategy_list)
        result = np.array(result)
        return -np.mean(result)
    
    # Get current best indicators as initial input
    print('Importing current best result...')
    best_config = indicator_config.get_best_config()
    print(best_config)
    initial_input = [[best_config[i][learnable] for learnable in learnables] for i in range(len(best_config))]

    print("_________Start training process__________")
    # Run Bayesian optimization (high-dimensional version)
    result = gp_minimize(func=get_avg_neg_profit,           # Function to minimize
                        dimensions=space,                   # Bounds for each feature
                        acq_func="EI",                      # Acquisition function: Expected Improvement, method of optimizing
                        x0 = initial_input,                 # Initial point to evaluate
                        n_initial_points=n_initial,         # Number of random points run before optimizing
                        n_calls=n_total,                    # Number of function evaluations
                        n_points=10000,
                        random_state=42,
                        verbose=True,
                        n_jobs=2)

    # Get the optimal parameters and function value
    optimal_params = result.x
    optimal_value = -result.fun

    kwargs = dict(zip([param.name for param in space], optimal_params))

    print("Optimal Parameters:", kwargs)
    print("Optimal Value:", optimal_value)

    # save the optimal parameters to a save file
    with open(savefile, "wb") as file:
        pickle.dump(kwargs, file)

    # print out the final result
    kwargs = dict(zip([param.name for param in space], optimal_params))

    run_strategy(ticker_list=ticker_list, start_date=start_date, end_date=end_date, 
                strategy_name=strategy_name, short=short, n_jobs=n_jobs, **kwargs)
