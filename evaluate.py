import numpy as np
import get_data, strategies
from utils import get_memory
from joblib import Parallel, delayed

def run_strategy(ticker_list, start_date=None, end_date=None, strategy_name='Indicators', short=True, n_jobs=5, **kwargs):
    
    strategy_list = []

    get_memory.get_memory_usage()

    # create a strategy object for each data set
    for ticker in ticker_list:
        data = get_data.get_data(ticker=ticker, start_date=start_date, end_date=end_date)
        if strategy_name == 'Indicators':
            strategy_list.append(strategies.Strategy_Indicators(data=data, short=short))
        elif strategy_name == 'CNN':
            strategy_list.append(strategies.Strategy_CNN_ATR(data=data, short=short))
    
    get_memory.get_memory_usage()

    starting_balance = 1
    print(f"Starting balance: {starting_balance}")
    
    def get_profit(strategy):
        return strategy.calculate_profit_w_verbal(starting_balance=starting_balance, **kwargs)
    
    pool = Parallel(n_jobs = n_jobs, backend = 'loky', verbose = 51, pre_dispatch = 'all')
    result = pool(delayed(get_profit)(strategy) for strategy in strategy_list)
    result = np.array(result)

    profit = np.mean(result[:, 0])
    num_long = np.mean(result[:, 1])
    len_long = np.mean(result[:, 2])
    num_short = np.mean(result[:, 3])
    len_short = np.mean(result[:, 4])
    profit_long = np.mean(result[:, 5])
    profit_short = np.mean(result[:, 6])
    mkt_move = np.mean(result[:, 7])

    get_memory.get_memory_usage()
    
    print(f"Number of long trades: {num_long}")
    print(f"Avg length of long trade: {len_long}")
    print(f"Number of short trades: {num_short}")
    print(f"Avg length of short trade: {len_short}")
    print(f"Profit ratio from long: {profit_long}")
    print(f"Profit ratio from short: {profit_short}")
    print(f"Final balance: {profit}")
    print(f"Avg market movement: {mkt_move}")