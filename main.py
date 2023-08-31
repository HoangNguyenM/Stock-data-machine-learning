import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time

from evaluate import run_strategy
from Bayesian_optim import Bayes
from utils import get_symbols, download_data
from config import indicator_config
import configargparse

if __name__ == "__main__":
    p = configargparse.ArgParser()

    p.add('--task', choices=['download', 'run', 'Bayes'])

    p.add('--strategy', choices=['Indicators', 'CNN'], default='Indicators')
    p.add('--short', type=bool, default=False)
    p.add('--start_date', default=None)
    p.add('--end_date', default=None)

    p.add('--n_jobs', type=int, default=-1)
    p.add('--agent_num', type=int, default=0)

    config = p.parse_args()

    # CHANGE TASK HERE
    task = 'download'

    SP500 = get_symbols.get_SP500_symbols()
    SP100 = get_symbols.get_SP100_symbols()
    #ticker_list = ["META", "TSLA", "GOOGL", "AMZN", "GS", "NVDA", "AMD", "JPM", "NFLX", "AAPL"]
    ticker_list = SP500
    ind = ticker_list.index('CMI')
    ticker_list = ticker_list[ind+1:]
    print(ticker_list)

    # change strategy_name to make different strategies
    strategy_name = 'Indicators'
    short = False
    n_jobs = 5

    start_date = None
    end_date = '2019-01-01'
    
    if short:
        savefile = f"config/{strategy_name}_full_{config.agent_num}.pkl"
    else:
        savefile = f"config/{strategy_name}_long_{config.agent_num}.pkl"

    ### download/update the data for all tickers ###
    if task == 'download':
        for ticker in ticker_list:
            #if ticker not in SP100:
            download_data.download_data(ticker=ticker)

    ### run Bayesian optimization to find the best indicators ###
    elif task == 'Bayes':
        Bayes(ticker_list=ticker_list, start_date=start_date, end_date=end_date, 
            strategy_name=strategy_name, short=short, n_jobs=n_jobs,
            n_initial=20, n_total=30, savefile=savefile)
        
    ### run/test the strategy ###
    elif task == 'evaluate':
        # get the best config and run the strategy
        learnables = indicator_config.get_learnables(strategy_name=strategy_name)
        best_config = indicator_config.get_best_config()
        
        for sub_config in best_config:
            print(f"Config to evaluate: {sub_config}")
            kwargs = {}
            for learnable in learnables:
                kwargs[learnable] = sub_config[learnable]

            start_time = time.time()
            run_strategy(ticker_list=ticker_list, start_date=start_date, end_date=end_date, 
                        strategy_name=strategy_name, short=short, n_jobs=n_jobs, **kwargs)
            end_time = time.time()
            print(f"The process took {end_time-start_time:.2f} seconds to run")