import configargparse
import run_lib
from utils.get_symbols import get_SP20_symbols, get_SP100_symbols, get_SP500_symbols
from config import Bayes_config

### This file includes the code for Bayesian optimization
# This algorithm is a black box search for the best strategies depending on the indicators given to the model
# Details about the strategy can be found in strategies.py, where the model use indicators such as RSI, MACD, EMA and search for the best way to use them in a given space

def run_main(config):
    ### Use the following lines to change configs for convenience, configs can also be modified in "config" folder or by command line ###
    # general config
    config.task = 'train'
    config.strategy = 'Indicators'
    config.ticker = 'manual'
    # data config
    config.short = False
    config.start_date = '2016-01-01'
    config.end_date = '2021-12-31'
    config.eval_start_date = '2022-01-01'
    config.eval_end_date = None
    # technical config
    config.n_jobs = 5

    model_config = Bayes_config.get_config()
    config.ticker_list = get_ticker(type = config.ticker)
    print(config.ticker_list)

    if config.task == 'train':
        run_lib.Bayes_train(config, model_config)
    elif config.task == 'evaluate':
        run_lib.Bayes_eval(config, model_config)

def get_ticker(type):
    if type == 'SP20':
        return get_SP20_symbols()
    elif type == 'SP100':
        return get_SP100_symbols()
    elif type == 'SP500':
        return get_SP500_symbols()
    elif type =='manual':
        return ["META", "TSLA", "GOOGL", "AMZN", "GS", "NVDA", "AMD", "JPM", "NFLX", "MSFT"]
    else:
        return None

if __name__ == "__main__":

    p = configargparse.ArgParser()

    # general config
    p.add('--task', choices=['train', 'evaluate'])
    p.add('--ticker', choices=['SP20', 'SP100', 'SP500', 'manual'], default='SP20')
    p.add('--strategy', choices=['Indicators', 'NN'], default='Indicators')

    # data config
    p.add('--short', type=bool, default=False)
    p.add('--start_date', default=None)
    p.add('--end_date', default=None)
    p.add('--eval_start_date', default=None)
    p.add('--eval_end_date', default=None)

    # technical config
    p.add('--n_jobs', type=int, default=-1)
    p.add('--agent_num', type=int, default=0)

    config = p.parse_args()

    run_main(config)