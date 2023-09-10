import os
import configargparse
import run_lib
from utils.get_symbols import get_SP20_symbols, get_SP100_symbols, get_SP500_symbols
from config import Bayes_config, WMANN_config, CNN_config

def run_main(config):
    ### Use the following lines to change configs for convenience, configs can also be modified in "config" folder or by command line ###
    # general config
    config.task = 'train'
    config.model = 'CNN'
    config.strategy = 'Indicators'
    config.ticker = 'SP20'
    # data config
    config.short = False
    config.start_date = '2016-01-01'
    config.end_date = '2021-12-31'
    config.eval_start_date = '2022-01-01'
    config.eval_end_date = None
    # technical config
    config.n_jobs = 5

    model_config = get_model_config(config.model)
    config.ticker_list = get_ticker(type = config.ticker)
    config.ticker_list = ["META", "TSLA", "GOOGL", "AMZN", "GS", "NVDA", "AMD", "JPM", "NFLX", "MSFT"]
    print(config.ticker_list)

    if config.task == 'download':
        run_lib.download(config)
    elif config.task == 'train':
        run_lib.train(config, model_config)
    elif config.task == 'evaluate':
        run_lib.evaluate(config, model_config)

def get_ticker(type):
    if type == 'SP20':
        return get_SP20_symbols()
    elif type == 'SP100':
        return get_SP100_symbols()
    elif type == 'SP500':
        return get_SP500_symbols()
    else:
        return None
    
def get_model_config(model):
    if model == 'Bayes':
        return Bayes_config.get_config()
    elif model == 'WMANN':
        return WMANN_config.get_config()
    elif model == 'CNN':
        return CNN_config.get_config()

if __name__ == "__main__":

    p = configargparse.ArgParser()

    # general config
    p.add('--task', choices=['download', 'train', 'evaluate'])
    p.add('--model', choices=['Bayes', 'WMANN', 'CNN', 'DQN'])
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