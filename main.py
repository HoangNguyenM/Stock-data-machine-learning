import configargparse
import logging
import torch

import run_lib
from utils.get_symbols import get_SP20_symbols, get_SP100_symbols, get_SP500_symbols
from config import MANet_config, CNN_config


def set_logger(gfile_stream):
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

def run_main(config):
    ### Use the following lines to change configs for convenience, configs can also be modified in "config" folder or by command line ###
    # general config
    config.task = 'train'
    config.model = 'MANet'
    config.ticker = 'manual'
    # data config
    config.short = False
    config.start_date = '2016-01-01'
    config.end_date = '2021-12-31'
    config.eval_start_date = '2022-01-01'
    config.eval_end_date = None
    # technical config
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = get_model_config(config.model)
    config.ticker_list = get_ticker(type = config.ticker)
    print(config.ticker_list)

    gfile_stream = open('stdout.txt', 'w')
    set_logger(gfile_stream)

    if config.task == 'train':
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
    elif type == 'manual':
        return ["META", "TSLA", "GOOGL", "AMZN", "GS", "NVDA", "AMD", "JPM", "NFLX", "MSFT"]
    else:
        return None
    
def get_model_config(model):
    if model == 'MANet':
        return MANet_config.get_config()
    elif model == 'CNN':
        return CNN_config.get_config()

if __name__ == "__main__":

    p = configargparse.ArgParser()

    # general config
    p.add('--task', choices=['train', 'evaluate'])
    p.add('--model', choices=['MANet', 'CNN', 'DQN'])
    p.add('--ticker', choices=['SP20', 'SP100', 'SP500', 'manual'], default='SP20')

    # data config
    p.add('--short', type=bool, default=False)
    p.add('--start_date', default=None)
    p.add('--end_date', default=None)
    p.add('--eval_start_date', default=None)
    p.add('--eval_end_date', default=None)

    config = p.parse_args()

    run_main(config)