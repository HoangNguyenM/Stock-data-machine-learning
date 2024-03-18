import configargparse
from utils.get_symbols import get_SP20_symbols, get_SP100_symbols, get_SP500_symbols
from utils.download_data import download_data

### This file contains the main code to download stock data from an api into "data" folder as csv files.
# Supported api include yfinance and alpha vantage, note that alpha vantage needs an api key

def run_main(config):
    ### Use the following lines to change configs for convenience, configs can also be modified in "config" folder or by command line ###
    # general config
    # config.ticker = 'SP500'
    # config.api = 'yfinance'

    config.ticker_list = get_ticker(type = config.ticker)
    print(config.ticker_list)

    ### download/update the data for all tickers ###
    for ticker in config.ticker_list:
        #if ticker not in SP100:
        download_data(ticker=ticker, API=config.api, api_key=config.api_key)

def get_ticker(type):
    if type == 'SP20':
        return get_SP20_symbols()
    elif type == 'SP100':
        return get_SP100_symbols()
    elif type == 'SP500':
        return get_SP500_symbols()
    else:
        return None

if __name__ == "__main__":
    p = configargparse.ArgParser()

    # general config
    p.add('--ticker', choices=['SP20', 'SP100', 'SP500', 'manual'], default='SP500')
    p.add('--api', choices=['yfinance', 'alpha vantage'], default='yfinance')
    p.add('--api_key', type=str, default=None)

    config = p.parse_args()

    run_main(config)