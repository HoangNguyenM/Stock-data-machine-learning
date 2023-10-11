import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import pytz
import os

def download_alpha_vantage(symbol='TQQQ', interval='1min', start_date=datetime(2000, 1, 1), end_date=datetime.now()):
    # Get API key for alpha vantage
    api_key = input(f"Enter API key:")
    
    if start_date is None:
        start_date=datetime(2000, 1, 1)

    # # If the time is before 6pm, do not read the current day, set end_date to the date before
    # eastern = pytz.timezone('US/Eastern')
    # current_time = datetime.now(eastern)
    # cutoff_time = current_time.replace(hour=18, minute=00, second=0, microsecond=0)
    # if current_time < cutoff_time:
    end_date += timedelta(days=-1)

    year = start_date.year
    month = start_date.month
    end_year = end_date.year
    end_month = end_date.month

    # Define a list to store the dataframes
    dfs = []

    # Loop through years and months
    while year < end_year or month <= end_month:
        # Format year and month strings
        year_str = str(year)
        month_str = str(month).zfill(2)
        time_str = f'{year_str}-{month_str}'
        
        # Construct the API URL
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'datatype': 'json',
            'outputsize': 'full',
            'month': time_str,
            'extended_hours': 'false'
        }

        print(f"Downloading {symbol} data for {time_str}...")
        # Define the base URL
        base_url = 'https://www.alphavantage.co/query'
        
        # Make the API request
        retry = True
        while retry:
            response = requests.get(base_url, params=params)
            # check if the response was successful, if not, wait and try again
            if response.status_code == 200:
                data = response.json()
                # done if Time Series read or data did not exist on that month
                if 'Time Series (1min)' in data or 'Error Message' in data:
                    retry = False
                # wait if the number of API calls exceed limit
                elif data == {} or 'Note' in data or 'Information' in data:
                    time.sleep(2)
                else:
                    print(data)
                    raise Exception("API problem!")
            else:
                time.sleep(2)

        if 'Time Series (1min)' in data:
            # Read and parse the downloaded data
            time_series = data['Time Series (1min)']
            
            # Convert time series data into a list of dictionaries
            time_series_list = []
            for timestamp, values in time_series.items():
                entry = {
                    'Datetime': pd.to_datetime(timestamp),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                }
                time_series_list.append(entry)
            
            time_series_list = time_series_list[::-1]

            # Filter out data before the start date and after the end date
            if time_series_list and time_series_list[0]['Datetime'].date() < pd.to_datetime(start_date).date():
                time_series_list = [entry for entry in time_series_list if entry['Datetime'].date() >= pd.to_datetime(start_date).date()]
            if time_series_list and time_series_list[-1]['Datetime'].date() > pd.to_datetime(end_date).date():
                time_series_list = [entry for entry in time_series_list if entry['Datetime'].date() <= pd.to_datetime(end_date).date()]

            if time_series_list:
                # Create a pandas DataFrame
                df = pd.DataFrame(time_series_list)
                df.set_index('Datetime', inplace=True)
                dfs.append(df)

        # conclude the month and move on to the next month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

    if dfs:
        # Concatenate dataframes
        final_df = pd.concat(dfs)
        print(final_df.tail)
    else:
        final_df = None
    return final_df

def download_yfinance(symbol='TQQQ', interval='1m', start_date=datetime(2000, 1, 1), end_date=datetime.now()):
    dfs = []
    next_start_date = start_date

    while next_start_date < end_date:
        # Calculate the end_date after start_date 7 days
        next_end_date = next_start_date + timedelta(days=7)
        if next_end_date > end_date:
            next_end_date = end_date

        # Download data for the specified range, yfinance only allows 7-day period for minutely data
        df = yf.download(symbol, start=next_start_date.strftime("%Y-%m-%d"), end=next_end_date.strftime("%Y-%m-%d"), interval=interval)
        
        if len(df) != 0:
            dfs.append(df)

        next_start_date = next_end_date + timedelta(days=1)

    if dfs:
        # Concatenate dataframes
        final_df = pd.concat(dfs)

        final_df['Close'] = final_df['Adj Close']
        final_df.drop('Adj Close', axis=1, inplace=True)

        # remove time zone from yfinance Datetime column
        def convert_datetime(date_str):
            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
            new_date = date.replace(tzinfo=None)
            return new_date
        
        # Apply the conversion function to the column
        final_df['Datetime'] = final_df.index.map(lambda x: convert_datetime(str(x)))
        final_df.index = final_df['Datetime']
        final_df.drop('Datetime', axis=1, inplace=True)

        print(final_df.tail)
    else:
        final_df = None
    return final_df

def update_data(ticker, API):
    if os.path.isfile(f"data/{ticker}.csv"):
        data = pd.read_csv(f"data/{ticker}.csv")
        last_date_collected = data['Datetime'].values[-1].split(' ')[0]
        start_date = datetime.strptime(last_date_collected, "%Y-%m-%d") + timedelta(days=1)
    else:
        start_date = None
    
    # download more data only if the start date is yesterday or earlier
    if start_date is None or start_date.date() <= (datetime.now() + timedelta(days=-1)).date():
        if API == 'yfinance':
            data = download_yfinance(ticker, start_date=start_date)
        elif API == 'alpha vantage':
            data = download_alpha_vantage(ticker, start_date = start_date)
        else:
            raise NotImplementedError(f'API {API} not implemented')
        
        if data is not None:
            if os.path.isfile(f"data/{ticker}.csv"):
                data.to_csv(f"data/{ticker}.csv", mode='a', header=False)
            else:
                data.to_csv(f"data/{ticker}.csv")
        print(f"{ticker} updated.")

    else:
        print(f"{ticker} already up to date.")

# implemented API include yfinance and alpha vantage
def download_data(ticker='TQQQ', API='yfinance'):
    # Use this function to download the latest 1min data
    update_data(ticker, API)