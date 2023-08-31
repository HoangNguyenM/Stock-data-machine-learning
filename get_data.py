import pandas as pd

def split_data_by_date(data, start_date, end_date):

    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data['Datetime'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data['Datetime'] <= end_date]
    
    return data.reset_index(drop=True)

def get_data(ticker='TQQQ', start_date=None, end_date=None, clean_first_day=False):
    # Read the data
    data = pd.read_csv(f"data/{ticker}.csv")

    # Convert the dataframe data from str to numeric type
    columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in columns:
        data[col] = pd.to_numeric(data[col])
    # Convert Datetime from str to datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # get the correct portion of data if start_date or end_date specified
    if start_date or end_date:
        data = split_data_by_date(data, start_date=start_date, end_date=end_date)

    # drop unncessary columns
    data = data.drop(labels="Datetime", axis=1)

    return data