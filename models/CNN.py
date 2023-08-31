import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

import get_data

def import_data(ticker_list):
    data_list = []
    # get data for each ticker
    for ticker in ticker_list:
        data = get_data.get_data(ticker=ticker, update=False)
        data_list.append(data)

    return data_list

def make_X_y(data, prior_duration=60, post_duration=30, scale=True):

    # Calculate percentage change in Close
    y = (data['Close'].values[prior_duration+post_duration:] / data['Close'].values[prior_duration:(-post_duration)] - 1) * 100

    # Scale the data
    close_mean = data['Close'].mean()
    close_std = data['Close'].std()
    volume_mean = data['Volume'].mean()
    volume_std = data['Volume'].std()

    # normalize the price
    data[['Open', 'High', 'Low', 'Close']] \
        = (data[['Open', 'High', 'Low', 'Close']] - close_mean) / close_std

    # normalize the volume
    data['Volume'] = (data['Volume'] - volume_mean) / volume_std

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    X = np.array([data[i:i+prior_duration] for i in range(len(data) - prior_duration)])
    
    return X, y

def make_CNN(prior_duration=60):
    # Build the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(prior_duration, 5), name='conv1d_1'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', name='conv1d_2'),
        MaxPooling1D(pool_size=2),

        Flatten(),
        
        Dense(64, activation='relu', name='dense_2'),
        Dropout(0.1),
        Dense(1, name='dense_1')
    ])

    return model
    
def train_model(data_list, prior_duration=60, post_duration=30):
    # create CNN, or load model if already pretrained
    if os.path.isfile(f'models/CNN_{prior_duration}_{post_duration}.h5'):
        model = load_model(f'models/CNN_{prior_duration}_{post_duration}.h5', compile=False)
    else:
        model = make_CNN(prior_duration=prior_duration)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    X_list, y_list = [], []

    for data in data_list:
        # get X and y
        X, y = make_X_y(data.copy(), prior_duration=prior_duration, post_duration=post_duration, scale=True)

        X = X[:(-post_duration)]

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(X.shape)
    print(y.shape)

    # Split the data into training and testing sets
    # Randomly shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_ratio = 0.8
    split_idx = int(split_ratio * len(X))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train the model
    model.fit(x=X_train, y=y_train, 
            epochs=50, batch_size=64, 
            validation_data=(X_test, y_test),
            validation_freq=10)

    model.save(f'models/CNN_{prior_duration}_{post_duration}.h5')

if __name__ == "__main__":

    ticker_list = ['AMD', 'AMZN', 'GOOGL', 'GS', 'MSFT', 'NVDA', 'SPOT', 'TQQQ', 'TSLA']
    data_list = import_data(ticker_list=ticker_list)

    # Create sequences of input data and target values
    prior_duration = [30, 45, 60, 90, 120]
    post_duration = [15, 30, 45]

    for prior in prior_duration:
        for post in post_duration:
            train_model(data_list=data_list, prior_duration=prior, post_duration=post)