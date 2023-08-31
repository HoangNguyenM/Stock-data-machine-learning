import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from CNN import import_data, make_X_y

def make_CNN(prior_duration=60):
    # Build the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(prior_duration, 5), kernel_regularizer=l2(0.01), name='conv1d_1'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), name='conv1d_2'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01), name='conv1d_3'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), name='dense_3'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01), name='dense_2'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, name='dense_1')
    ])

    return model

def train_model(data_list, prior_duration=60, post_duration=30):
    pretrained = False
    # create CNN, or load model if already pretrained
    if os.path.isfile(f'models/CNN_deeper_{prior_duration}_{post_duration}.h5'):
        model = load_model(f'models/CNN_deeper_{prior_duration}_{post_duration}.h5', compile=False)
        pretrained = True
    else:
        model = make_CNN(prior_duration=prior_duration)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # if the model is trained for the first time, get the weights from smaller models
    if not pretrained:
        small_model = load_model(f'models/CNN_deep_{prior_duration}_{post_duration}.h5', compile=False)

        for layer_name in ['conv1d_1', 'conv1d_2', 'conv1d_3', 'dense_3', 'dense_2', 'dense_1']:
            model.get_layer(layer_name).set_weights(small_model.get_layer(layer_name).get_weights())

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

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(x=X_train, y=y_train, 
            epochs=100, batch_size=64, 
            validation_data=(X_test, y_test),
            validation_freq=10,
            callbacks=[early_stopping])

    model.save(f'models/CNN_deeper_{prior_duration}_{post_duration}.h5')

if __name__ == "__main__":

    ticker_list = ['AMD', 'AMZN', 'GOOGL', 'GS', 'MSFT', 'NVDA', 'SPOT', 'TQQQ', 'TSLA']
    data_list = import_data(ticker_list=ticker_list)

    # Create sequences of input data and target values
    prior_duration = [30, 45, 60, 90, 120]
    post_duration = [15, 30, 45]

    for prior in prior_duration:
        for post in post_duration:
            train_model(data_list=data_list, prior_duration=prior, post_duration=post)