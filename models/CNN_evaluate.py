import numpy as np

from keras.models import load_model
from CNN_deep import make_X_y, import_data


def evaluate_model(data_list, prior_duration=60, post_duration=30):
    CNN_model = load_model(f'models/CNN_{prior_duration}_{post_duration}.h5', compile=False)
    CNN_deep_model = load_model(f'models/CNN_deep_{prior_duration}_{post_duration}.h5', compile=False)
    CNN_deeper_model = load_model(f'models/CNN_deeper_{prior_duration}_{post_duration}.h5', compile=False)

    X_list, y_list = [], []

    for data in data_list:
        # get X and y
        X, y = make_X_y(data.copy(), prior_duration=prior_duration, post_duration=post_duration, scale=True)

        X = X[:(-post_duration)]

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    CNN_model.compile(optimizer='adam', loss='mean_squared_error')
    CNN_deep_model.compile(optimizer='adam', loss='mean_squared_error')
    CNN_deeper_model.compile(optimizer='adam', loss='mean_squared_error')

    CNN_model.evaluate(x=X,y=y)
    CNN_deep_model.evaluate(x=X,y=y)
    CNN_deeper_model.evaluate(x=X,y=y)

if __name__ == "__main__":

    ticker_list = ['AMD', 'AMZN', 'GOOGL', 'GS', 'MSFT', 'NVDA', 'SPOT', 'TQQQ', 'TSLA']
    data_list = import_data(ticker_list=ticker_list)

    # Create sequences of input data and target values
    prior_duration = [30, 45, 60, 90, 120]
    post_duration = [15, 30, 45]

    for prior in prior_duration:
        for post in post_duration:
            print(f"__Evaluating models with prior_duration: {prior} and post_duration: {post}")
            evaluate_model(data_list=data_list, prior_duration=prior, post_duration=post)