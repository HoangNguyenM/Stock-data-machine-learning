import os
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

class CNN:
    def __init__(self, prior_duration=120, post_duration=60):
        super(CNN, self).__init__()
        self.prior_duration = prior_duration
        self.post_duration = post_duration

        self.model = self.get_model()

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def make_CNN(self):
        # Build the CNN model
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.prior_duration, 5), name='conv1d_1'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu', name='conv1d_2'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=256, kernel_size=3, activation='relu', name='conv1d_3'),
            MaxPooling1D(pool_size=2),
            
            Flatten(),
            
            Dense(128, activation='relu', name='dense_3'),
            Dropout(0.1),
            Dense(64, activation='relu', name='dense_2'),
            Dense(1, name='dense_1')
        ])

        return model

    def get_model(self):
        # create CNN, or load model if already pretrained
        if os.path.isfile(f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.keras'):
            model = load_model(f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.keras', compile=False)
        else:
            model = self.make_CNN()
        return model

    def train(self, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=64, steps_per_epoch=None, validation_freq=10, early_stop=False):
        # Define the EarlyStopping callback
        if early_stop:
            callbacks = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        else:
            callbacks = None
        # Train the model
        self.model.fit(x=X_train, y=Y_train, 
                epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                validation_data=(X_test, Y_test), validation_freq=validation_freq,
                callbacks=callbacks)

        self.model.save(f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.keras')

    def evaluate(self, X_test, Y_test, batch_size=64):
        loss = self.model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
        return loss
    
def make_X_y(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close are divided by max High; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)
    
    indices_dim0 = np.arange(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = np.arange(prior_duration)
    X = data[indices_dim0[..., None] + indices_dim1[None, ...], ...]
    norm_factor = np.max(X[..., 3], axis=-1)
    volume_norm_factor = np.max(X[..., 4], axis=-1)
    X = np.concatenate([X[..., :4] / norm_factor[..., None, None], X[..., 4:] / volume_norm_factor[..., None, None]], axis=-1)

    indices_dim0 = np.arange(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = np.arange(post_duration)
    Y = data[indices_dim0[..., None] + indices_dim1[None, ...], ...]
    Y = np.concatenate([Y[..., :4] / norm_factor[..., None, None], Y[..., 4:] / volume_norm_factor[..., None, None]], axis=-1)
    
    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = (np.max(Y[..., 3], axis=-1) / X[..., -1, 3] - 1) * 100
    max_loss = (np.min(Y[..., 3], axis=-1) / X[..., -1, 3] - 1) * 100
    percent_change = (Y[..., -1, 3] / X[..., -1, 3] - 1) * 100
    
    # X has shape (n_samples, sequence_len, n_channels), Y has shape (n_samples,)
    return X, max_gain, max_loss, percent_change