import os
import numpy as np
import tensorflow as tf
import torch

def make_X_Y(lib):
    if lib == 'numpy':
        return make_X_Y_numpy
    elif lib == 'tensorflow':
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        return make_X_Y_tf
    elif lib == 'torch':
        return make_X_Y_torch

def make_X_Y_numpy(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close: lowest low to -1, highest high to 1; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)
    
    data = np.array(data, dtype=np.float32)

    # normalize function
    def scale(x, min_value, max_value):
        return -1 + 2 * (x - min_value) / (max_value - min_value)

    indices_dim0 = np.arange(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = np.arange(prior_duration)
    X = data[indices_dim0[..., None] + indices_dim1[None, ...], ...]
    indices_dim0 = np.arange(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = np.arange(post_duration)
    Y = data[indices_dim0[..., None] + indices_dim1[None, ...], ...]

    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = (np.max(Y[..., 3], axis=-1) / X[..., -1, 3] - 1) * 100
    max_loss = (np.min(Y[..., 3], axis=-1) / X[..., -1, 3] - 1) * 100
    avg_change = (np.mean(Y[..., 3], axis=-1) / X[..., -1, 3] - 1) * 100
    percent_change = (Y[..., -1, 3] / X[..., -1, 3] - 1) * 100

    # Normalize X and Y
    min_norm_factor = np.min(X[..., 2], axis=-1)
    max_norm_factor = np.max(X[..., 1], axis=-1)
    volume_norm_factor = np.max(X[..., 4], axis=-1)
    X = np.concatenate([scale(X[..., :4], min_norm_factor, max_norm_factor), X[..., 4:] / volume_norm_factor[..., None, None]], axis=-1)
    Y = np.stack([max_gain, max_loss, avg_change, percent_change], axis=-1)
    
    # X has shape (n_samples, sequence_len, n_channels), Y has shape (n_samples,)
    return X, Y

@tf.function
def make_X_Y_tf(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close: lowest low to -1, highest high to 1; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)

    data = tf.convert_to_tensor(data, dtype=tf.float32)

    # normalize function
    def scale(x, min_value, max_value):
        temp = x - min_value[..., tf.newaxis, tf.newaxis]
        coef = tf.constant(2, dtype=tf.float32) / (max_value - min_value)
        temp = temp * coef[..., tf.newaxis, tf.newaxis]
        temp = tf.add(temp, tf.constant(-1, dtype=tf.float32))
        return temp

    indices_dim0 = tf.range(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = tf.range(prior_duration)
    X = tf.gather(data, indices_dim0[..., tf.newaxis] + indices_dim1[tf.newaxis, ...], axis=0)
    indices_dim0 = tf.range(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = tf.range(post_duration)
    Y = tf.gather(data, indices_dim0[..., tf.newaxis] + indices_dim1[tf.newaxis, ...], axis=0)

    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = ((tf.reduce_max(Y[..., 3], axis=-1) / X[..., -1, 3]) - 1) * 100
    max_loss = ((tf.reduce_min(Y[..., 3], axis=-1) / X[..., -1, 3]) - 1) * 100
    avg_change = ((tf.reduce_mean(Y[..., 3], axis=-1) / X[..., -1, 3]) - 1) * 100
    percent_change = ((Y[..., -1, 3] / X[..., -1, 3]) - 1) * 100

    # Normalize X and Y
    min_norm_factor = tf.reduce_min(X[..., 2], axis=-1)
    max_norm_factor = tf.reduce_max(X[..., 1], axis=-1)
    volume_norm_factor = tf.reduce_max(X[..., 4], axis=-1)
    X = tf.concat([scale(X[..., :4], min_norm_factor, max_norm_factor), X[..., 4:] / volume_norm_factor[..., tf.newaxis, tf.newaxis]], axis=-1)
    Y = tf.stack([max_gain, max_loss, avg_change, percent_change], axis=-1)

    # X has shape (n_samples, sequence_len, n_channels), Y has shape (n_samples,)
    return X, Y

def make_X_Y_torch(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close: lowest low to -1, highest high to 1; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)

    data = torch.tensor(data, dtype=torch.float32)

    # Normalize function
    def scale(x, min_value, max_value):
        temp = x - min_value[..., None, None]
        coef = torch.tensor(2.0, dtype=torch.float32) / (max_value - min_value)
        temp = temp * coef[..., None, None]
        temp = temp + torch.tensor(-1.0, dtype=torch.float32)
        return temp

    indices_dim0 = torch.arange(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = torch.arange(prior_duration)
    X = data[indices_dim0[..., None] + indices_dim1[None, ...]]

    indices_dim0 = torch.arange(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = torch.arange(post_duration)
    Y = data[indices_dim0[..., None] + indices_dim1[None, ...]]

    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = ((Y[..., 3].max(dim=-1).values / X[..., -1, 3]) - 1) * 100
    max_loss = ((Y[..., 3].min(dim=-1).values / X[..., -1, 3]) - 1) * 100
    avg_change = ((Y[..., 3].mean(dim=-1) / X[..., -1, 3]) - 1) * 100
    percent_change = ((Y[..., -1, 3] / X[..., -1, 3]) - 1) * 100

    # Normalize X and Y
    min_norm_factor = X[..., 2].min(dim=-1).values
    max_norm_factor = X[..., 1].max(dim=-1).values
    volume_norm_factor = X[..., 4].max(dim=-1).values
    X = torch.cat([scale(X[..., :4], min_norm_factor, max_norm_factor), X[..., 4:] / volume_norm_factor[..., None, None]], dim=-1)
    Y = torch.stack([max_gain, max_loss, avg_change, percent_change], dim=-1)

    # X has shape (n_samples, sequence_len, n_channels), Y has shape (n_samples,)
    return X, Y