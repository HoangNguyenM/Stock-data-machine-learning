import tensorflow as tf

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

@tf.function
def make_X_y(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close are divided by max High; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)

    indices_dim0 = tf.range(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = tf.range(prior_duration)
    X = tf.gather(data, indices_dim0[..., tf.newaxis] + indices_dim1[tf.newaxis, ...], axis=0)
    norm_factor = tf.reduce_max(X[..., 3], axis=-1)
    volume_norm_factor = tf.reduce_max(X[..., 4], axis=-1)
    X = tf.concat([X[..., :4] / norm_factor[..., tf.newaxis, tf.newaxis], X[..., 4:] / volume_norm_factor[..., tf.newaxis, tf.newaxis]], axis=-1)

    indices_dim0 = tf.range(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = tf.range(post_duration)
    Y = tf.gather(data, indices_dim0[..., tf.newaxis] + indices_dim1[tf.newaxis, ...], axis=0)
    Y = tf.concat([Y[..., :4] / norm_factor[..., tf.newaxis, tf.newaxis], Y[..., 4:] / volume_norm_factor[..., tf.newaxis, tf.newaxis]], axis=-1)

    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = ((tf.reduce_max(Y[..., 3], axis=-1) / X[..., -1, 3]) - 1) * 100
    max_loss = ((tf.reduce_min(Y[..., 3], axis=-1) / X[..., -1, 3]) - 1) * 100
    percent_change = ((Y[..., -1, 3] / X[..., -1, 3]) - 1) * 100

    # X has shape (n_samples, sequence_len, n_channels), Y has shape (n_samples,)
    return X, max_gain, max_loss, percent_change
