import tensorflow as tf
import keras
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LayerNormalization, LeakyReLU
from keras.regularizers import l2

class CNN(keras.Model):
    def __init__(self, model_config):
        super(CNN, self).__init__()
        self.prior_duration = model_config.prior_duration
        self.post_duration = model_config.post_duration
        
        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        self.conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool1 = MaxPooling1D(pool_size=2)
        self.conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool2 = MaxPooling1D(pool_size=2)
        self.conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool3 = MaxPooling1D(pool_size=2)
        
        self.flatten = Flatten()
        
        self.dropout1 = Dropout(model_config.dropout)
        self.fc1 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.dropout2 = Dropout(model_config.dropout)
        self.fc2 = Dense(64, activation='relu', kernel_initializer=initializer)
        self.fc3 = Dense(4, kernel_initializer=initializer)

        self.build(input_shape=(None, self.prior_duration, 5))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x