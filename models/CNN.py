import tensorflow as tf
import keras
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LayerNormalization, LeakyReLU
from keras.regularizers import l2

import torch
import torch.nn as nn

class CNN:
    def __new__(cls, model_config):
        if model_config["lib"] == "tensorflow" or model_config["lib"] == "numpy":
            return CNN_tf(model_config)
        elif model_config["lib"] == "torch":
            return CNN_torch(model_config)

class CNN_tf(keras.Model):
    def __init__(self, model_config):
        super(CNN_tf, self).__init__()
        self.model_config = model_config

        for key, value in model_config.items():
            setattr(self, key, value)
        
        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        self.conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool1 = MaxPooling1D(pool_size=2)
        self.conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool2 = MaxPooling1D(pool_size=2)
        self.conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', kernel_initializer=initializer)
        self.maxpool3 = MaxPooling1D(pool_size=2)
        
        self.flatten = Flatten()
        
        self.dropout1 = Dropout(self.dropout)
        self.fc1 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.dropout2 = Dropout(self.dropout)
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
    
    def get_config(self):
            config = super(CNN_tf, self).get_config()
            config.update({'model_config': self.model_config})
            return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CNN_torch(nn.Module):
    def __init__(self, model_config):
        super(CNN_torch, self).__init__()
        self.model_config = model_config

        for key, value in model_config.items():
            setattr(self, key, value)

        self.initializer = nn.init.xavier_normal_
        
        self.conv1 = nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.dropout1 = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(256 * (self.prior_duration // 8), 128)
        self.relu4 = nn.ReLU()

        self.dropout2 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(64, 4)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu5(x)

        x = self.fc3(x)

        return x