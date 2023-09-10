import tensorflow as tf
import keras
from keras.layers import Conv1D, Conv2D, Flatten, Dense, Dropout, LayerNormalization, LeakyReLU
from keras.regularizers import l2

class WMANN(keras.Model):
    def __init__(self, model_config):
        super(WMANN, self).__init__()
        self.model_config = model_config

        for key, value in model_config.items():
            setattr(self, key, value)

        initializer = tf.keras.initializers.GlorotNormal(seed=42)

        self.input_len = [self.get_input_len(self.kernel_list[i], self.stride_size, self.ma_output_size) for i in range(len(self.kernel_list))]
        self.conv1 = []

        for i in range(len(self.kernel_list)):
            self.conv1.append(Conv1D(self.nnodes1, kernel_size=int(self.kernel_list[i]), strides=self.stride_size, kernel_initializer=initializer))

        self.conv2 = Conv2D(self.nnodes2, kernel_size=(1,self.ma_combine), strides=1, kernel_initializer=initializer)
        self.flatten = Flatten()

        self.dropout1 = Dropout(self.dropout)
        self.fc1 = Dense(512, kernel_regularizer=l2(self._lambda), bias_regularizer=l2(self._lambda), kernel_initializer=initializer)
        self.leakyrelu1 = LeakyReLU(0.2)
        self.norm1 = LayerNormalization()

        self.dropout2 = Dropout(self.dropout)
        self.fc2 = Dense(128, bias_regularizer=l2(self._lambda), kernel_initializer=initializer)
        self.leakyrelu2 = LeakyReLU(0.2)
        self.norm2 = LayerNormalization()

        self.fc3 = Dense(4, kernel_initializer=initializer)

        self.build(input_shape=(None, self.prior_duration, 5))

    def get_input_len(self, kernel_size, strides, output_size):
        return kernel_size + strides * (output_size - 1)

    def call(self, inputs):
        outs = []
        for i in range(len(self.kernel_list)):
            x = self.conv1[i](inputs[..., - self.input_len[i] : , :]) # output shape is (None, ma_output_size, self.nnodes1), e.g. (None, 16, 64)
            outs.append(x)

        outs = tf.stack(outs, axis=-2) # output shape is (None, ma_output_size, kernel_list size, self.nnodes1), e.g. (None, 16, 12, 64) 

        x = self.conv2(outs) # output shape is (None, ma_output_size, kernel_list size - 1, self.nnodes1), e.g. (None, 16, 11, 4) 
        x = self.flatten(x)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.leakyrelu1(x)
        x = self.norm1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.norm2(x)

        x = self.fc3(x)

        return x
    
    def get_config(self):
        config = super(WMANN, self).get_config()
        config.update({'model_config': self.model_config})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)