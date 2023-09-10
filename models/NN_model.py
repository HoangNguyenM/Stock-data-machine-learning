import os
import numpy as np

from models.WMANN import WMANN
from models.CNN import CNN

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class NN_model:
    def __init__(self, model_config):
        super(NN_model, self).__init__()
        self.prior_duration = model_config.prior_duration
        self.post_duration = model_config.post_duration
        self.model_name = model_config.model_name
        self.lib = model_config.lib
        self.save_file = f'checkpoints/{self.model_name}_{self.prior_duration}_{self.post_duration}.keras'

        self.model = self.get_model(model_config)
    
        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        self.model.compile(optimizer=adam, loss='mean_squared_error')

    def get_model(self, model_config):
        # create CNN, or load model if already pretrained
        if os.path.isfile(self.save_file):
            model = load_model(self.save_file, compile=False)
        else:
            if self.model_name == 'WMANN':
                model = WMANN(dict(model_config))
            elif self.model_name == 'CNN':
                model = CNN(dict(model_config))
            else:
                raise NotImplementedError(f"{self.model_name} not implemented.")
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
                callbacks=callbacks, verbose=1)

        self.model.save(self.save_file)

    def evaluate(self, X_test, Y_test, batch_size=64):
        loss = self.model.evaluate(x=X_test, y=Y_test, batch_size=batch_size, verbose=1)
        Y_pred = self.model.predict(X_test)[..., 0]
        # Generate random numbers without duplicates
        rand_ind = np.random.choice(np.arange(len(Y_test)), size=20, replace=False).tolist()
        if self.lib == 'numpy':
            print(list(zip(Y_test[rand_ind], Y_pred[rand_ind])))
        if self.lib == 'tensorflow':
            print(list(zip(Y_test.numpy()[rand_ind], Y_pred[rand_ind])))