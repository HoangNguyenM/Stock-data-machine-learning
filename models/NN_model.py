import os
import numpy as np

from models.WMANN import WMANN
from models.CNN import CNN

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NN_model:
    def __new__(cls, model_config):
        if model_config.lib == "tensorflow" or model_config.lib == "numpy":
            return NN_model_tf(model_config)
        elif model_config.lib == "torch":
            return NN_model_torch(model_config)
        else:
            raise ValueError(f"ML lib {model_config.lib} not supported for {model_config.model_name}.")

class NN_base:
    def __init__(self, model_config):
        self.prior_duration = model_config.prior_duration
        self.post_duration = model_config.post_duration
        self.model_name = model_config.model_name
        self.lib = model_config.lib
        if self.lib == 'tensorflow' or self.lib == 'numpy':
            self.save_file = f'checkpoints/{self.model_name}_{self.prior_duration}_{self.post_duration}.keras'
        elif self.lib == 'torch':
            self.save_file = f'checkpoints/{self.model_name}_{self.prior_duration}_{self.post_duration}.pt'

        self.model = self.get_model(model_config)

    def get_model(self, model_config):
        # create NN model, and load model if already pretrained
        if self.model_name == 'WMANN':
            model = WMANN(dict(model_config))
        elif self.model_name == 'CNN':
            model = CNN(dict(model_config))
        else:
            raise NotImplementedError(f"{self.model_name} not implemented.")
        
        if os.path.isfile(self.save_file):
            if self.lib == 'tensorflow' or self.lib == 'numpy':
                model.load_weights(self.save_file)
            elif self.lib == 'torch':
                model.load_state_dict(self.save_file)
        return model

class NN_model_tf(NN_base):
    def __init__(self, model_config):
        super(NN_model_tf, self).__init__(model_config)
        self.optimizer = Adam(learning_rate=model_config.max_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-9)
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        self.model.summary()

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

        self.model.save_weights(self.save_file)

    def evaluate(self, X_test, Y_test, batch_size=64):
        loss = self.model.evaluate(x=X_test, y=Y_test, batch_size=batch_size, verbose=1)
        Y_pred = self.model.predict(X_test)
        # Generate samples for reference
        rand_ind = np.random.choice(np.arange(len(Y_test)), size=20, replace=False).tolist()
        if self.lib == 'numpy':
            print("Some example (true labels, pred_labels) pairs are: ", list(zip(Y_test[rand_ind], Y_pred[rand_ind])))
        if self.lib == 'tensorflow':
            print("Some example (true labels, pred_labels) pairs are: ", list(zip(Y_test.numpy()[rand_ind], Y_pred[rand_ind])))

class NN_model_torch(NN_base):
    def __init__(self, model_config):
        super(NN_model_torch, self).__init__(model_config)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_config.max_lr, betas=(0.9, 0.999), eps=1e-9)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")

    def train(self, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=64, steps_per_epoch=None, validation_freq=10, early_stop=False):
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            if (epoch+1) % validation_freq == 0:
                self.model.eval()
                eval_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, targets.view(-1, 1))
                        eval_loss += loss.item()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader)}, Eval Loss: {eval_loss / len(test_loader)}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader)}")

        # Save the model
        torch.save(self.model.state_dict(), self.save_file)

    def evaluate(self, X_test, Y_test, batch_size=64):
        self.model.eval()
        predictions = []
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                predictions.extend(outputs.tolist())
                loss = self.loss_function(outputs, targets.view(-1, 1))
                eval_loss += loss.item()

        print(f"Test Loss: {eval_loss / len(test_loader)}")

        # Generate samples for reference
        rand_ind = np.random.choice(np.arange(len(Y_test)), size=20, replace=False).tolist()
        print("Some example (true labels, pred_labels) pairs are: ", list(zip(Y_test.numpy()[rand_ind], predictions[rand_ind])))