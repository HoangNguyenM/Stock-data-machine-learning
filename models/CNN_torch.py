import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self, prior_duration=120, post_duration=60):
        super(CNN, self).__init__()
        self.prior_duration = prior_duration
        self.post_duration = post_duration

        self.model = self.get_model()

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def make_CNN(self):
        # Build the CNN model
        model = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(3328, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        return model

    def forward(self, x):
        return self.model(x)
    
    def get_model(self):
        # create CNN, or load model if already pretrained
        if os.path.isfile(f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.pt'):
            checkpoint = torch.load(f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.pt')
            model.load_state_dict(checkpoint)
        else:
            model = self.make_CNN()
        return model

    def train(self, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=64):
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_function(outputs, targets.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

        # Save the model
        torch.save(self.model.state_dict(), f'checkpoints/CNN_{self.prior_duration}_{self.post_duration}.pt')

    def evaluate(self, X_test, Y_test, batch_size=64):
        self.model.eval()
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self(inputs)
                loss = self.loss_function(outputs, targets.view(-1, 1))
                total_loss += loss.item()
        
        return total_loss / len(test_loader)

def make_X_y(data, prior_duration=120, post_duration=60):
    # normalize the state: Open, High, Low, Close are divided by max High; Volume is divided by max Volume
    # X contains the data from prior windows, Y contains the data from post windows
    # Y is also normalized using the same factors as X
    # X has shape (data.shape[0]-prior-post, prior, 5), Y has shape (data.shape[0]-prior-post, post, 5)

    indices_dim0 = torch.arange(data.shape[0] - prior_duration - post_duration)
    indices_dim1 = torch.arange(prior_duration)
    X = data[indices_dim0.view(-1, 1) + indices_dim1[None, ...], ...]
    norm_factor = torch.max(X[..., 3], dim=-1)[0]
    volume_norm_factor = torch.max(X[..., 4], dim=-1)[0]
    X = torch.cat([X[..., :4] / norm_factor[..., None, None], X[..., 4:] / volume_norm_factor[..., None, None]], dim=-1)

    indices_dim0 = torch.arange(prior_duration, data.shape[0] - post_duration)
    indices_dim1 = torch.arange(post_duration)
    Y = data[indices_dim0.view(-1, 1) + indices_dim1[None, ...], ...]
    Y = torch.cat([Y[..., :4] / norm_factor[..., None, None], Y[..., 4:] / volume_norm_factor[..., None, None]], dim=-1)

    # Calculate different outputs from the post duration, all in percent, relative to the last input in X
    max_gain = (torch.max(Y[..., 3], dim=-1)[0] / X[..., -1, 3] - 1) * 100
    max_loss = (torch.min(Y[..., 3], dim=-1)[0] / X[..., -1, 3] - 1) * 100
    percent_change = (Y[..., -1, 3] / X[..., -1, 3] - 1) * 100

    # for torch purpose, X needs to have shape (n_samples, n_channels, sequence_len)
    return torch.transpose(X, 1, 2), max_gain, max_loss, percent_change
