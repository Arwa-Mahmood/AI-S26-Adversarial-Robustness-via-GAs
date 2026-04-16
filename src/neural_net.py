# NEURAL NETWROK BASELINE   -- 2 hidden layers 

import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train_nn(X_train, y_train, epochs=10, lr=0.001): ...
    # DataLoader → CrossEntropyLoss → Adam → train loop

def evaluate_nn(model, X_test, y_test): ...
    # returns accuracy

def predict_nn(model, X): ...
    # returns predicted class array

def predict_proba_nn(model, X): ...
    # returns softmax probabilities (for GA fitness)