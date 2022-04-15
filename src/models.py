
import torch
import torch.nn as nn


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
pl.utilities.seed.seed_everything(seed=42)


class NeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features, units=512):
        super(NeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.flatten = nn.Flatten()
        self.input_linear = nn.Linear(self.in_features, self.units)
        self.hidden_layer = nn.Linear(self.units, self.units)
        self.output_layer = nn.Linear(self.units, self.out_features)

    def forward(self, x):
        x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        x = F.relu(self.input_linear(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer_name='rmsprop', 
        lr=0.003, 
        loss_fn_name='mse'
        ):

        self.model = model
        self.lr = lr
        self.optimizer_name=optimizer_name
        self.loss_fn_name = loss_fn_name
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}-device")

        if self.loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()

        if self.optimizer_name.lower() == 'rmsprop': 
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)

        elif self.optimizer_name.lower() == 'adam':
            pass

    def _set_optimizer(self):
        try:
            pass
        except:
            pass

    def _set_loss(self):
        try:
            pass
        except:
            pass

    def check_optimizer_loss_args(self):
        print(f'Allowed opmimizer names are:')
        print(f'Allowed loss function names are:')

    def fit_epochs(self, train_loader, valid_loader=None, epochs=5):
        for epoch in epochs:
            self.fit_one_epoch(train_loader, valid_loader)

    def fit_one_epoch(self, train_loader, valid_loader=None):
        size = len(train_loader)
        self.model.to(self.device).train()
        for batch, data in enumerate(train_loader):
            x = data['features']
            y = data['target']
            x, y = x.to(self.device), y.to(self.device)
            self._run_train_step(x, y, batch, size)

        if valid_loader is not None:
            size = len(valid_loader)
            with torch.no_grad():
                for batch, data in enumerate(valid_loader):
                    xval = data['features']
                    yval = data['target']
                    xval, yval = xval.to(self.device), yval.to(self.device)
                    self.evaluate(xval, yval, batch, size)

    def evaluate(self, x, y, batch, size):
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        loss += self.loss_fn(pred, y).item()
        print(f'val-loss: {loss.item()} [{batch * len(x)/{size}}]')


    def _run_train_step(self, x, y, batch, size):
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())
        # if batch % 100 == 0:
        print(f'loss: {loss.item()} [{batch * len(x)}/{size}]')
