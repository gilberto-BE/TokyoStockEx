
import torch
import torch.nn as nn


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
pl.utilities.seed.seed_everything(seed=42)
from torch import nn, Tensor
import math


class EmbeddingNetwork(nn.Module):
    def __init__(
        self, 
        units, 
        no_embedding, 
        emb_dim
    ):
        super(EmbeddingNetwork, self).__init__()
        self.units=units
        self.no_embedding = no_embedding
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(self.no_embedding, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.units)
        self.out = nn.Linear(self.units, 1)
        
    def forward(self, x):
        x = F.relu(self.embedding(x))
        print('x.shape after F.relu(embedding(k)):', x.shape)
        x = F.relu(self.linear(x))
        print('x.shape after linear + relu:', x.shape)
        x = self.out(x)
        print('x.shape after self.out(x):', x.shape)
        print()
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[ :x.size(0)]
        return self.dropout(x)


class NeuralNetwork(nn.Module):
    
    def __init__(
        self, 
        in_features, 
        out_features,
        categorical_dim=3, 
        units=512, 
        no_embedding=None, 
        emb_dim=None):

        """
        TODO:
        * Add normalization layers
        * Add regularization
        * Test other optimizers
        * Add positional encoding
        """

        super(NeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.no_embedding = no_embedding
        self.emb_dim = emb_dim
        self.categorical_dim = categorical_dim
        self.flatten = nn.Flatten()
        if no_embedding and emb_dim:
            self.embedding = nn.Embedding(self.no_embedding, self.emb_dim)
            self.embedding_to_hidden = nn.Linear(self.emb_dim, self.units)
            self.embedding_output = nn.Linear(self.units, self.out_features)
        
        self.cont_input = nn.Linear(self.in_features, self.units)
        self.hidden_layer = nn.Linear(
            self.units + self.categorical_dim, 
            self.units + self.categorical_dim
            )
        self.output_layer = nn.Linear(
            self.units + self.categorical_dim, 
            self.out_features
            )
        # self.pool_layer = nn.MaxPool1d(3, 2)

    def forward(self, x, x_cat=None):
        """
        TODO:
        * Add residual connictions.
        
        """
        if x_cat is not None:
            emb_residual = x_cat
            x_out = self.embedding(x_cat)
            x_out = F.relu(self.embedding_to_hidden(x_out))
            x_out = torch.squeeze(torch.real(torch.fft.fft2(self.embedding_output(x_out))))
            # x_out += emb_residual

        cont_residual = x
        x = torch.real(torch.fft.rfft(x))
        x = F.relu(self.cont_input(x))
        x += cont_residual
        x = torch.cat((x, x_out.view((x_out.shape[0], -1))), dim=1)
        tot_residual = x
        x = F.relu(self.hidden_layer(x))
        x = F.relu(self.hidden_layer(x))
        x += tot_residual
        x = self.output_layer(x)
        return x
    

def load_trained_model():
    pass


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer_name='rmsprop', 
        lr=3e-6, 
        loss_fn_name='mse'
        ):

        """
        TODO
        1) Fix repeating code in many places.
        """

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

    def fit_epochs(
        self, train_loader, valid_loader=None, use_cyclic_lr=False, epochs=5, x_cat=None
        ):
        for epoch in range(epochs):
            print(f'Epoch: <<< {epoch} >>>')
            self.fit_one_epoch(train_loader, valid_loader, use_cyclic_lr, x_cat)

    def fit_one_epoch(
        self, train_loader, valid_loader=None, use_cyclic_lr=False, x_cat=None
        ):
        if use_cyclic_lr:
            """add parameters for scheduler to constructor."""
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr, max_lr=0.1)
        size = len(train_loader)
        self.model.to(self.device).train()
        for batch, data in enumerate(train_loader):
            xtrain = data['num_features']
            if x_cat is not None:
                xtrain_cat = data['cat_features']
            
            ytrain = data['target']
            train_loss = self._run_train_step(xtrain, ytrain, batch, size, scheduler, xtrain_cat)

        if valid_loader is not None:
            size = len(valid_loader)
            with torch.no_grad():
                for batch_val, data_val in enumerate(valid_loader):
                    xval = data_val['num_features']
                    if x_cat is not None:
                        xval_cat = data_val['cat_features']
                    yval = data_val['target']
                    val_loss = self.evaluate(xval, yval, batch_val + 1, size, xval_cat)

    def evaluate(self, x, y, batch, size, x_cat=None):
        loss = 0.0
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)
            pred = self.model(x, x_cat)
        else:
            pred = self.model(x)
        loss += self.loss_fn(pred, y)#.item()
        self.valid_loss.append(loss.item())
        print(f'Val-Loss: {loss.item()} [{batch}/{size}]')
        return loss.item()

    def _run_train_step(self, x, y, batch, size, scheduler, x_cat=None):
        x, y = x.to(self.device), y.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)
            pred = self.model(x, x_cat)
        else:
            pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())
        # if batch % 100 == 0:
        print(f'Train-Loss: {loss.item()} [{batch }/{size}]')
        if scheduler:
            scheduler.step()
        return loss.item()

    def get_error_loss(self, loss_type='train'):
        """
        Get error-loss for training and validation sets.
        """
        if loss_type == 'train':
            return self.train_loss
        else:
            if len(self.valid_loss) > 0:
                return self.valid_loss

    def save_model(self):
        pass

    