
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
    
    def __init__(
        self, 
        in_features, 
        out_features, 
        units=512, 
        num_embedding=None, 
        embedding_dim=None):

        super(NeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.flatten = nn.Flatten()
        if num_embedding and embedding_dim:
            self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)

        # input dim to emb_input is embedding_dim * categorical_features 
        self.emb_input = nn.Linear(self.embedding_dim, self.units)
        self.emb_output = nn.Linear(self.units, self.out_features)
        
        self.cont_input = nn.Linear(self.in_features, self.units)
        self.hidden_layer = nn.Linear(self.units + self.embedding_dim, self.units + self.embedding_dim)
        self.output_layer = nn.Linear(self.units + self.embedding_dim, self.out_features)

    def forward(self, x, x_cat=None):
        """Add categorical data"""
        if x_cat is not None:
            # x = torch.cat((x, x_cat), dim=1)
            print('x.shape:', x.size())
            print('x_cat.shape before flatten:', x_cat.size())
            x_out = self.embedding(x_cat).view((-1, x_cat.shape[0]))#.view((1, -1))
            print('x_cat after embedding:', x_cat.size()) # shape: (1, 6)
            x_out = self.flatten(x_out)
            x_out = F.relu(self.emb_input(x_out))
            print('x_out.size():', x_out.size())
            x_out = self.emb_output(x_cat)#.view(x_cat.shape[0], -1)
            print('x_cat.shape from self.emb_output():', x_cat.size())
            print()
            # x = torch.cat((x, x_cat), dim=1)

        x = self.flatten(x)
        print('x.shape after second flatten:', x.shape)
        x = F.relu(self.cont_input(x))
        print('x.shape after F.relu(self.cont_input():', x.shape)
        x = torch.cat((x, x_out), dim=1)
        print('x.shape after torch.cat:', x.shape)
        x = F.relu(self.hidden_layer(x))
        print('x.shape right before output layer:', x.shape)
        print()
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

    def fit_epochs(self, train_loader, valid_loader=None, epochs=5):
        for epoch in epochs:
            print(f'Epoch: <<< {epoch} >>>')
            self.fit_one_epoch(train_loader, valid_loader)

    def fit_one_epoch(
        self, 
        train_loader, 
        valid_loader=None, 
        use_cyclic_lr=False, 
        x_cat=None
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
            self._run_train_step(xtrain, ytrain, batch, size, scheduler, xtrain_cat)

        if valid_loader is not None:
            size = len(valid_loader)
            with torch.no_grad():
                for batch_val, data_val in enumerate(valid_loader):
                    xval = data_val['num_features']
                    if x_cat is not None:
                        xval_cat = data_val['cat_features']
                    yval = data_val['target']
                    self.evaluate(xval, yval, batch_val + 1, size, xval_cat)

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
        print(f'Val-Loss: {loss.item()} [{batch * len(x)}/{size}]')

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
        print(f'Train-Loss: {loss.item()} [{batch * len(x)}/{size}]')
        if scheduler:
            scheduler.step()

    def get_loss(self, loss_type='train'):
        if loss_type == 'train':
            return self.train_loss
        else:
            if len(self.valid_loss):
                return self.valid_loss

    def save_model(self):
        pass

    