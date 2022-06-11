
import torch
import torch.nn as nn
torch.manual_seed(0)

# import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
# pl.utilities.seed.seed_everything(seed=42)
from torch import nn, Tensor
import math
from metrics import metrics
import matplotlib.pyplot as plt


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
        super(PositionalEncoding, self).__init__()
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


class GatedLinearUnit(nn.Module):
    def __init__(self):
        super(GatedLinearUnit, self).__init__()

    def forward(self, x):
        pass


class NeuralBlock(nn.Module):

    def __init__(self, units, categorical_dim, dropout=0.1):
        super(NeuralBlock, self).__init__()

        self.in_features = units + categorical_dim
        self.out_features = units + categorical_dim
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Linear(self.in_features, self.out_features)
        self.layer2 = nn.Linear(self.in_features, self.out_features)
        self.layer3 = nn.Linear(self.in_features, self.out_features)
        self.layer4 = nn.Linear(self.in_features, self.out_features)
        self.output = nn.Linear(self.in_features, self.out_features)
        self.res_output = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        res = x
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        # x = x + res
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = x + res
        x = F.relu(self.layer3(x))
        x = self.dropout(x)
        # x = x + res
        x = F.relu(self.layer4(x))
        x = self.output(x)
        x = x + res
        x_res = self.res_output(x - res)
        return x_res, x


class NeuralStack(nn.Module):
    def __init__(
        self, 
        n_blocks, 
        units, 
        categorical_dim, 
        output_dim=1
        ):
        super(NeuralStack, self).__init__()

        self.stacks = nn.ModuleList([
            NeuralBlock(units, categorical_dim) for _ in range(n_blocks)
            ]
            )
        # Create list of outputs
        self.output_layers = nn.ModuleList([
            nn.Linear(units + categorical_dim, output_dim) for _ in range(n_blocks)
            ])

    def forward(self, x):
        self.outputs = []
        for block in self.stacks:
            x_back, pred = block(x)
            x = x_back
            for o_layer in self.output_layers:
                stack_pred = o_layer(pred)
            self.outputs.append(stack_pred)
        tot_pred = 0
        for o in self.outputs:
            tot_pred = tot_pred + o
        return x_back, tot_pred


class NeuralNetwork(nn.Module):
    
    def __init__(
        self, 
        in_features, 
        out_features,
        categorical_dim=3, 
        units=512, 
        no_embedding=None, 
        emb_dim=None,
        dropout=0.1,
        n_blocks=10,
        n_stacks=8
        ):

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
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks

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
        self.dropout = nn.Dropout(dropout)
        self.stacks = nn.ModuleList([
            NeuralStack(
                    self.n_blocks, 
                    self.units, 
                    self.categorical_dim, 
                    self.out_features) for _ in range(self.n_stacks)
                    ])

    def forward(self, x, x_cat=None):
        """
        TODO:
        * Add residual connictions.

        """
        if x_cat is not None:
            x_cat = x_cat.to(torch.int64)
            emb_residual = x_cat
            x_cat = self.embedding(x_cat)
            # x_cat = self.position_enc(x_cat)
            x_cat = torch.squeeze(torch.real(torch.fft.fft2(x_cat)))
            x_cat = self.embedding_to_hidden(x_cat)
            x_cat = F.relu(x_cat)
            x_cat = self.dropout(x_cat)
            x_cat = F.relu(self.embedding_output(x_cat))
            x_cat = self.dropout(x_cat)
        # cont_residual = x
        x = torch.real(torch.fft.fft(x))
        x = F.relu(self.cont_input(x))
        # x = x + cont_residual.view(x.shape)
        x = torch.cat((x, x_cat.view((x_cat.shape[0], -1))), dim=1)

        stack_outs = []
        for n_stack in self.stacks:
            x, pred = n_stack(x)
            stack_outs.append(pred)
        tot_pred = 0
        for o in stack_outs:
            tot_pred = tot_pred + o
        return tot_pred

        # out_stack = []
        # for _ in range(self.n_stacks):
        #     x, pred = self.neural_stacks(x)
        #     out_stack.append(self.output_layer(pred))
        # # out = self.output_layer(out)
        # tot_pred = 0
        # for o in out_stack:
        #     tot_pred = tot_pred + o
        # return tot_pred

        # res = x
        # out_block = []
        # # for _ in range(self.n_stacks):
        # for _ in range(self.n_blocks):
        #     x = self.nn_block(x)
        #     output_from_block = self.output_layer(x)
        #     out_block.append(output_from_block)
        # x = x + res
        # x = self.output_layer(x)
        # for o in out_block:
        #     x = x + o
        # return x

    def nn_block(self, x):
        res = x
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = x + res
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = x + res
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = x + res
        x = F.relu(self.hidden_layer(x))
        x_res = F.relu(self.hidden_layer(x - res))
        return x_res, x
    
    def neural_stacks(self, x):
        out_block = []
        for _ in range(self.n_blocks):
            x_res, pred = self.nn_block(x)
            x = x_res
            # output_from_block = pred
            out_block.append(pred)
        tot_out = 0
        for o in out_block:
            tot_out = tot_out + o
        return x_res, tot_out


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer_name='adam', 
        lr=3e-6, 
        loss_fn_name='mse'
        ):

        """
        TODO
        1) Add early stopping
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
        self.model.to(self.device)

        if self.loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()

        if self.optimizer_name.lower() == 'rmsprop': 
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)

        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

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

    def fit_epochs(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader:torch.utils.data.DataLoader=None, 
        use_cyclic_lr:bool=False, 
        epochs:int=5, 
        x_cat=None
        ):
        for epoch in range(epochs):
            print(f'Epoch: <<< {epoch} >>>')
            pred_train, avg_loss_train, pred_val, avg_loss_val = self.fit_one_epoch(
                train_loader, 
                valid_loader, 
                use_cyclic_lr, 
                x_cat=x_cat
                )
            print(f'Average train loss: {avg_loss_train} | Average val loss: {avg_loss_val}')
            print('.' * 20, f'End of epoch {epoch}','.' * 20)
            self.train_loss.append(avg_loss_train)
            self.valid_loss.append(avg_loss_val.cpu().detach().numpy())

        fig, ax = plt.subplots()
        train_loss, = ax.plot(range(epochs), self.train_loss, label='Train-loss')
        val_loss, = ax.plot(range(epochs), self.valid_loss, label='Valid-loss')
        # plt.legend('Valid loss.')
        plt.xlabel('Epochs')
        ax.legend(handles=[train_loss, val_loss])
        plt.show()

    def plot_loss(self, train_loss, val_loss):
        pass

    def fit_one_epoch(
        self, 
        train_loader, 
        valid_loader=None, 
        use_cyclic_lr=False, 
        x_cat=None
        ):
        if use_cyclic_lr:
            """add parameters for scheduler to constructor."""
        self.model.train()
        pred_train, avg_loss_train = self.run_train_step(train_loader, x_cat=x_cat)

        if valid_loader is not None:
            pred_val, avg_loss_val = self.run_val_step(valid_loader, x_cat=x_cat)
        return pred_train, avg_loss_train, pred_val, avg_loss_val

    def run_train_step(self, train_loader, loss_every=20, x_cat=True):
        running_loss = 0.0
        last_loss = 0.0
        for batch, data in enumerate(train_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = self.model(x, x_cat).to(self.device)
            else:
                pred = self.model(x).to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if batch % loss_every == 0:
                last_loss = running_loss/loss_every
                running_loss = 0.0

        train_metrics = metrics(pred, y)
        print(f'Train metrics: {train_metrics}')
        return pred, last_loss

    def run_val_step(self, valid_loader, x_cat=True):
        running_loss = 0.0
        self.model.eval()
        # with torch.no_grad:
        for batch, data in enumerate(valid_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = self.model(x, x_cat).to(self.device)
            else:
                pred = self.model(x)
            loss = self.loss_fn(pred, y)
            running_loss += loss
        avg_loss = running_loss/(batch + 1)
        val_metrics = metrics(pred, y)
        print(f'Validation metrics: {val_metrics}')
        return pred, avg_loss

    def get_error_loss(self, loss_type='train'):
        """
        Get error-loss for training and validation sets.
        """
        if loss_type == 'train':
            return running_loss
        else:
            if len(self.valid_loss) > 0:
                return self.valid_loss

    def save_model(self):
        pass

    