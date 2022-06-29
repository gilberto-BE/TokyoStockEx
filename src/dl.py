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
from metrics.metrics import compute_metrics
import matplotlib.pyplot as plt
import numpy as np


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
        self.embedding_layer = nn.Embedding(self.no_embedding, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.units)
        self.out = nn.Linear(self.units, 1)
        
    def forward(self, x):
        x = F.relu(self.embedding_layer(x))
        print('x.shape after F.relu(embedding_layer(k)):', x.shape)
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
    """
    Need to check all dimensions
    """
    def __init__(self, units=10):
        super(GatedLinearUnit, self).__init__()
        self.units = units
        self.layer = nn.Linear(self.units, self.units)
        self.output = nn.Linear(self.units, self.units)

    def forward(self, x):
        return self.layer(x) @ F.sigmoid(self.output(x))


class GatedResidualNetwork(nn.Module):
    """
    Need to check all dimensions
    """
    def __init__(self, in_features, out_features, dropout=0.1, units=50):
        super(GatedResidualNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.layer1 = nn.Linear(self.in_features, self.units)
        self.layer2 = nn.Linear(self.in_features, self.out_features)
        self.dropout = nn.Dropout(dropout)
        self.gated_linear_unit = GatedLinearUnit(self.in_features)
        self.layer_norm = nn.LayerNorm(self.in_features)
        self.layer2 = nn.Linear(self.units, self.units)

    def forward(self, inputs):
        x = F.elu(self.layer1(x))
        x = self.layer2(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.layer2(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class FFResNetBlock(nn.Module):
    """
    TODO:
    Replace, ModuleList with regular list and sequential class!
    """
    def __init__(self, in_features, out_features, n_layers=3):
        super(FFResNetBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.layers = nn.ModuleList(nn.Linear(self.in_features, self.out_features) for _ in range(self.n_layers))
        self.layer_norm = nn.LayerNorm(self.out_features)

    def forward(self, x):
        res = x
        for L in self.layers:
            x = F.relu(self.layer_norm(L(x)))
        return x + res


class NeuralBlock(nn.Module):

    def __init__(self, units, categorical_dim, res_layers=3, res_blocks=2):
        super(NeuralBlock, self).__init__()

        self.in_features = units + categorical_dim
        self.out_features = units + categorical_dim
        self.res_layers = res_layers
        self.res_blocks = res_blocks
        self.resnet_block = nn.ModuleList(
            FFResNetBlock(self.in_features, self.out_features, self.res_layers) for _ in range(self.res_blocks)
            )
        self.fwr_layer = nn.Linear(self.in_features, self.in_features)
        self.output = nn.Linear(self.in_features, self.out_features)
        self.backcast_layer = nn.Linear(self.in_features, self.in_features)
        self.back_cast = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        res = x
        x = torch.real(torch.fft.fft2(x))
        for r_block in self.resnet_block:
            x = r_block(x)
        x_res = res - x

        x = torch.real(torch.fft.ifft2(x))
        x = F.relu(self.fwr_layer(x))
        x = self.output(x)

        x_res = torch.real(torch.fft.ifft2(x_res))
        x_res = F.relu(self.backcast_layer(x_res))
        x_res = self.back_cast(x_res)
        return x_res, x


class NeuralStack(nn.Module):
    def __init__(
        self, 
        n_blocks, 
        units, 
        categorical_dim, 
        output_dim=1,
        res_layers=3,
        res_blocks=2
        ):
        super(NeuralStack, self).__init__()
        self.n_blocks = n_blocks
        self.units = units
        self.categorical_dim = categorical_dim
        self.output_dim = output_dim
        self.res_layers = res_layers
        self.res_blocks = res_blocks

        self.blocks = nn.ModuleList(
            [
                NeuralBlock(self.units, self.categorical_dim, self.res_layers, self.res_blocks) for _ in range(self.n_blocks)
            ])
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(self.units + self.categorical_dim, self.output_dim) for _ in range(self.n_blocks)
            ])

    def forward(self, x):
        predictions = 0
        x_back = x
        for block_nr, block in enumerate(self.blocks):
            x_back, pred = block(x_back)
            predictions += self.output_layers[block_nr](pred)        
        return x_back, predictions


class NeuralNetwork(nn.Module):
    """
    TODO:
    1) IMPLEMENT PREPROCESSING AS A LAYER IN THE NETWORK.
    """
    
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
        n_stacks=8,
        pooling_sizes=3,
        res_layers=3,
        res_blocks=2
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
        self.pooling_sizes = pooling_sizes
        self.res_layers = res_layers
        self.res_blocks = res_blocks

        if no_embedding and emb_dim:
            self.embedding_layer = nn.Embedding(self.no_embedding, self.emb_dim)
            self.embedding_to_hidden = nn.Linear(self.emb_dim, self.units)
            self.embedding_output = nn.Linear(self.units, self.out_features)  
            
        # if adding cont vars subtract from in-features
        self.cont_input = nn.Linear(self.in_features, self.units)
        self.dropout = nn.Dropout(dropout)
        self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        self.stacks = nn.ModuleList([
            NeuralStack(
                    self.n_blocks, 
                    self.units, 
                    self.categorical_dim, 
                    self.out_features, 
                    self.res_layers, 
                    self.res_blocks) for _ in range(self.n_stacks)
                    ])

    def forward(self, x, x_cat=None):
        """
        TODO:
        * Add residual connictions.
        """
        # if x_cat is not None:
        x_cat = x_cat.to(torch.int64)
        # print('x_cat.shape before embedding:', x_cat.shape)
        
        x_cat = self.embedding_layer(x_cat)
        # print('x_cat after embedding:', x_cat.shape)
        
        # x_cat = torch.squeeze(torch.real(torch.fft.fft2(x_cat)))
        # print('TEST UNSQUEEZE ON x_cat: ', torch.unsqueeze(x_cat, 0).shape)
        # print('x_cat after fft2:', x_cat.shape)
        
        x_cat = F.relu(self.embedding_to_hidden(x_cat))
        # print('x_cat after relu + linear:', x_cat.shape)
        
        x_cat = F.relu(self.embedding_output(x_cat))
        # print('Shape of x_cat before cat:', x_cat.shape)

        x = torch.real(torch.fft.fft2(x))
        # print('x.shape after fft2:', x.shape)
        x = self.pooling_layer(x)
        # print('x.shape after pooling:', x.shape)
        x = F.relu(self.cont_input(x))
        # print('x.shape after first cont layer:', x.shape)
        # We are adding cat vars

        x = torch.cat((x, x_cat.view((x_cat.size(0), -1))), dim=1)
        # print('x.shape after torch.cat:', x.shape)
        x = self.dropout(x)

        tot_preds = 0
        block_input = x
        for S in self.stacks:
            block_input, pred = S(block_input)
            tot_preds += pred
        return tot_preds


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer_name:str='adam', 
        lr: float = 3e-6, 
        loss_fn_name: str = 'mse',
        weight_decay: float = 0.0,
        scale_ts: bool = False

        ):

        """
        TODO
        1) Add early stopping
        """
        self.lr = lr
        self.optimizer_name=optimizer_name
        self.loss_fn_name = loss_fn_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}-device")
        self.model = model.to(self.device)
        self.weight_decay = weight_decay
        self.scale_ts = scale_ts

        if self.loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()

        if self.optimizer_name.lower() == 'rmsprop': 
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), 
                self.lr, 
                weight_decay=self.weight_decay
                )

        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                self.lr, 
                weight_decay=self.weight_decay
                )

    def fit_epochs(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader:torch.utils.data.DataLoader=None, 
        use_cyclic_lr:bool=False, 
        epochs:int=5, 
        x_cat=None
        ):
        train_loss = []
        valid_loss = []
        train_mae = []
        valid_mae = []
        best_valid_loss = 1_000_000
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            'min',
            factor=0.5, 
            patience=5, 
            threshold=1e-3, 
            verbose=True
            )
        for epoch in range(epochs):
            result = self.fit_one_epoch(train_loader, valid_loader, use_cyclic_lr, x_cat=x_cat)
            scheduler.step(result['avg_loss_val'])
            if epoch % 10  == 0:
                # print(f'Epoch: <<< {epoch} >>>')
                print(
                    f"""
                    Average train loss: {result["avg_loss_train"]} | 
                    Train-Mae: {result["train_mae"]} |

                    Average val loss: {result["avg_loss_val"]}|
                    Val-Mae: {result["val_mae"]}
                    """
                    )
                # print('.' * 20, f'End of epoch {epoch}','.' * 20)
            if result['avg_loss_val'] < best_valid_loss:
                """
                SAVE THE BEST MODEL BASED ON BEST VALID LOSS
                """
                pass
            train_loss.append(result["avg_loss_train"])
            valid_loss.append(result["avg_loss_val"].cpu().detach().numpy())
            train_mae.append(result["train_mae"])
            valid_mae.append(result["val_mae"])

        return train_loss, train_mae, valid_loss, valid_mae

    def fit_one_epoch(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        valid_loader:torch.utils.data.DataLoader=None, 
        use_cyclic_lr:bool=False, 
        x_cat:bool=None
        ):
        if use_cyclic_lr:
            """add parameters for scheduler to constructor."""
        self.model.train()
        pred_train, avg_loss_train, train_metrics = self.run_train_step(train_loader, x_cat=x_cat)

        if valid_loader is not None:
            pred_val, avg_loss_val, val_metrics = self.run_val_step(valid_loader, x_cat=x_cat)

        result = {
            'pred_train': pred_train, 
            'avg_loss_train': avg_loss_train, 
            'pred_val': pred_val, 
            'avg_loss_val': avg_loss_val,
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae']
            }
        return result

    def run_train_step(self, train_loader, loss_every=100, x_cat=True):
        running_loss = 0.0
        last_loss = 0.0
        for batch, data in enumerate(train_loader):
            x = data['num_features'].to(self.device)
            y = data['target'].to(self.device)
            self.optimizer.zero_grad()

            """
            TODO: CHECK SCALING ISSUES
            USE MAX-SCALING???
            """
            # if self.scale_ts:
            #     scaling_factor = torch.abs(x.max(1)[0].unsqueeze(1))
            #     x = x/scaling_factor

            if x_cat is not None:
                x_cat = data['cat_features'].to(self.device)
                pred = self.model(x, x_cat).to(self.device)
            else:
                pred = self.model(x).to(self.device)
            
            # if self.scale_ts:
            #     pred = pred * scaling_factor
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if batch % loss_every == 0:
                last_loss = running_loss/loss_every
                running_loss = 0.0

        train_metrics = compute_metrics(pred, y)
        return pred, last_loss, train_metrics

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
        val_metrics = compute_metrics(pred, y)
        return pred, avg_loss, val_metrics

    def save_model(self, model, path='./trained_model.pt'):
        torch.save(model, path)

    def load_model(self, path='./notebooks/trained_model.pt'):
        model = torch.load(path)
        return model.eval()


def plot_loss(line1, line2, title1='Train-MAE', title2='Valid-MAE'):
    fig, ax = plt.subplots()
    training_mae, = ax.plot(range(len(line1)), line1, label=title1)
    val_mae, = ax.plot(range(len(line2)), line2, label=title2)
    plt.xlabel('Epochs')
    ax.legend(handles=[training_mae, val_mae])
    plt.show()


if __name__  == '__main__':
    # Test new networks
    tgt = torch.tensor(np.array(range(10)) * 0.01)
    feat = torch.tensor(np.array(range(100)).reshape(10, 10))
    print(feat.shape)
    print(feat)
    print(tgt.shape)
    print(tgt)
