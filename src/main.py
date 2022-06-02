import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
pl.utilities.seed.seed_everything(seed=42)
import numpy as np
import pandas as pd

import sys, os
source_path = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(source_path)
source_path = os.path.join(os.getcwd(), os.pardir, 'preprocessing')
sys.path.append(source_path)
source_path = os.path.join(os.getcwd(), os.pardir, 'metrics')
sys.path.append(source_path)
import matplotlib.pyplot as plt

from dl import NeuralNetwork, Trainer
from preprocess import (
    show_df, 
    date_features, 
    preprocess, 
    ToTorch, 
    get_loader, 
    ts_split,
    cont_cat_split
)
from metrics import calc_spread_return_sharpe
import torch
from sklearn.impute import SimpleImputer


def read_data(root_path=None):
    computer_names = dict(computer_name1 = 'gilbe', computer_name2 = 'Gilberto-BE')
    alt1 = computer_names["computer_name1"]
    alt2 = computer_names["computer_name2"]
    prompt = input(f'Choose 1 for {alt1} or 2 for {alt2}')
    if prompt == '1':
        computer_name = alt1
    elif prompt == '2':
        computer_name = alt2

    if root_path is not None:
        ROOT_PATH = root_path
    else:
        ROOT_PATH = f'c:/Users/{computer_name}/Documents/TokyoData'

    '/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv'
    '/train_files/trades.csv'

    train_df = pd.read_csv(f'{ROOT_PATH}/train_files/stock_prices.csv')
    train_df['Date'] = pd.to_datetime(train_df['Date']) 
    train_df.set_index('Date', inplace=True)
    # train_df = date_features(train_df)

    print('Stock-data:')
    print(train_df.head(2))

    train_options = pd.read_csv(f'{ROOT_PATH}/train_files/options.csv', low_memory=False)
    train_financials = pd.read_csv(f'{ROOT_PATH}/train_files/financials.csv', low_memory=False)
    train_trades = pd.read_csv(f'{ROOT_PATH}/train_files/trades.csv', low_memory=False)

    data_dict = {
        'train_df': train_df, 
        'train_options': train_options, 
        'train_financials': train_financials, 
        'train_trades': train_trades
        }

    return data_dict


def create_loaders(train_df, ):

    print('Raw Time Series data shape:', train_df.shape)
    print('No Unique Securities code:', train_df['SecuritiesCode'].nunique())

    """SELECT ONE STOCK"""

    df_1301 = train_df[train_df['SecuritiesCode'] == 1301].drop(['SecuritiesCode', 'Volume'], axis=1)

    print('df_1301.head()')
    print(df_1301.head(2))
    print(df_1301.info())

    df_1301 = date_features(df_1301)

    """ 
    Add RowId as extra catcol.
    """
    # cont, cat = cont_cat_split(df_1301, 'int64')
    cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId']
    cont, cat = cont_cat_split(df_1301, cat_cols=cat_cols)
    print('categorical shape:', cat.shape)

    df_train_cat, df_val_cat = ts_split(cat)
    df_train, df_val = ts_split(cont)

    xtrain, ytrain = preprocess(df_train, 'Target', 1, continous_cols=['Close'])
    xval, yval = preprocess(df_val, 'Target', 1, continous_cols=['Close'])

    print('xtrain.shape:', xtrain.shape)
    print(xtrain[:5])
    print()
    print('ytrain.shape:', ytrain.shape)
    print(ytrain[:5])
    print('df_train_cat.shape:', df_train_cat.shape)
    print(df_train_cat.head())

    """ xtrain and df_train_cat have different shapes!!!!!"""

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    batch_size = 64
    train_dataloader = get_loader(x=xtrain, y=ytrain, batch_size=batch_size, x_cat=df_train_cat.to_numpy())
    val_dataloader = get_loader(x=xval, y=yval, batch_size=batch_size, x_cat=df_val_cat.to_numpy())
    return train_dataloader, val_dataloader


CAT_FEATURES = cat.shape[1]
print('CAT_FEATURES:', CAT_FEATURES)
EMBEDDING_DIM = 10
NO_EMBEDDING = 2 * len(df_train_cat)
# cat_features = cat_features * embedding_dim
# print('in_features:', xtrain.shape[1] + cat_features)

model = NeuralNetwork(
    in_features=xtrain.shape[1], 
    units=500,
    out_features=1, 
    categorical_dim=CAT_FEATURES,
    no_embedding=NO_EMBEDDING, 
    emb_dim=EMBEDDING_DIM
)

print(model)

trainer = Trainer(model, lr=3.3e-6)
trainer.fit_epochs(
    train_dataloader, 
    val_dataloader, 
    use_cyclic_lr=True, 
    x_cat=True, 
    epochs=25
)


if __name__ == '__main__':
    read_data()