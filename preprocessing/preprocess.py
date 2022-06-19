
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


def get_data(folder='train_files'):
    computer_name1 = 'gilbe'
    computer_name2 = 'Gilberto-BE'

    ROOT_PATH = f'c:/Users/{computer_name1}/Documents/TokyoData'
    train_df = pd.read_csv(f'{ROOT_PATH}/{folder}/stock_prices.csv')
    train_df['Date'] = pd.to_datetime(train_df['Date']) 
    train_df.set_index('Date', inplace=True)
    return train_df


class TextTransform:
    def __init__(self, tokens):
        self.token_np = np.array(list(tokens)) #if isinstance(tokens, (pd.DataFrame, pd.Series)) else tokens
        self.vocab = np.unique(self.token_np)
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

    def transform(self):
        self.idx = [self.w2idx[i] for i in self.token_np]
        return self.idx

    def inv_transform(self):
        return [self.idx2w[i] for i in self.idx]


def dataloader_by_stock(
    train_df, 
    sec_code, 
    batch_size=32,  
    continous_cols=['Close'],
    return_scaler=False
    ):
    df = train_df[train_df['SecuritiesCode'] == sec_code].drop(['SecuritiesCode'], axis=1)
    df = date_features(df)
    
    cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId']
    cont, cat = cont_cat_split(df, cat_cols=cat_cols)
    
    print('continuos shape:', cont.shape, '', 'categorical shape:', cat.shape)
    
    df_train_cat, df_val_cat = ts_split(cat)
    df_train, df_val = ts_split(cont)

    xtrain, ytrain = preprocess(df_train, 'Target', 1, continous_cols=continous_cols)
    xval, yval = preprocess(df_val, 'Target', 1, continous_cols=continous_cols)

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xval = scaler.transform(xval)

    train_loader = get_loader(
        x=xtrain, 
        y=ytrain, 
        batch_size=batch_size, 
        x_cat=df_train_cat.to_numpy()
        )
    val_dataloader = get_loader(
        x=xval, 
        y=yval, 
        batch_size=batch_size, 
        x_cat=df_val_cat.to_numpy()
        )
    if return_scaler:
        return train_loader, val_dataloader, scaler
    return train_loader, val_dataloader


def dataloader_test_by_stock(
    train_df, 
    sec_code, 
    transformer=None, 
    batch_size=32,  
    continous_cols=['Close'],
    target_col='Target'
    ):
    df = train_df[train_df['SecuritiesCode'] == sec_code].drop(['SecuritiesCode'], axis=1)
    df = date_features(df)

    # df['Target'] = df['Close'].shift().pct_change()
    # print(df.head())
    
    cat_cols = ['day_of_year', 'month', 'day_of_week', 'RowId']
    cont, cat = cont_cat_split(df, cat_cols=cat_cols)
    
    print('continuos shape:', cont.shape, '', 'categorical shape:', cat.shape)
    xtest = preprocess(cont, target_col, 1, continous_cols=continous_cols)

    if transformer is not None:
        xtest = transformer.transform(xtest)
    else:
        print('Notice that the transformer is None.')

    test_dataloader = get_predict_loader(
        x=xtest, 
        batch_size=batch_size, 
        x_cat=cat.to_numpy()
        )
    return test_dataloader


def ts_split(raw_data, train_size=0.75, val_size=None):

    train_sz = int(len(raw_data) * train_size)
    train_set = raw_data[ :train_sz]
    # if val_size and test_size:
    #     assert len(raw_data) == 100 * int(train_size * len(raw_data))
    valid_set = raw_data[train_sz: ]
    return train_set, valid_set    


def show_df(
    df, 
    show_info=True, 
    show_head=True, 
    show_tail=True, 
    dataframe_name='financials'
):
    print(f'<<< {dataframe_name} >>>')
    print(df.shape)
    if show_info:
        print(df.info())
    if show_head:
        print(df.head())
    if show_tail:
        print(df.tail())
    print('-.' * 80)
    
    
def date_features(df, date_col=None):
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    df.loc[:, 'day_of_year'] = df.index.dayofyear
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_of_week'] = df.index.day
    return  df


def preprocess(
    df, 
    target_col='Target',
    target_dim=1, 
    continous_cols=['Open', 'Close', 'High', 'Low', 'Volume']
    ):
    """
    -----------------
    Return x, y
    """
    rows = len(df)
    df['Volume'] = df['Volume'].astype(float)
    x = df.drop(target_col, axis=1) if target_col is not None else df
    x = x[continous_cols]
    for col in continous_cols:
        x[col] = x[col] * df['AdjustmentFactor']

    # if continous_cols:
    #     x[continous_cols] = x[continous_cols].pct_change()
    x = x.select_dtypes(include=[int, float]).dropna().to_numpy()
    if target_col is not None:
        
        y = df[target_col].dropna()
        y.plot()
        y = y.to_numpy().reshape(rows, target_dim)

        return x, y
    else:
        # x = df
        return x


def cont_cat_split(df, col_type=None, cat_cols=None):
    """
    Return transformer!!!
    """
    if cat_cols:
        cat = df[cat_cols]
        if 'RowId' in cat_cols:
           enc = OrdinalEncoder()
           # Transform to int??
           txt_transfom = TextTransform(cat['RowId'])
           cat.loc[:, ['RowId']] = txt_transfom.transform()
    elif col_type is not None:
        cat = df.select_dtypes(include=col_type)
    cat_cols = cat.columns
    cont = df.drop(cat_cols, axis=1)
    return cont, cat


class ToTorch(Dataset):

    def __init__(
            self,
            num_features,
            target,
            cat_features=None
            ):
        self.num_features = num_features
        self.target = target
        self.cat_features = cat_features

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        num_features = self.num_features[idx]
        target = self.target[idx]
        cat_features = self.cat_features[idx]

        if self.cat_features is not None:
            return {
                'num_features': torch.from_numpy(np.array(num_features)).float(), 
                'target': torch.from_numpy(np.array(target)).float(),
                'cat_features': torch.from_numpy(np.array(cat_features)).int()
                }

        return {
            'num_features': torch.from_numpy(np.array(num_features)).float(), 
            'target': torch.from_numpy(np.array(target)).float()
            }
    

def get_loader(x, y, batch_size, x_cat=None):
    # Return dict with {'num_features', 'targets'}
    if x_cat is not None:
        return DataLoader(ToTorch(x, y, x_cat), batch_size=batch_size)
    return DataLoader(ToTorch(x, y), batch_size=batch_size)


class TestLoader(Dataset):

    def __init__(
            self,
            num_features,
            cat_features=None
            ):
        self.num_features = num_features
        self.cat_features = cat_features

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        num_features = self.num_features[idx]
        # target = self.target[idx]
        cat_features = self.cat_features[idx]

        if self.cat_features is not None:
            return {
                'num_features': torch.from_numpy(np.array(num_features)).float(), 
                'cat_features': torch.from_numpy(np.array(cat_features)).int()
                }

        return {
            'num_features': torch.from_numpy(np.array(num_features)).float(), 
            }

def get_predict_loader(x, batch_size, x_cat=None):
    if x_cat is not None:
        return DataLoader(TestLoader(x, x_cat), batch_size=batch_size)
    return DataLoader(TestLoader(x), batch_size=batch_size)


if __name__ == '__main__':
    token_list = ['Prime Market', 'Prime Market', 'two', 'three', 'four', 'four', 'four']

    # token_list = [sent_tokenize(token_list)]
    print('raw tokens')
    print(token_list)
    txt_trans = TextTransform(token_list)
    print('Get indexes')
    print(txt_trans.transform())
    print()
    print('get dict with vocab')
    print(txt_trans.w2idx)
    print()
    print('inverse transform')
    print(txt_trans.inv_transform())
