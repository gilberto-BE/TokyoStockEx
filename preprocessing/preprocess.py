
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


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
    # Check if index is datetime.
#     if isinstance(df, pd.core.series.Series):
#         df = pd.DataFrame(df, index=df.index)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    df.loc[:, 'day_of_year'] = df.index.dayofyear
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_of_week'] = df.index.day
#     df.loc[:, 'hour'] = df.index.hour
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
    y = df[target_col].dropna().to_numpy().reshape(rows, target_dim)
    x = df.drop(target_col, axis=1)
    x = x[continous_cols]

    if continous_cols:
        x[continous_cols] = x[continous_cols].pct_change()
    x = x.select_dtypes(include=[int, float]).dropna().to_numpy()
    return x, y


def cont_cat_split(df, col_type=None, cat_cols=None):
    """
    Return transformer!!!
    """
    if cat_cols:
        cat = df[cat_cols]
        if 'RowId' in cat_cols:
           enc = OrdinalEncoder()
           # Transform to int??
           cat['RowId'] = enc.fit_transform(df['RowId'].to_numpy().reshape(-1, 1))
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