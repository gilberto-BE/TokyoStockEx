
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer



def ts_split(raw_data, train_size=0.75, val_size=None):

    train_sz = int(len(raw_data) * train_size)
    train_set = raw_data.iloc[ :train_sz]
    # if val_size and test_size:
    #     assert len(raw_data) == 100 * int(train_size * len(raw_data))
    valid_set = raw_data.iloc[train_sz: ]

    valid_set = raw_data.iloc[train_size : ]
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
    num_targets=1, 
    continous_cols=['Open', 'Close', 'High', 'Low', 'Volume']
    ):
    """

    -----------------
    Return x, y
    """
    y = df[target_col].dropna().to_numpy().reshape(len(df), num_targets)
    x = df.drop(target_col, axis=1)

    if continous_cols:
        x[continous_cols] = x[continous_cols].pct_change()
    x = x.select_dtypes(include=[int, float]).dropna().to_numpy()

    return x, y


class ToTorch(Dataset):

    def __init__(
            self,
            features,
            target
            ):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]
        return {
            'features': torch.from_numpy(np.array(features)).float(), 
            'target': torch.from_numpy(np.array(target)).float()
            }
    

def get_loader(x, y, batch_size):
    # Return dict with {'features', 'targets'}
    return DataLoader(ToTorch(x, y), batch_size=batch_size)