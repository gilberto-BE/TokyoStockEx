
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer


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


def preprocess(df, diff_cols=['Open', 'Close', 'High', 'Low', 'Volume']):
    if diff_cols:
        df[diff_cols] = df[diff_cols].pct_change()
    df = df.select_dtypes(include=[int, float])

    return df



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