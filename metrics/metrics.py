
import numpy as np
import pandas as pd
from torchmetrics import (
    MeanSquaredError, 
    # SymmetricMeanAbsolutePercentageError,
    R2Score,
    MeanAbsoluteError,
    # MeanAbsolutePercentageError
    )


def calc_spread_return_sharpe(
    df: pd.DataFrame, portfolio_size: int = 200, 
    toprank_weight_ratio: float = 2
    ) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(
        df, portfolio_size, toprank_weight_ratio
        ):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        
        purchase = (
            df.sort_values(by='Rank')['Target'][:portfolio_size] * weights
            ).sum() / weights.mean()

        short = (
            df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights
            ).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


def metrics(ytrue, ypred):
    mse = MeanSquaredError()
    # smape = SymmetricMeanAbsolutePercentageError()
    r2_score = R2Score()
    mae = MeanAbsoluteError()
    # wmape = MeanAbsolutePercentageError()

    metrics_dict = {
        'mse': mse(ytrue, ypred).item(), 
        # 'smape': smape(ytrue, ypred),
        # 'r2_score': r2_score(ytrue, ypred).item(),
        'mae': mae(ytrue, ypred).item(),
        # 'mape': wmape(ytrue, ypred).item()
        }

    return metrics_dict