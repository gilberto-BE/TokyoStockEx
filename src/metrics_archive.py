from torchmetrics import (
    MeanSquaredError, 
    # SymmetricMeanAbsolutePercentageError,
    R2Score
    )


def metrics(ytrue, ypred):
    mse = MeanSquaredError()
    # smape = SymmetricMeanAbsolutePercentageError()
    r2_score = R2Score()

    metrics_dict = {
        'mse': mse(ytrue, ypred), 
        # 'smape': smape(ytrue, ypred),
        'r2_score': r2_score(ytrue, ypred)
        }

    return metrics_dict