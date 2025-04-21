from typing import List

import numpy as np
import pandas as pd


def standardized_mae(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> List[float]:
    """Compute standardised MAE (using the std deviation of each column)

    Args:
        df_true (pd.DataFrame): regression labels for QM9
        df_pred (pd.DataFrame): model predictions for QM9 (rows corresponding)

    Returns:
        List[float]: standard MAE per regression objective
    """
    df_true = df_true.to_numpy()
    std = np.std(df_true, axis=0)
    df_pred = df_pred.to_numpy()

    absolute_error = np.abs(df_pred - df_true)
    mae = absolute_error.mean(axis=0)
    std_mae = mae / std
    return std_mae
