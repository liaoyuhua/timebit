"""
Use time series as pandas dataframe.

basic time series data structure.
"""
from typing import Union, Optional, List
import pandas as pd
import numpy as np


class Series:
    """
    A time series data structure which can be used like DataFrame.
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> None:
        pass
