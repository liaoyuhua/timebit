"""
Use time series as pandas dataframe.

basic time series data structure.
"""
from typing import Union, Optional, List
import pandas as pd
import numpy as np


class Series:
    """
    A time series data structure which can be used like pandas.DataFrame.
    """

    def __init__(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> None:
        pass

    @classmethod
    def from_series(cls):
        pass

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        time_idx: Optional[str] = None,
        group_ids: Optional[Union[str, List[str]]] = None,
        target: Optional[Union[str, List[str]]] = None,
        covars: Optional[Union[str, List[str]]] = None,
        freq: Optional[Union[str, int]] = None,
        fill_missing_dates: Optional[bool] = False,
    ):
        pass

    @classmethod
    def from_pikle(cls):
        pass

    @property
    def freq(self):
        """The frequency of the series."""
        return self._freq

    @property
    def dtype(self):
        """The dtype of the series' values."""
        return self._xa.values.dtype

    def astype(self, dtype: Union[str, np.dtype]) -> "TimeSeries":
        """
        Converts this series to a new series with desired dtype.
        Parameters
        ----------
        dtype
            A NumPy dtype (np.float32 or np.float64)
        Returns
        -------
        TimeSeries
            A TimeSeries having the desired dtype.
        """
        return self.__class__(self._xa.astype(dtype))

    def start_time(self) -> Union[pd.Timestamp, int]:
        """
        Start time of the series.
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the first time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[0]

    def end_time(self) -> Union[pd.Timestamp, int]:
        """
        End time of the series.
        Returns
        -------
        Union[pandas.Timestamp, int]
            A timestamp containing the last time of the TimeSeries (if indexed by DatetimeIndex),
            or an integer (if indexed by RangeIndex)
        """
        return self._time_index[-1]

    def values(self, copy: bool = True, sample: int = 0) -> np.ndarray:
        """
        Return a 2-D array of shape (time, component), containing this series' values for one `sample`.
        Parameters
        ----------
        copy
            Whether to return a copy of the values, otherwise returns a view.
            Leave it to True unless you know what you are doing.
        sample
            For stochastic series, the sample for which to return values. Default: 0 (first sample).
        Returns
        -------
        numpy.ndarray
            The values composing the time series.
        """
        raise_if(
            self.is_deterministic and sample != 0,
            "This series contains one sample only (deterministic),"
            "so only sample=0 is accepted.",
            logger,
        )
        if copy:
            return np.copy(self._xa.values[:, :, sample])
        else:
            return self._xa.values[:, :, sample]

    def head(
        self, size: Optional[int] = 5, axis: Optional[Union[int, str]] = 0
    ) -> "TimeSeries":
        """
        Return a TimeSeries containing the first `size` points.
        Parameters
        ----------
        size : int, default 5
               number of points to retain
        axis : str or int, optional, default: 0
               axis along which to slice the series
        Returns
        -------
        TimeSeries
            The series made of the first `size` points along the desired `axis`.
        """

        axis_str = self._get_dim_name(axis)
        display_n = min(size, self._xa.sizes[axis_str])

        if axis_str == self._time_dim:
            return self[:display_n]
        else:
            return self.__class__(self._xa[{axis_str: range(display_n)}])

    def tail(
        self, size: Optional[int] = 5, axis: Optional[Union[int, str]] = 0
    ) -> "TimeSeries":
        """
        Return last `size` points of the series.
        Parameters
        ----------
        size : int, default: 5
            number of points to retain
        axis : str or int, optional, default: 0 (time dimension)
            axis along which we intend to display records
        Returns
        -------
        TimeSeries
            The series made of the last `size` points along the desired `axis`.
        """

        axis_str = self._get_dim_name(axis)
        display_n = min(size, self._xa.sizes[axis_str])

        if axis_str == self._time_dim:
            return self[-display_n:]
        else:
            return self.__class__(self._xa[{axis_str: range(-display_n, 0)}])

    def concatenate(
        self,
        other: "TimeSeries",
        axis: Optional[Union[str, int]] = 0,
        ignore_time_axis: Optional[bool] = False,
        ignore_static_covariates: bool = False,
        drop_hierarchy: bool = True,
    ) -> "TimeSeries":
        pass

    def slice(
        self, start_ts: Union[pd.Timestamp, int], end_ts: Union[pd.Timestamp, int]
    ):
        pass

    def shift(self, n: int) -> "TimeSeries":
        pass

    def append(self, other: "TimeSeries") -> "TimeSeries":
        pass

    def stack(self, other: "TimeSeries") -> "TimeSeries":
        pass

    def _fill_missing_timestep(self):
        pass

    def to_dataframe(self):
        pass

    def to_csv(self, *args, **kwargs):
        pass

    def to_pickle(self, path: str, protocol: int = pickle.HIGHEST_PROTOCOL):
        pass

    def plot(
        self,
        new_plot: bool = False,
        central_quantile: Union[float, str] = 0.5,
        low_quantile: Optional[float] = 0.05,
        high_quantile: Optional[float] = 0.95,
        default_formatting: bool = True,
        label: Optional[Union[str, Sequence[str]]] = "",
        *args,
        **kwargs,
    ):
        pass

    def __len__(self):
        pass

    def __getitem(self):
        pass


class TimeDataSet:
    pass
