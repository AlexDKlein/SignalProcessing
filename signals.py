import numpy as np
import scipy
import scipy.signal
from .util import chunk, pad_axis

def nyquist_frequency(X, t=1.0):
    if np.iterable(t):
        t = np.ptp(t, axis=-1).mean()
    sampling_rate = (X.shape[-1]) / t
    return sampling_rate / 2

def sampling_rate(a, t=None):
    """
        Return the mean sampling rate for a group of time-series measurements.
        Parameters
        ===========
        a: array-like
            Array of time-series measurements
        t: array-like or number or None, default=None
            If number, the average time per set of measurements.
            If array-like, the times at which each value in `a` was measured.
            If None, defaults to a.shape[-1]
    """
    if np.iterable(t):
        t_ptp = np.ptp(t, axis=-1)
        avg_time = t_ptp.mean()
    else:
        avg_time = a.shape[-1] if t is None else t 
    fs = a.shape[-1] / avg_time
    return fs

