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

def stft(a, nperseg=256, noverlap=None, window='hann', detrend=True, center=False, pad=True, norm=False, axis=-1):
    """
        Perform a Short-Time Fourier Transform along the specified axis. 
        Equivalent to scipy.signal.stft except slightly faster (~10%) with modified usage/behavior.
        Parameters
        ===========
        a: array-like
            Array of time-series measurements
        nperseg: int, default=256
            Length of each segment
        noverlap: int or None, default=None
            Number of overlapping items in each segment. If None, defaults to nperseg // 2.
        window: str or None, default='hann'
            The window to use. If set to None, values in each segment are weighted equally.
            See scipy.signal.get_window documentation for more options.
        detrend: bool, default=True
            Whether to detrend the signal by subtracting its mean.
        center: bool, default=False
            Whether to center the signal. If True, the signal is zero-padded by nperseg // 2 on each end.
        pad: bool, default=True
            Whether to pad the signal to fit an integer number of segments. If True, the end of the signal is zero padded.
        norm: bool, default=False
            Whether to use the 'ortho' norm during the fourier transform. See np.fft for details.
        axis: int, default=-1
            The axis along which the fourier transform is applied.

        Returns
        ==========
        output: np.ndarray
            The transformed array. The first axis corresponds to the segment times.
    """
    if noverlap is None:
        noverlap = nperseg // 2
        
    if window is None:
        window = np.ones(nperseg, dtype=a.dtype)
    elif window == 'hann':
        window = np.hanning(nperseg + 1)[:-1]
    else:
        window = scipy.signal.get_window(window, nperseg)
    
    if center:
        a = pad_axis(a, 0, nperseg // 2, axis=axis)
        a = pad_axis(a, 0, nperseg // 2, axis=axis, append=True)
    
    segments = chunk(a, nperseg, noverlap, axis, pad=pad)
    if detrend:
        segments = segments - np.expand_dims(segments.mean(axis), axis)
    output = np.fft.rfft(segments * window, norm='ortho' if norm else None)
    
    return np.moveaxis(output / sum(window), axis if axis >= 0 else axis - 1, 0)
    
def dft(a, axis=-1, real=True, detrend=True, norm=False, window='hann'):
    """Perform a windowed discrete Fourier Transform along the specified axis.
    Equivelent to np.fft.rfft if window is None.

    Parameters
    ============
    a: array-like
        Array of time-series measurements
    axis: int, default=-1
        The axis along which the fourier transform is applied
    real: bool, default=True
        Whether to use the real-valued fourier transform.
    detrend: bool, default=True
        Whether to detrend the signal by subtracting its mean.
    norm: bool, default=False
        Whether to apply the 'ortho' norm.
    window: str or None, default='hann'
        The specified window, if any, to use during the transform
    
    Returns
    ==========
    output: np.ndarray
        The transformed array.
    """
    if window is None:
        window = np.ones(a.shape[axis], dtype=a.dtype)
    elif window == 'hann':
        window = np.hanning(a.shape[axis] + 1)[:-1]
    else:
        window = scipy.signal.get_window(window, a.shape[axis])
    if detrend:
        a = a - np.expand_dims(a.mean(axis), axis)
    a = a * window
    ft_func = np.fft.rfft if real else np.dual.fft
    output = ft_func(a, norm='ortho' if norm else None) if real else ft_func(a)
    return output / sum(window)

def dftinv(a, axis=-1, real=True, norm=False, window='hann'):
    output_shape = (a.shape[axis] - 1) * 2 if real else a.shape
    if window is None:
        window = np.ones(output_shape, dtype='d')
    elif window == 'hann':
        window = np.hanning(output_shape + 1)[:-1]
    else:
        window = scipy.signal.get_window(window, output_shape)
    a = a * sum(window)
    ftinv_func = np.fft.irfft if real else np.dual.ifft
    output = ftinv_func(a, axis=axis, norm='ortho' if norm else None) if real else ftinv_func(a, axis=axis)
    output = np.round(output / window, 4)
    return output