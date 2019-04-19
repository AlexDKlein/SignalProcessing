import numpy as np
import scipy

class SignalFilter():
    def transform(self, X, t=1.0):
        if np.iterable(t):
            t = np.ptp(t, axis=-1).mean()
        d = t / (X.shape[-1])
        if np.isreal(X).all():
            X_transformed = np.fft.rfft(X)
            freqs = np.fft.rfftfreq(X.shape[-1], d)
        else:
            X_transformed = np.fft.fft(X)
            freqs = np.fft.fftfreq(X.shape[-1], d) 
        return X_transformed, freqs


class BandPassFilter(SignalFilter):
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high
    
    def apply(self, X, t=1.0):
        ft, freqs = self.transform(X, t)
        if self.low is not None and self.high is not None:
            ft = np.where((self.low < freqs) & (freqs < self.high), ft, 0)
        elif self.low is not None:
            ft = np.where(self.low < freqs, ft, 0)
        elif self.high is not None:
            ft = np.where(freqs < self.high, ft, 0)
        if np.isreal(X).all():
            return np.fft.irfft(ft, n=X.shape[-1])
        else:
            return np.fft.ifft(ft, n=X.shape[-1])


class MultiBandFilter(SignalFilter):
    def __init__(self, *bands):
        self.bands = bands
        
    def apply(self, X, t=1.0):
        ft, freqs = self.transform(X, t)
        mask = np.zeros_like(freqs, dtype=bool)
        for low,high in self.bands:
            mask = np.where((freqs > low) & (freqs < high), True, mask)
        ft = np.where(mask, ft, 0)
        if np.isreal(X).all():
            return np.fft.irfft(ft, n=X.shape[-1])
        else:
            return np.fft.ifft(ft, n=X.shape[-1])
   
    
class LowPassFilter(SignalFilter):
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def apply(self, X, t=1.0):
        ft, freqs = self.transform(X, t)
        ft = np.where(freqs < self.cutoff, ft, 0)
        if np.isreal(X).all():
            return np.fft.irfft(ft, n=X.shape[-1])
        else:
            return np.fft.ifft(ft, n=X.shape[-1])


class HighPassFilter(SignalFilter):
    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def apply(self, X, t=1.0):
        ft, freqs = self.transform(X, t)
        ft = np.where(freqs > self.cutoff, ft, 0)
        if np.isreal(X).all():
            return np.fft.irfft(ft, n=X.shape[-1])
        else:
            return np.fft.ifft(ft, n=X.shape[-1])