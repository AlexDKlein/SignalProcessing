import numpy as np
import scipy

class BandPassFilter():
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high
    
    def apply(self, X):
        ft = np.fft.rfft(X)
        freqs = np.fft.rfftfreq(X.shape[-1])
        if self.low is not None and self.high is not None:
            ft = np.where((self.low < freqs) & (freqs < self.high), ft, 0)
        elif self.low is not None:
            ft = np.where(self.low < freqs, ft, 0)
        elif self.high is not None:
            ft = np.where(freqs < self.high, ft, 0)
        return np.fft.irfft(ft)


class MultiBandFilter():
    def __init__(self, *bands):
        self.bands = bands
        
    def apply(self, X):
        ft = np.fft.rfft(X)
        freqs = np.fft.rfftfreq(X.shape[-1])
        mask = np.zeros_like(freqs, dtype=bool)
        for low,high in self.bands:
            mask = np.where((freqs > low) & (freqs < high), True, mask)
        ft = np.where(mask, ft, 0)
        return np.fft.irfft(ft)
    