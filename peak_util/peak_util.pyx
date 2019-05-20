import cython
cimport numpy as np
import numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
def local_maxima_1d(np.ndarray[dtype=double, ndim=1] x):
    """local_maxima_1d(x)

        Find all local maxima in a one-dimensional signal.
    
        Parameters
        ===========
        x: np.ndarray[dtype=double, ndim=1]
            The input array.

        Returns
        ===========
        midpoints: np.ndarray[dtype=long, ndim=1]
            The midpoints of the peaks.
        left, right: np.ndarray[dtype=long, ndim=1]
            The left and right edges of the peaks.
    """
    cdef:
        np.ndarray[long, ndim=1] left, right, midpoints
        int n, m, i, i_max
        double peak
    
    n, m = 0, 0
    peak = x[0]
    i_max = x.shape[0]
    
    left = np.empty((i_max) // 2, dtype='l')
    right = np.empty((i_max) // 2, dtype='l')
    midpoints = np.empty((i_max) // 2, dtype='l')
    with nogil:
        for i in range(1, i_max):
            if m > n:
                if peak > x[i]:
                    right[n] = i - 1
                    midpoints[n] = ((i-1 + left[n])//2)
                    n += 1
                    peak = x[i]
                elif x[i] > peak:
                    left[n] = i
                    peak = x[i]
            elif x[i] > peak:
                left[m] = i
                m += 1
                peak = x[i]
            elif x[i] < peak:
                peak = x[i]

    left.resize(n, refcheck=False)
    right.resize(n, refcheck=False)
    midpoints.resize(n, refcheck=False)
    return midpoints, left, right

@cython.wraparound(False)
@cython.boundscheck(False)
cdef prominence(double[:] x, long[:] peaks):
    """
        Returns the prominences as well as the left and right bases
        for the provided peaks.
    """
    cdef:
        double peak, m, left_min, right_min
        long left, right, i, p, j
        long n_peaks = peaks.shape[0]
        np.ndarray[double, ndim=1] output = np.empty(n_peaks)
        np.ndarray[long, ndim=2] bases = np.empty((n_peaks, 2), dtype='l')
        
    with nogil:
        for i in range(n_peaks):
            p = peaks[i]
            peak = left_min = right_min = x[p]
            
            for j in range(p+1, len(x)):
                if x[j] > peak:
                    break
                elif x[j] < right_min:
                    right_min = x[j]
                    bases[i, 1] = j

            for j in range(p-1, -1, -1):
                if x[j] > peak:
                    break
                elif x[j] < left_min:
                    left_min = x[j]
                    bases[i, 0] = j

            m = left_min if left_min > right_min else right_min
            output[i] = x[p] - m
            
    return output, bases[:, 0], bases[:, 1]

def peak_prominence(x, i=None):
    """peak_prominence(x, i=None)

        Return the prominence of peaks in an input signal x.

        Parameters
        ===========
        x: np.ndarray[ndim=1]
            The input signal.
        i: int, array-like of ints, or None, default=None
            The indices of peaks for which to find the prominence.
            If None, all peaks are used.
        
        Returns 
        ==========
        output: np.ndarray[ndim=1]
            The prominences for each peak specified by `i`.
    """
    x = np.asanyarray(x, dtype='d')
    i = np.asanyarray(i) if i is not None else local_maxima_1d(x)[0]
    if np.ndim(i) < 1:
        i = np.expand_dims(i, 0)
    return prominence(x, i)[0]

@cython.wraparound(False)
@cython.boundscheck(False)
def argpeaks(np.ndarray[double, ndim=1, cast=True] x, long min_distance=0, long n=-1):
    """argpeaks(x, min_distance=0, n=-1)

    Find the position of peaks in an array x. 
    If min_distance > 0, peaks will be no closer than min_distance apart with the most highest peaks given priority.
    
    Parameters
    ==========
    x: np.ndarray[ndim=1]
        The input signal.
    min_distance: int
        The minimum distance between peaks. Higher peaks given priority.
    n: int, default=-1
        The number of peaks to keep. Default of -1 equals all peaks. Prominent peaks given priority.
    """
    cdef:
        np.ndarray[long, ndim=1] peaks = local_maxima_1d(x)[0]
        long n_peaks = len(peaks)
        long i,j,k
        np.ndarray[long, ndim=1] rank = np.argsort(x[peaks])
        np.uint8_t[::1] drop = np.zeros(n_peaks, dtype=np.uint8) 
    
    if n > 0:
            peaks = argpeaks(x, min_distance)
            return np.sort(peaks[peak_prominence(x, peaks).argsort()][-n:])
        
    with nogil:

        for i in range(n_peaks - 1, -1, -1):
            j = rank[i]
            if drop[j] == 1:
                continue

            k = j - 1
            while 0 <= k and peaks[j] - peaks[k] < min_distance:
                drop[k] = 1
                k -= 1

            k = j + 1
            while k < n_peaks and peaks[k] - peaks[j] < min_distance:
                drop[k] = 1
                k += 1
            
    return peaks[~drop.base.view(dtype=np.bool)]

@cython.wraparound(False)
@cython.boundscheck(False)
def peak_width(double[:] x, long[:] peaks, double rel_height=0.5):
    """peak_width(x, peaks, rel_height=0.5)
    
        Return the widths of peaks in a signal. By default, this is the
        full-width at half-prominence.

        Parameters 
        ===========
        x: np.ndarray[dtype=double, ndim=1]
            The input signal.
        peaks: np.ndarray[dtype=long, ndim=1]
            The indices of the selected peaks.
        rel_height: double, default=0.5
            The relative prominence where the width is found.
            By default, this is the half-maximum i.e. half-prominence.
        
        Returns
        ==========
        output: np.ndarray[dtype=double, ndim=1]
            The widths of the selected peaks.
    """
    cdef:
        double half_max, left, right
        long i, j, p
        long[:] left_bases, right_bases
        np.ndarray[double, ndim=1] prominences
        double[:] widths = np.empty_like(peaks, dtype='d')
    
    prominences, left_bases, right_bases = prominence(x, peaks)
    with nogil:
        for i in range(peaks.shape[0]):
            p = peaks[i]
            half_max = x[p] - prominences[i] * rel_height
            
            for j in range(p+1, right_bases[i] + 1):
                if x[j] <= half_max:
                    right = j - (half_max - x[j]) / (x[j - 1] - x[j])
                    break
                    
            for j in range(p-1, left_bases[i] - 1, -1):
                if x[j] <= half_max:
                    left = j + (half_max - x[j]) / (x[j + 1] - x[j])
                    break
                    
            widths[i] = right - left
    return widths.base
        