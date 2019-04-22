import numpy as np

def pad_axis(a, value, size=1, axis=-1, append=False):
    """
        Pad an array with a single value along a chosen axis.
        
        Parameters
        ===========
        a: np.ndarray
            The array to pad
        value: scalar
            The value with which to extend the array
        size: int
            The number of times `value` is added along the axis
        axis: int
            The axis along which items are added
        append: bool, default=False
            If True, add items to the end of axis. 
            If False, add them to the beginning.
        
        Returns
        ==========
        output: np.ndarray
            A copy of `a` with the added values
    """
    shape = list(a.shape)
    shape[axis] = size
    b = np.zeros(shape, a.dtype) + value
    ars = [a, b] if append else [b, a]
    return np.concatenate(ars, axis=axis)

def chunk(a, nperseg, noverlap=None, axis=-1, copy=False, pad=False):
    """
        Break an array into even-sized segments along a chosen axis.
        By default, returns an array whose elements are an immutable view of 'a'.

        Parameters
        ===========
        a: array-like
            Array to segment
        nperseg: int
            The number of items in each segment
        noverlap: int or None, default=None
            The number of shared items in adjacent segments. 
            If None, defaults to `nperseg // 2`
        axis: int, default=-1
            The axis to segmentize
        copy: bool, default=True
            If True, return a mutable copy of the array.
        pad: bool, default=False
            If True, zero-pad the array until the chosen axis can be split evenly into an integer number of segments.
            If False, no padding occurs and some values of `a` may not be present in the returned array.
        
        Returns
        ==========
        output: np.ndarray
            The segmented array. Iterating along `axis` will yield the new segments.

        Examples
        ==========
        a = np.arange(1, 6)
        chunk(a, 4, 2, pad=True)
        -> array([[1, 2, 3, 4]
                  [3, 4, 5, 0])
        
        chunk(a, 4, 2, pad=False)
        -> array([[1, 2, 3, 4]])

        a = np.array([[0], [1], [2], [3]])
        chunk(a, 2, 1, axis=0)
        -> array([[[0],
                   [1]],

                  [[1],
                   [2]],
                   
                  [[2],
                   [3]]])
    """
    if axis < 0:
        axis += np.ndim(a)
    a = np.asanyarray(a)
    if noverlap is None:
        noverlap = nperseg // 2
    if pad:
        npad = (nperseg - a.shape[axis]) % (nperseg - noverlap) % nperseg
        a = pad_axis(a, 0, npad, axis, append=True)
    b = restride(a, nperseg, axis)
    c = slice_axis(b, axis, step=nperseg-noverlap)
    if copy:
        return c.copy()
    else:
        return c
    
def restride(a, n, axis=-1):
    """
        Create a view of `a` that replaces the dimension along `axis` with new dimensions of sizes 
        `a.shape[axis] - n + 1` and `n` and strides `a.strides[axis]`.
        Useful for creating overlapping segments of an array. By default, the returned array is not writeable.
        Parameters
        ===========
        a: np.ndarray
            Array for which to create a view
        n: int
            Length of the second added dimension
        axis: int
            The axis to be replaced.
        
        Returns
        ==========
        output: np.ndarray
            Immutable view of `a`. 
    """
    while axis < 0:
        axis += np.ndim(a)
    shape = [*a.shape[:axis], a.shape[axis] - n + 1, n, *a.shape[axis+1:]]
    b = np.lib.stride_tricks.as_strided(a, 
                                        shape=shape, 
                                        strides=np.insert(a.strides, [axis], a.strides[axis]), 
                                        writeable=False)
    return b

def slice_axis(a, axis, start=None, stop=None, step=None):
    """
        Slice an array along a specified axis.
        Parameters
        ===========
        a: np.ndarray
            Array to slice
        axis: int
            The axis along which to slice
        start, stop, step: int or None, default=None
            Arguments for the applied slice.
        
        Returns
        ==========
        output: np.ndarray
            The sliced array
    """
    return a[tuple(slice(None) if i != axis else slice(start, stop, step) for i in range(np.ndim(a)))]