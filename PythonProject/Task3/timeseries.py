# Task3/timeseries_utils.py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def rolling_window_1d(a, window):
    """Return shape (n-window+1, window) view using stride tricks."""
    if window > a.shape[0]:
        raise ValueError("Window too large")
    shape = (a.shape[0] - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return as_strided(a, shape=shape, strides=strides)

def rolling_mean_numpy_1d(a, window):
    win = rolling_window_1d(a, window)
    return win.mean(axis=1)

def rolling_mean_pandas_1d(a, window):
    s = pd.Series(a)
    return s.rolling(window).mean().to_numpy()[window-1:]

def ewma_numpy(x, alpha):
    # efficient vectorized EWMA using recursion
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out

def fft_bandpass_1d(x, low_freq, high_freq, fs=1.0):
    # x: 1D signal; fs: sampling frequency
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    Xf = np.fft.rfft(x)
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    Xf_filtered = Xf * mask
    x_filt = np.fft.irfft(Xf_filtered, n=n)
    return x_filt

# Optional numba-accelerated rolling mean
if NUMBA_AVAILABLE:
    from numba import njit
    @njit
    def rolling_mean_numba(a, window):
        n = a.shape[0]
        outlen = n - window + 1
        out = np.empty(outlen, dtype=np.float64)
        for i in range(outlen):
            s = 0.0
            for j in range(window):
                s += a[i+j]
            out[i] = s / window
        return out
else:
    def rolling_mean_numba(a, window):
        raise RuntimeError("Numba not available")
