import numpy as np
from warnings import warn
from matrixprofile.cycore import muinvn


def valid_windows(ts, w):
    """
    This determines the windows of a time series which should have a normalized
    representation.

    Parameters
    ----------
    ts : numpy.ndarray
        The input time series
    w : int
        The window size.

    Returns
    -------
    
    first: int
        The index of the first valid window or None.
    last: int
        The index of the last valid window or None.
    isvalidwindow: np.ndarray[bool]
        A boolean array set to True for each valid window in the interval first, last+1.

    """

    if not (1 < w <= ts.shape[0]):
        raise ValueError("Windowed normalization of time series is undefined for window lengths less than 2")
    
    windowcount = max(0, ts.shape[0] - w + 1)
    
    if windowcount == 0:
        return None, None, np.empty(0, dtype='bool')

    windowcount = max(0, ts.shape[0] - w + 1)
    # find interior windows containing non-finite elements
    isvalidwindow = np.ones(windowcount, dtype='bool')
    nonfinite, = np.where(np.invert(np.isfinite(ts)))
    
    prev = 0
    for pos in nonfinite:
        exclbegin = max(prev, pos - w + 1)
        isvalidwindow[exclbegin:pos+1] = False

    # Now check whether any remaining windows are sequences of constants, 
    # which definitiion lack a normalized form
    
    finitewindows, = np.where(isvalidwindow)

    for pos in finitewindows:
        if np.all(ts[pos] == ts[pos+1:pos+w]):
             isvalidwindow[pos] = False

    # Truncate the mask to begin and end on a valid window if possible
    validwindows, = np.where(isvalidwindow)
    validcount = validwindows.shape[0]

    if validcount == 0:
        first = last = None
    elif validcount == 1:
        first = last = validwindows[0]
    else:
        first = validwindows[0]
        last = validwindows[-1]

    # Truncate so that this starts and ends
    # on a valid window if any exist.

    if (first, last) != (0, windowcount - 1):
        if (first, last) == (None, None):
            isvalidwindow = np.empty(0, dtype='bool')
        else:
            isvalidwindow = isvalidwindow[first:last+1]
    
    return first, last, isvalidwindow


def norm_parameters(ts, w):
    """
    Computes a reduced time series in addition to mean and inverse centered norm
    data, based on the normalizable windows in that time series.

    Parameters
    ----------
    ts : numpy.ndarray
        The input time series
    
    w : int
        The window size.

    Returns
    -------

    ts : np.ndarray
        The time series, based on remaining bounds after removal of leading and trailing
        non-normalizable windows.

    mu : np.ndarray
        The per window mean for each normalizable window within the remaining time series bounds.

    invn : np.ndarray
        The per window inverse centered norm for each normalizable window within the 
        remaining time series bounds.
    
    first : int
        The index of the first valid window or None.

    last : int
        The index of the last valid window or None.

    isvalidwindow : np.ndarray[bool]
        A boolean array set to True for each valid window in the interval first, last+1.

    """

    windowcount = ts.shape[0] - w + 1
    
    first, last, isvalidwindow = valid_windows(ts, w)

    if not isvalidwindow.shape[0]:
        return ts[:0], np.empty(0, dtype='bool'), first, last, isvalidwindow

    if (first, last) != (0, windowcount - 1):
        ts = ts[first:last+w]
        windowcount = ts.shape[0] - w + 1

    # For the purpose of computing normalization parameters
    # we need to remove only non-finite elements, not sequences
    # of constants.
    nonfinite, = np.where(np.invert(np.isfinite(ts)))

    if np.count_nonzero(ts):
        ts = ts.copy()
        ts[nonfinite] = 0

    mu, invn = muinvn(ts, w)
    validcheck = invn != 0 & np.isfinite(invn)

    # Check for cases where very poor conditioning invalidates additional windows.
    if np.any(isvalidwindow & np.invert(validcheck)):    
        warn("Loss of precision warning. One or more windows containing valid data failed to normalize correctly")
        isvalidwindow &= validcheck
        # recompute boundary conditions to exclude edge windows which 
        # could not be correctly normalized
        validwindows, = np.where(isvalidwindow)
        validcount = validwindows.shape[0]
        if validcount == 0:
            isvalidwindow = np.empty(0, dtype='bool')
        elif validcount == 1:
            isvalidwindow = np.ones(1, dtype='bool')
            pos = validwindows[0]
            ts = ts[pos:pos+w]
            mu = mu[pos:pos+1]
            invn = invn[pos:pos+1]
            last = first = first + pos
        else:
            first += validwindows[0]
            last -= isvalidwindow.shape[0] - 1 - validwindows[-1]
            ts = ts[first:last+w]
            mu = mu[first:last+w]

    return ts, mu, invn, first, last, isvalidwindow
