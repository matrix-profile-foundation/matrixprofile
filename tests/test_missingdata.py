import numpy as np
from matrixprofile.missingdata import valid_windows, norm_parameters


def test_window_length_bounds():
    ts = np.random.randn(2**14)
    #shortest allowed
    w = 2
    for w in (w, ts.shape[0]):
        windowcount = ts.shape[0] - w + 1
        first, last, isvalid = valid_windows(ts, w)
        np.testing.assert_array_equal((first, last), (0, windowcount - 1))
        ts, mu, invn, first, last, isvalid = norm_parameters(ts, w)
        np.testing.assert_array_equal((first, last), (0, windowcount - 1))
        np.testing.assert_array_equal(isvalid, True)
    

def test_missing_leading():
    ts = np.random.randn(2**14)
    w = 200
    missingat = 100
    ts[missingat] = np.nan
    windowcount = ts.shape[0]- w + 1
    first, last, isvalid = valid_windows(ts, w)
    np.testing.assert_array_equal((first, last),(missingat+1, windowcount - 1))
    ts_, mu, invn, first, last, isvalid = norm_parameters(ts, w)
    np.testing.assert_array_equal((first, last), (missingat+1, windowcount - 1))
    np.testing.assert_array_equal(ts[missingat+1:], ts_)
    np.testing.assert_array_equal(isvalid, True)
    np.testing.assert_array_equal(np.isfinite(mu), True)
    np.testing.assert_array_equal(np.isfinite(invn), True)


def test_missing_trailing():
    ts = np.random.randn(2**14)
    w = 200
    windowcount = ts.shape[0]- w + 1
    missingat = windowcount - 50
    ts[missingat:] = np.nan
    windowcount = ts.shape[0]- w + 1
    first, last, isvalid = valid_windows(ts, w)
    np.testing.assert_array_equal((first, last), (0, missingat-w))
    ts_, mu, invn, first, last, isvalid = norm_parameters(ts, w)
    np.testing.assert_array_equal((first, last), (0, missingat-w))
    np.testing.assert_array_equal(isvalid, True)
    np.testing.assert_array_equal(np.isfinite(mu), True)
    np.testing.assert_array_equal(np.isfinite(invn), True)


def test_interior_nooverlap():
    ts = np.random.randn(2**14)
    w = 200
    # Non overlapping interior windows should each 
    # remove w windows from consideration
    missingat = np.asarray([400, 800, 3051, 9000]) 
    ts[missingat] = np.nan
    windowcount = ts.shape[0]- w + 1
    first, last, isvalid = valid_windows(ts, w)
    np.testing.assert_array_equal(isvalid[missingat+1], True)
    np.testing.assert_array_equal(isvalid[missingat-1], False)
    ts_sparse, mu, invn, first, last, isvalid = norm_parameters(ts, w)
    np.testing.assert_array_equal((first, last), (0, windowcount - 1))
    np.testing.assert_array_equal(np.count_nonzero(ts != ts_sparse), 4)
    np.testing.assert_array_equal(np.isfinite(mu), True)
    np.testing.assert_array_equal(np.count_nonzero(isvalid), windowcount - 4 * w)
  