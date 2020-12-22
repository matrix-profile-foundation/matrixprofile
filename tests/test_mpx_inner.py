#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This tests the refactored C interface inner_self through its Python wrapper.
Tests were adapted from the general mpx tests. Missing components are computed inline.

"""
import os
import numpy as np
import matrixprofile
from matrixprofile.algorithms import cympx_inner
from matrixprofile import cycore
MODULE_PATH = matrixprofile.__path__[0]


def get_numpy_crosscov(ts, mu, cmpseq):
    """
    Parameters
    ----------

    ts: np.ndarray
        The input time series.

    mu: np.ndarray
        The per window mean of the time series.

    cmpseq: np.ndarray
        The comparison sequence.


    Returns
    -------

    cc:
        The cross covariance as computed by numpy. 
        
    
    See also numpy.correlate.

    """

    w, = cmpseq.shape
    subseqct = ts.shape[0] - w + 1
    if mu.shape[0] != subseqct:
        raise ValueError
    cc = np.empty(subseqct)
    for i in range(subseqct):
        cc[i] = np.dot(ts[i:i+w]-mu[i], cmpseq)
    return cc


def get_numpy_covdiff(ts, mu, w, sep):
    """

    Given:
        U = T[i : i + w] - mu[i],
        V = T[sep : sep + w] - mu[sep],
        W = T[i + 1 : i + w + 1] - mu[i + 1],
        X = T[sep + 1 : sep + w + 1] - mu[sep + 1]
        0 <= i,sep <= len(T) - w

    Return the values

       <W, X>  - <U, V> 

    for i=0....ts.shape[0] - w +1

    Normally we compute this quantity using a simpler formula.
    This acts as a control test, when the data is not extremely
    ill conditioned.


    Parameters
    ----------
    ts: numpy.ndarray
        The input time series.
    
    mu: numpy.ndarray
        The mean vector corresponding to the input time series.
    
    w: int
        The window length used for subsequence comparisons.

    sep: int
        The starting position of the first comparison sequence in ts.


    Returns
    -------
    
    diffs: numpy.ndarray
        Sequence of differences between co-moments.
        
    """

    assert(0 < w < ts.shape[0])
    subseqct = ts.shape[0] - w + 1
    assert(subseqct == mu.shape[0])
    cmtct = subseqct - sep

    cov = np.empty(cmtct, dtype='d')
    
    for i in range(cmtct):
        cov[i] = np.dot(ts[i:i+w] - mu[i], ts[i+sep:i+sep+w] - mu[i+sep])
    
    return np.diff(cov)
    

def test_crosscov():
    """
    Test calculation of windowed co-moments against a result produced by numpy.

    """
    ts = np.random.randn(2**14)
    windowlens = [3, 10, 100, 512]
    for w in windowlens:
        subseqct = ts.shape[0] - w + 1
        mu, invn = cycore.muinvn(ts, w)
        cc = np.empty(subseqct-w)
        cmpseq = ts[:w] - mu[0]
        np.testing.assert_almost_equal(np.sum(cmpseq), 0)
        cympx_inner.compute_cross_cov(cc, ts[w:], mu[w:], cmpseq)
        desired = get_numpy_crosscov(ts[w:], mu[w:], cmpseq)
        np.testing.assert_allclose(cc, desired, rtol=1e-8)


def run_ab_compare(cov, df_a, dg_a, df_b, dg_b, invn_a, invn_b, w):
    
    subseqct_a = invn_a.shape[0]
    subseqct_b = invn_b.shape[0]
    assert(df_a.shape[0] == dg_a.shape[0] == invn_a.shape[0]-1)
    assert(df_b.shape[0] == dg_b.shape[0] == invn_b.shape[0]-1)

    mp_a = np.full(subseqct_a, -1.0, dtype='d')
    mp_b = np.full(subseqct_b, -1.0, dtype='d')
    
    mpi_a = np.full(subseqct_a, -1, dtype=np.int_)
    mpi_b = np.full(subseqct_b, -1, dtype=np.int_)
    
    cympx_inner.compute_ab_compare(mp_a, mp_b, mpi_a, mpi_b, cov, df_a, df_b, dg_a, dg_b, invn_a, invn_b)

    return mp_a, mp_b, mpi_a, mpi_b
    

def cross_cov_self_last(ts, mu, subseqlen, minsep):
    subseqct = ts.shape[0] - subseqlen + 1
    cv = np.empty(diagct)
    lastpos = subseqct - 1
    cmpseq = ts[lastpos:] - mu[lastpos]
    diagct = lastpos - minsep
    for i in range(trunc):
        offs = diagct - i
        cv[i] = np.dot(ts[offs:offs+w] - mu[offs], cmpseq)

    return cv


def test_mpx_self_stability():
    ts = np.random.randn(2**14)
    w = 100
    minsep = w
    subseqct = ts.shape[0] - w + 1
    crosscmpct = subseqct - minsep
    mu, invn = cycore.muinvn(ts, w)
    mu = np.asarray(mu, dtype='d')
    invn = np.asarray(invn, dtype='d')
    
    df = np.empty(subseqct-1, dtype='d')
    dg = np.empty(subseqct-1, dtype='d')
    cympx_inner.compute_difference_equations(df, dg, ts, mu)
    
    cc = np.empty(crosscmpct, dtype='d')
    cympx_inner.compute_cross_cov(cc, ts[minsep:], mu[minsep:], ts[:w] - mu[0])

    # Check initial co-moments
    cc_chk = get_numpy_crosscov(ts[minsep:], mu[minsep:], ts[:w] - mu[0])
    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)
    mp = np.full(subseqct, -1.0, dtype='d')
    mpi = np.full(subseqct, -1, dtype=np.int_)
    cympx_inner.compute_self_compare(mp, mpi, cc, df, dg, invn, w, minsep)

    # Now the ith element of "cc" should contain the co-moment for the last window of the time series
    # and another sequence, at a separation of minsep + i from the last

    last = ts[subseqct-1:] - mu[-1]
    for i in range(crosscmpct):
        bound = crosscmpct - i - 1
        cc_chk[i] = np.dot(ts[bound:bound+w] - mu[bound], last)

    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)


def test_mpx_ab_stability():
    """
    This tests whether an A -> B comparison results in co-moment divergence when computing
    update steps, regardless of whether the perturbation is visible from nearest neighbor outputs. 

    """

    ts_a = np.random.randn(2**14)
    ts_b = np.random.randn(2**14)
    w = 100

    mu_a, invn_a = cycore.muinvn(ts_a, w)
    mu_a = np.asarray(mu_a, dtype='d')
    invna_a = np.asarray(invn_a, dtype='d')
    
    mu_b, invn_b = cycore.muinvn(ts_b, w)
    mu_b = np.asarray(mu_b, dtype='d')
    invn_b = np.asarray(invn_b, dtype='d')

    subseqct_a = mu_a.shape[0]
    subseqct_b = mu_b.shape[0]

    df_a = np.empty(subseqct_a-1)
    dg_a = np.empty(subseqct_a-1)
    df_b = np.empty(subseqct_b-1)
    dg_b = np.empty(subseqct_b-1)

    # Note: These are allocated here, because allocating them on the cython side 
    # has caused reference counting issues in the past under Ubuntu, Python 3.8 Anaconda distribution. 
    # I haven't yet located the bug on the Cython side.
    cympx_inner.compute_difference_equations(df_a, dg_a, ts_a, mu_a)
    cympx_inner.compute_difference_equations(df_b, dg_b, ts_b, mu_b)

    # Check initial cross cov against numpy
    cc = np.empty(subseqct_a)
    cmpseq = ts_b[:w] - mu_b[0]
    cympx_inner.compute_cross_cov(cc, ts_a, mu_a, cmpseq)
    cc_chk = get_numpy_crosscov(ts_a, mu_a, cmpseq)
    
    # Check that initial cross cov roughly agrees with numpy
    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)
    
    # Now test the stability of calculations on the upper triangular portion of the implicitly
    # defined similarity matrix

    mp_a_u, mp_b_u, mpi_a_u, mpi_b_u = run_ab_compare(cc, df_a, dg_a, df_b, dg_b, invn_a, invn_b, w)
    
    # After running the comparison, cov[i] holds the co-moments between the final comparisons
    cc_chk = np.empty(subseqct_a, dtype='d')

    # At least one of the comparison subsequences is always the last subsequence 
    # in one of the two time series. 
    last_a = ts_a[subseqct_a-1:] - mu_a[-1]
    last_b = ts_b[subseqct_b-1:] - mu_b[-1]
    for i in range(subseqct_a):
        max_steps_a = subseqct_a - i - 1
        if max_steps_a < subseqct_b:
            cc_chk[i] = np.dot(last_a, ts_b[max_steps_a:max_steps_a+w] - mu_b[max_steps_a])
        elif subseqct_b <  max_steps_a:
            cc_chk[i] = np.dot(ts_a[subseqct_b-1:subseqct_b+w-1] - mu_a[subseqct_b-1], last_b)
        else:
            cc_chk[i] = np.dot(last_a, last_b)
        
    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)

    # Now perform the same test for calculations performed on the lower triangular portion 
    # of the implicitly defined similarity matrix
    cc = np.empty(subseqct_b)
    cmpseq = ts_a[:w] - mu_a[0]
    cympx_inner.compute_cross_cov(cc, ts_b, mu_b, cmpseq)
    cc_chk = get_numpy_crosscov(ts_b, mu_b, cmpseq)
    
    # Check that initial cross cov roughly agrees with numpy
    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)
    
    # lower triangular portion runs with parameters corresponding to ts_a, ts_b swapped
    mp_b_l, mp_a_l, mpi_b_l, mpi_a_l = run_ab_compare(cc, df_b, dg_b, df_a, dg_a, invn_b, invn_a, w)
    cc_chk = np.empty(subseqct_b)

    last_a = ts_a[subseqct_a-1:] - mu_a[-1]
    last_b = ts_b[subseqct_b-1:] - mu_b[-1]
    for i in range(subseqct_a):
        max_steps_b = subseqct_b - i - 1
        if max_steps_b < subseqct_a:
            cc_chk[i] = np.dot(last_b, ts_a[max_steps_b:max_steps_b+w] - mu_b[max_steps_b])
        elif subseqct_a <  max_steps_b:
            cc_chk[i] = np.dot(ts_b[subseqct_b-1:subseqct_b+w-1] - mu_b[subseqct_b-1], last_a)
        else:
            cc_chk[i] = np.dot(last_a, last_b)
    
    np.testing.assert_allclose(cc, cc_chk, rtol=1e-8)


def test_difference_equations():
    """

    This test compares the numerical output of the formula we use here with one obtained
    by directly taking the difference of two explicitly computed co-moments. 
    
    Some cancellation is to be expected, but the two results should still be close.
        
    More specifically, this verifies that for time series T, 

    if U = T[i : i + w] - mu[i],
        V = T[j : j + w] - mu[j],
        W = T[i + 1 : i + w + 1] - mu[i + 1],
        X = T[j + 1 : j + w + 1] - mu[j + 1]
        0 <= i,j <= len(T) - w

    then

       <W, X>  - <U, V>  == df[i + 1] * dg[j + 1] + df[j + 1] * dg[i + 1]

    with df, dg computed by the corresponding function in cympx_inner. 

    """
    ts = np.loadtxt('tests/sampledata.txt')
    w = 200
    sep = w
    subseqct = ts.shape[0] - w + 1
    cmtct = subseqct - sep
    mu, _ = cycore.muinvn(ts, w)
    df = np.empty(subseqct-1, dtype='d')
    dg = np.empty(subseqct-1, dtype='d')
    cympx_inner.compute_difference_equations(df, dg, ts, mu)

    desired = get_numpy_covdiff(ts, mu, w, sep)
    actual = np.empty(cmtct-1, dtype='d')

    for i in range(cmtct - 1):
        actual[i] = df[i] * dg[i+sep] + df[i+sep] * dg[i]
    
    np.testing.assert_allclose(actual, desired, rtol=1e-8)

    for i in range(cmtct-1):
        k = i + sep
        actual[i] = df[i] * dg[k] + df[k] * dg[i]
    
    np.testing.assert_allclose(actual, desired, rtol=1e-8)    
    cmtct = subseqct - sep
    cov = np.empty(cmtct, dtype='d')
    
    for i in range(cmtct):
        cov[i] = np.dot(ts[i:i+w] - mu[i], ts[i+sep:i+sep+w] - mu[i+sep])

    desired = get_numpy_covdiff(ts, mu, w, sep)
    actual = np.empty(cmtct-1, dtype='d')

    for i in range(cmtct - 1):
        actual[i] = df[i] * dg[i+sep] + df[i+sep] * dg[i]
        
    np.testing.assert_allclose(actual, desired, rtol=1e-8)

    sep = 5
    cmtct = subseqct - sep

    desired = get_numpy_covdiff(ts, mu, w, sep)
    actual = np.empty(cmtct - 1, dtype='d')
    
    for i in range(cmtct-1):
        actual[i] = df[i] * dg[i+sep] + df[i+sep] * dg[i]

    np.testing.assert_allclose(actual, desired, rtol=1e-8)
