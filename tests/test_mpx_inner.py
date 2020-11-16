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


def dfdg_with_verif(ts, mu, test_offsets):
    """

    Parameters
    ----------
    ts: array_like
        The input time series.
    
    mu: array_like
        The mean vector corresponding to the input time series.
    
    w: int
        The window length used for subsequence comparisons.

    test_offsets: array_like[int]
        Positional offsets. The implementation is testing by explicitly
        computing cross covariance differences at these offsets, then comparing
        to the outcome of the same thing, computed using difference equations.

    Returns
    -------
    df, dg : array_like, array_like
    
    Test diff_equations through the python interface compute_dfdg
    
    This verifies that for four windows of a time series, 

    if U = T[i : i + w] - mu[i],
        V = T[j : j + w] - mu[j],
        W = T[i + 1 : i + w + 1] - mu[i + 1],
        X = T[j + 1 : j + w + 1] - mu[j + 1]
        0 <= i,j <= len(T) - w

    then

       <W, X>  - <U, V>  == df[i + 1] * dg[j + 1] + df[j + 1] * dg[i + 1]

    with df, dg computed by the corresponding function.

    """

    sseqct = mu.shape[0]
    w = ts.shape[0] - sseqct + 1
    df = np.empty(sseqct, dtype='d')
    dg = np.empty(sseqct, dtype='d')

    cympx_inner.compute_diff_eqns(df, dg, ts, mu)
    
    for offset in test_offsets:
        cmtct = sseqct - offset
        cov = np.empty(cmtct, dtype='d')
    
        for i in range(cmtct):
            cov[i] = np.dot(ts[i:i+w] - mu[i], ts[i+offset:i+offset+w] - mu[i+offset])
    
        desired = np.diff(cov)
        actual = np.empty(cmtct-1, dtype='d')

        for i in range(cov.shape[0] - 1):
            # the offset by 1 covers that we're indexing differences only here
            actual[i] = df[i+1] * dg[i+offset+1] + df[i+offset+1] * dg[i+1]
  
            # We can use a tighter tolerance if the parts that aren't being explicitly tested here
            # are computed using an arbitrary precision library.
        
        np.testing.assert_allclose(actual, desired, rtol=1e-8)

    return df, dg


def cross_cov_with_verif(ts, mu, cmpseq):
    """
    Tests unscaled cross covariance between a time series and the first window of the time series.
    Also returns the computed result to be used in downstream tests.

    Parameters
    ----------

    ts : array_like
        The input time series.

    cmpseq: array_like
        Mean centered comparison sequence, defaults to the first in the time series.

    Outputs
    -------
    cov: array_like
        Unscaled co-moment comparing ts[:w] - mu[0] to each mean centered window ts[:w] - mu[0] ..... ts[-w:] - mu[-w].

    """
    w = cmpseq.shape[0]
    sseqct = ts.shape[0] - w + 1
    assert(mu.shape[0] == sseqct)
    np.testing.assert_almost_equal(np.sum(cmpseq), 0)
    actual = np.empty(sseqct)
    cympx_inner.compute_cross_cov(actual, ts, mu, cmpseq)
    desired = np.empty(sseqct)
    for i in range(sseqct):
        desired[i] = np.dot(ts[i : i + w] - mu[i], cmpseq)  
    # If I set tolerance too tight here, it fails.
    # Testing against numpy results is a little tenuous. 
    # We should use a high precision library to aid in verification. 
    np.testing.assert_allclose(actual, desired, rtol=1e-10)
    return actual


def run_self_compare(ts, w):
    sseqct = ts.shape[0] - w + 1
    minlag = w // 4
    mu, sig = cycore.muinvn(ts, w)
    mu = np.asarray(mu, dtype='d')
    mp = np.full(sseqct, -1.0, dtype='d')
    mpi = np.full(sseqct, -1, dtype=np.int_)
    sig = np.asarray(sig, dtype='d')
    df,dg = dfdg_with_verif(ts, mu, test_offsets=[w])
    cmpseq = ts[:w] - mu[0]
    np.testing.assert_almost_equal(np.sum(cmpseq), 0)
    cov = cross_cov_with_verif(ts[minlag:], mu[minlag:], cmpseq)
    cympx_inner.compute_self_cmp(mp, mpi, cov, df, dg, sig, w, minlag)
    return mp, mpi


def run_ab_compare(ts_a, ts_b, w):
    mu_a, sig_a = cycore.muinvn(ts_a, w)
    mu_a = np.asarray(mu_a, dtype='d')
    siga_a = np.asarray(sig_a, dtype='d')
    
    mu_b, sig_b = cycore.muinvn(ts_b, w)
    mu_b = np.asarray(mu_b, dtype='d')
    sig_b = np.asarray(sig_b, dtype='d')

    subseqct_a = ts_a.shape[0] - w + 1
    subseqct_b = ts_b.shape[0] - w + 1

    df_a, dg_a = dfdg_with_verif(ts_a, mu_a, test_offsets=[w])
    df_b, dg_b = dfdg_with_verif(ts_b, mu_b, test_offsets=[w])
    
    cmpseq_u = ts_b[:w] - mu_b[0]
    cmpseq_l = ts_a[:w] - mu_a[0]
    
    cov_u = cross_cov_with_verif(ts_a, mu_a, cmpseq_u)
    cov_l = cross_cov_with_verif(ts_b, mu_b, cmpseq_l)

    mp_a_u = np.full(subseqct_a, -1.0, dtype='d')
    mp_a_l = np.full(subseqct_a, -1.0, dtype='d')
    mp_b_u = np.full(subseqct_b, -1.0, dtype='d')
    mp_b_l = np.full(subseqct_b, -1.0, dtype='d')
    
    mpi_b_l = np.full(subseqct_b, -1, dtype=np.int_)
    mpi_b_u = np.full(subseqct_b, -1, dtype=np.int_)
    mpi_a_u = np.full(subseqct_a, -1, dtype=np.int_)
    mpi_a_l = np.full(subseqct_a, -1, dtype=np.int_)
    
    cympx_inner.compute_ab_cmp(mp_a_u, mp_b_u, mpi_a_u, mpi_b_u, cov_u, df_a, df_b, dg_a, dg_b, sig_a, sig_b)
    cympx_inner.compute_ab_cmp(mp_b_l, mp_a_l, mpi_b_l, mpi_a_l, cov_l, df_b, df_a, dg_b, dg_a, sig_b, sig_a)
    
    mp_a = np.empty(mu_a.shape[0], dtype='d')
    mp_b = np.empty(mu_b.shape[0], dtype='d')
    mpi_a = np.empty(mu_a.shape[0], dtype=np.int_)
    mpi_b = np.empty(mu_b.shape[0], dtype=np.int_)

    for i, (u, v, j, k) in enumerate(zip(mp_a_u, mp_a_l, mpi_a_u, mpi_a_l)):
        if u > v:
            mp_a[i] = u
            mpi_a[i] = j
        else:
            mp_a[i] = v
            mpi_a[i] = k

    for i, (u, v, j, k) in enumerate(zip(mp_b_u, mp_b_l, mpi_b_u, mpi_b_l)):
        if u > v:
            mp_b[i] = u
            mpi_b[i] = j
        else:
            mp_b[i] = v
            mpi_b[i] = k
    
    # stability issue around poles, 
    # some of which are contained in the test cases
    # I hope to be able to weaken this restriction
    np.clip(mp_a, -1.0, 1.0, out=mp_a)
    np.clip(mp_b, -1.0, 1.0, out=mp_b)

    return mp_a, mp_b, mpi_a, mpi_b


def test_mpx_inner_small_series_self_join_pearson():
    ts = np.asarray([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1], dtype='d')
    w = 4
    desired = np.array([0.522232967867094, 0.577350269189626, 0.904534033733291, 1, 1, 0.522232967867094, 0.904534033733291, 1, 1])
    desired_pi = np.array([4, 2, 6, 7, 8, 1, 2, 3, 4])

    actual, actual_pi = run_self_compare(ts, w)
    np.clip(actual, -1.0, 1.0, out=actual)

    np.testing.assert_almost_equal(actual, desired, decimal=4)
    np.testing.assert_almost_equal(actual_pi, desired_pi, decimal=4)


def test_mpx_ab_inner_matlab():
    ts_a = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    ts_b = ts_a[199:300]
    w = 32
    ml_mp_a = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpa.txt'))
    ml_mp_b = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpb.txt'))
    
    mp_a, mp_b, mpi_a, mpi_b = run_ab_compare(ts_a, ts_b, w)
    
    # Since this comparison expects normalized euclidean distance
    mp_a = np.sqrt(2 * w * (1 - mp_a))
    mp_b = np.sqrt(2 * w * (1 - mp_b))
    np.testing.assert_almost_equal(ml_mp_a, mp_a, decimal=4)
    np.testing.assert_almost_equal(ml_mp_b, mp_b, decimal=4)
    return mp_a, mp_b
