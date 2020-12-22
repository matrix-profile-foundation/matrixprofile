# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False
import numpy as np
cimport numpy as cnp


def compute_difference_equations(df, dg, ts, mu):
    """  
    Function compute_difference_equations is a python interface meant for testing difference equations used in mpx.

    Parameters
    ----------

    df : array_like
        output array for first mpx difference equation, original formula

    dg : array_like
        output array for second mpx difference equation, original formula

    ts : array_like
        input time series

    mu : array_like
        input per window means

    Returns
    -------

    """

    subseqct = mu.shape[0]
    w = ts.shape[0] - subseqct + 1
    if w < 2:
        raise ValueError(f'Inferred subsequence length {w} is outside of the supported range.')
    difference_equations(df, dg, ts, mu, w)


def compute_cross_cov(cc, ts, mu, cmpseq):
    """  
    Function compute_cross_cov provides a python interface to cross_cov. This is primarily intended for internal testing 
    without the need to compile test cases.

    Parameters
    ----------

    cc : array_like or None
        co-moment output buffer

    ts : array_like
        input time series

    mu : array_like
        input per window means

    cmpseq : array_like
        comparison sequence, assumed to be mean centered

    Returns
    -------

    """

    assert(cc.shape[0] == mu.shape[0] == ts.shape[0] - cmpseq.shape[0] + 1)
    cross_cov(cc, ts, mu, cmpseq)


def compute_self_compare(mp, mpi, cov, df, dg, sig, w, minlag, index_offset=0):
    """
    Function compute_self_compare is a python interface meant for internal testing of the C interface inner_self.

    Parameters
    ----------

    mp: array_like
        initialized input buffer for matrix profile output

    mpi: array_like
        initialized input buffer for matrix profile index output

    cov : array_like
        initial sequence of co-moments

    df : array_like
        first difference equation

    dg : array_like
        second difference equation

    w : int
        window or "subsequence" length

    index_offset : int
        The minimum index of any time series window used in computing cov.
        That is to say, given time series T and index_offset i,
        cov[0] = dot(T[i:i+w] - mu[i], T[i+minlag:i+minlag+w] - mu[i+minlag])

    cmpseq : array_like
        comparison sequence, assumed to be mean centered

    Returns
    -------

    """

    subseqct = sig.shape[0]
    assert(df.shape[0] == dg.shape[0] == sig.shape[0] - 1 == mp.shape[0] - 1 == mpi.shape[0] - 1)
    self_compare(mp, mpi, cov, df, dg, sig, w, minlag, index_offset)


def compute_ab_compare(mp_a, mp_b, mpi_a, mpi_b, cov, df_a, df_b, dg_a, dg_b, sig_a, sig_b, offset_a=0, offset_b=0):
    """
    Function compute_ab_compare is a python interface meant for the C interface inner_ab.
    This is a low level interface, used for internal testing.

    Parameters
    ----------

    mp_a : array_like
        initialized matrix profile buffer, which receives the nearest neighbor similarity of each subsequence of the 
        first time series with respect to the second. This is updated in place, starting from the existing values.

    mp_b : array_like
        initialized matrix profile buffer, which receives the nearest neighbor similarity of each subsequence of the 
        second time series with respect to the first. This is updated in place, starting from the existing values.

    mpi_a : array_like
        initialized matrix profile index buffer, which receives the index of the nearest neighbor of each subsequence of the 
        first time series with respect to the second, whenever mp_a is updated. This is updated in place, starting
        from the existing values.

    mpi_b : array_like
        initialized matrix profile index buffer, which receives the index of the nearest neighbor of each subsequence of the 
        second time series with respect to the first, whenever mp_b is updated. This is updated in place, starting
        from the existing values. 

    cov : array_like
        initial sequence of co-moments

    df_a : array_like
        first difference equation for the first array "a".

    df_b : array_like
        first difference equation for the second array "b".

    dg_a : array_like
        Second difference equation for the first array "a".

    dg_b : array_like
        Second difference equation for the second array "b".

    sig_a : array_like
        Sequence of reciprocal mean centered norms for each window in sequence "a".

    sig_b : array_like
        Sequence of reciprocal mean centered norms for each window in sequence "b".

    offset_a : array_like
        Offset from the beginning of array "a" to the first position represented by 
        cov, df_a, dg_a, and sig_a.

    offset_b : array_like
        Offset from the beginning of array "b" to the first position represented by 
        cov, df_b, dg_b, and sig_b.

    Returns
    -------

    """

    subseqct_a = sig_a.shape[0]
    subseqct_b = sig_b.shape[0]

    assert(subseqct_a == mp_a.shape[0] == cov.shape[0] == sig_a.shape[0] == df_a.shape[0] + 1 == dg_a.shape[0] + 1)
    assert(subseqct_b == mp_b.shape[0] == sig_b.shape[0] == df_b.shape[0] + 1 == dg_b.shape[0] + 1)
  
    ab_compare(mp_a, mp_b, mpi_a, mpi_b, cov, df_a, df_b, dg_a, dg_b, sig_a, sig_b, offset_a, offset_b) 
   

cdef void difference_equations(double[::1] df, double[::1] dg, double[::1] ts, double[::1] mu, Py_ssize_t w) nogil:
    """
    Function difference_equations sets up difference equations for a time series using the original formula.

    Parameters
    ----------

    df : array_like
        Output buffer for first difference equation.

    dg : array_like
        Output buffer for second difference equation.

    ts : array_like
        The input time series.

    mu : array_like
        The mean of each overlapping length "w" window of "ts"

    w : int
        The window (or subsequence) length used.

    Returns
    -------

    """

    cdef Py_ssize_t i,j
    cdef Py_ssize_t subseqct = ts.shape[0] - w + 1
    # Cython appears to have an issue with allocating arrays via a call to Numpy in a Python interface which extends a pyx file.
    if not (subseqct - 1 == df.shape[0] == dg.shape[0]):
        raise ValueError(f'Mismatched dimensions between df or dg and subsequence count, expected {subseqct} got df: {df.shape[0]} dg: {dg.shape[0]}')

    for i in range(w, ts.shape[0]):
        j = i - w
        df[j] = (0.5 * (ts[i] - ts[i - w]))
        dg[j] = (ts[i] - mu[j + 1]) + (ts[j] - mu[j])


cdef void cross_cov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] cmpseq) nogil:
    """
    Function cross_cov provides a low level interface and reference implementation for co-moment calculations. 
    It's assumed that any mean centering of the second operand is done externally.

    Parameters
    ----------
    
    out : array_like
        A buffer for the output sequence.
    
    ts : array_like
        The input time series.
    
    mu : array_like
        The mean of each window of ts. Window length is inferred from the length of the time series and the length of the output sequence.
    
    cmpseq : array_like
        The comparison sequence.
    
    Returns
    -------

    """
    
    cdef Py_ssize_t subseqct = out.shape[0]
    cdef double accum, m_
    if subseqct != mu.shape[0]:
        raise ValueError
    elif cmpseq.shape[0] != ts.shape[0] - subseqct + 1:
        raise ValueError
    cdef Py_ssize_t i, j
    for i in range(subseqct):
        accum = 0.0
        m_ = mu[i]
        for j in range(cmpseq.shape[0]):
            accum += (ts[i + j] - m_) * cmpseq[j]
        out[i] = accum


cdef void self_compare(double[::1] mp, np.int_t[::1] mpi, double [::1] cov, double[::1] df, double[::1] dg, double[::1] sig, Py_ssize_t subseqlen, Py_ssize_t minlag, Py_ssize_t index_offset) nogil:
    """
    Function self_compare provides a low level interface and reference implementation for block matrix profile calculations involving a single time series. 
    This is intended for internal use.

    Parameters
    ----------
    
    mp : array_like
        A buffer for the output profile sequence. This is updated in place, starting from its initial input values.
    
    mpi : array_like
        A buffer for the output profile index sequence. This is updated in place, starting from its initial input values.
        
    cov : array_like
        A sequence of initial co-moments, assumed to be computed between window[0] and each window[minlag]....window[profile_len].
    
    df : array_like
        The first difference equation, used to update co-moment calculations.
    
    dg : array_like
        The second difference equation, used to update co-moment calculations.
    
    sig : array_like
        A seqeuence containing the reciprocal of the centered L2 norm of each window in the input time series.
    
    minlag : integer
        The minimum separation between two windows, which should result in a comparison. It corresponds to the offset used in cov[0]. 
    
    index_offset: integer
        The value of the smallest absolute index with respect to the overall time series. If this is nonzero, it means that mp, mpi, df, dg, and sig are sub-arrays
        corresponding to array[index_offset: index_offset + sub_array_length].
        
    Returns
    -------

    """

    cdef Py_ssize_t profile_len = mp.shape[0]
    cdef Py_ssize_t diag, offset, col, i
    cdef double c, c_cmp

    for i in range(cov.shape[0]):
        c = cov[i]
        diag = i + minlag
        for offset in range(profile_len - diag):
            col = offset + diag
            if offset > 0:
                c = c + df[offset-1] * dg[col-1] + df[col-1] * dg[offset-1]
            c_cmp = c * sig[offset] * sig[col]
            if c_cmp > mp[offset]:
                mp[offset] = c_cmp
                mpi[offset] = col + index_offset
            if c_cmp > mp[col]:
                if c_cmp > 1.0:
                    c_cmp = 1.0
                mp[col] = c_cmp
                mpi[col] = offset + index_offset
        cov[i] = c

cdef void ab_compare(double[::1] mp_a, double[::1] mp_b, np.int_t[::1] mpi_a, np.int_t[::1] mpi_b, double [::1] cov, double[::1] df_a, double[::1] df_b, double[::1] dg_a, double[::1] dg_b, double[::1] sig_a, double[::1] sig_b, Py_ssize_t offset_a, Py_ssize_t offset_b) nogil:
    """
    Function ab_cmp provides a low level interface and reference implementation for block matrix profile calculations. 
    This is intended for internal use.

    Parameters
    ----------
    
    mp_a : array_like
        A buffer for the nearest neighbor similarity of the first output profile sequence taken with respect to the second input time series (b). Nearest neighbor 
        similarities are updated in place starting from their initial values.
    
    mp_b : array_like
        A buffer for the nearest neighbor similarity of the second output profile sequence taken with respect to the first input time series (a). Nearest neighbor 
        similarities are updated in place starting from their initial values.
    
    mpi_a : array_like
        A buffer for the output index sequence which stores the index of each nearest neighbor subsequence from the second time series (b).
    
    mpi_b : array_like
        A buffer for the output index sequence which stores the index of each nearest neighbor subsequence from the first time series (a).
        
    cov : array_like
        A sequence of initial co-moments betweeen windows (or subsequences) of the first time series (a) and windows of the second time series (b).
    
    df_a : array_like
        The first sequence of difference equations for the first input sequence.
    
    df_b : array_like
        The first sequence of difference equations for the second input sequence.
    
    dg_a : array_like
        The second sequence of difference equations for the first input sequence.
    
    dg_b : array_like
        The second sequence of difference equations for the second input sequence.   
    
    sig_a : array_like
        The reciprocal of the mean centered L2 norm of each window for the first input sequence.
    
    sig_b : array_like
        The reciprocal of the mean centered L2 norm of each window for the second input sequence. 
    
    index_offset_a: integer
        The value of the smallest absolute index with respect to the full time series "a".
        This is 0 if comparisons start from the beginning. It's assumed that if this is nonzero
        it's assumed that the views of mp_a, mpi_a, df_a, dg_a, and sig_a start from position index_offset with respect to the overall time series.
    
    index_offset_b: integer
        The value of the smallest absolute index with respect to the full time series "b".
        This is 0 if comparisons start from the beginning. It's assumed that if this is nonzero
        it's assumed that the views of mp_b, mpi_b, df_b, dg_b, and sig_b start from position index_offset with respect to the overall time series.
    
    Returns

    -------

    """

    cdef Py_ssize_t profile_len_a = mp_a.shape[0]
    cdef Py_ssize_t profile_len_b = mp_b.shape[0]
    cdef double cov_, corr_
    cdef Py_ssize_t mx, i, j, k

    for i in range(cov.shape[0]):
        mx = (profile_len_a - i) if (profile_len_a - i) < profile_len_b else profile_len_b
        cov_ = cov[i]
        for j in range(mx):
            k = j + i
            if j > 0:
                cov_ = cov_ + df_a[k-1] * dg_b[j-1] + dg_a[k-1] * df_b[j-1]
            corr_ = cov_ * sig_a[k] * sig_b[j]
            if corr_ > mp_a[k]:
                mp_a[k] = corr_
                mpi_a[k] = j + offset_b
            if corr_ > mp_b[j]:
                mp_b[j] = corr_
                mpi_b[j] = k + offset_a
        cov[i] = cov_
