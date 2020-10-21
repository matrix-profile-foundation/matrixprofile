# -*- coding: utf-8 -*-
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

cimport numpy as np


cdef void diff_equations(double[::1] df, double[::1] dg, double[::1] ts, double[::1] mu, Py_ssize_t w) nogil:
    """
    Function diff_equations sets up difference equations for a time series using the original formula.

    Parameters
    ----------

    df: array_like
        output buffer for first difference equation

    dg: array_like
        output buffer for second difference equation

    ts : array_like
        The input time series

    mu : array_like
        The mean of each overlapping length "w" window of "ts"

    w : int
        The window (or subsequence) length used

    Returns
    -------

    dfdg: array_like
        First and second difference equations for the original formula used in mpx
        First row is df. Second is dg. 
        
        This layout is used because Cython has trouble
        placing arrays inside of tuples in any C interface.

    """
    cdef Py_ssize_t i,j
    cdef Py_ssize_t subseqct = ts.shape[0] - w + 1
    if not (subseqct == df.shape[0] == dg.shape[0]):
        raise ValueError('mismatched dimensions')

    df[0] = 0
    dg[0] = 0
    for i in range(w, ts.shape[0]):
        j = i - w
        df[j + 1] = (0.5 * (ts[i] - ts[i - w]))
        dg[j + 1] = (ts[i] - mu[j+ 1]) + (ts[j] - mu[j])


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
    cmpseq: array_like
        The comparison sequence.
    
    Returns
    -------
    None

    """
    cdef Py_ssize_t sseqct = out.shape[0]
    cdef double accum, m_
    if sseqct != mu.shape[0]:
        raise ValueError
    elif cmpseq.shape[0] != ts.shape[0] - sseqct + 1:
        raise ValueError
    cdef Py_ssize_t i, j
    for i in range(sseqct):
        accum = 0.0
        m_ = mu[i]
        for j in range(cmpseq.shape[0]):
            accum += (ts[i + j] - m_) * cmpseq[j]
        out[i] = accum


cdef void inner_self(double[::1] mp, np.int_t[::1] mpi, double [::1] cov, double[::1] df, double[::1] dg, double[::1] sig, Py_ssize_t subseqlen, Py_ssize_t minlag, Py_ssize_t index_offset) nogil:
    """
    Function inner_self provides a low level interface and reference implementation for block matrix profile calculations involving a single time series. 
    This is intended for internal use.

    Parameters
    ----------
    mp : array_like
        A buffer for the output sequence.
    mpi : array_like
        The input time series.
    cov : array_like
        A sequence of initial co-moments, assumed to be computed between window[0] and each window[minlag]....window[profile_len]
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
    None

    """
    cdef Py_ssize_t profile_len = mp.shape[0]
    cdef Py_ssize_t diag, offset, col, i
    cdef double c, c_cmp

    # Since this may be a partition of a larger problem, we iterate over
    # cov but set boundary conditions using 
    # Todo: need a test c
    for i in range(cov.shape[0]):
        c = cov[i]
        diag = i + minlag
        for offset in range(profile_len - diag):
            col = offset + diag
            c = c + df[offset] * dg[col] + df[col] * dg[offset]
            c_cmp = c * sig[offset] * sig[col]
            
            # update the distance profile and profile index
            if c_cmp > mp[offset]:
                mp[offset] = c_cmp
                mpi[offset] = col + index_offset
            
            if c_cmp > mp[col]:
                if c_cmp > 1.0:
                    c_cmp = 1.0
                mp[col] = c_cmp
                mpi[col] = offset + index_offset


cdef void inner_ab(double[::1] mp_a, double[::1] mp_b, np.int_t[::1] mpi_a, np.int_t[::1] mpi_b, double [::1] cov, double[::1] df_a, double[::1] df_b, double[::1] dg_a, double[::1] dg_b, double[::1] sig_a, double[::1] sig_b, Py_ssize_t offset_a, Py_ssize_t offset_b) nogil:
    """
    Function inner_ab provides a low level interface and reference implementation for block matrix profile calculations. 
    This is intended for internal use.

    Parameters
    ----------
    mp_a : array_like
        A buffer for the output profile sequence taken with respect to the "first" input sequence.
    mp_b : array_like
        A buffer for the output profile sequence taken with respect to the "second" input sequence.
    mpi_a : array_like
        A buffer for the output index sequence taken with respect to the "first" input sequence.
    mpi_b : array_like
        A buffer for the output index sequence taken with respect to the "second" input sequence.
    cov : array_like
        A sequence of initial co-moments betweeen windows (or subsequences) of time series "a" and windows of time series "b"
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
    None

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
            cov_ = cov_ + df_a[k] * dg_b[j] + dg_a[k] * df_b[j]
            corr_ = cov_ * sig_a[k] * sig_b[j]

            if corr_ > mp_a[k]:
                mp_a[k] = corr_
                mpi_a[k] = j + offset_b

            if corr_ > mp_b[j]:
                mp_b[j] = corr_
                mpi_b[j] = k + offset_a
