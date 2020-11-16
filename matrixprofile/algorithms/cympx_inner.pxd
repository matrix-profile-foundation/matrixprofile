cimport numpy as np


cdef void diff_eqns(double[::1] df, double[::1] dg, double[::1] ts, double[::1] mu, Py_ssize_t w) nogil

cdef void cross_cov(double[::1] out, double[::1] ts, double[::1] mu, double[::1] cmpseq) nogil

cdef void self_cmp(double[::1] mp, np.int_t[::1] mpi, double [::1] cov, double[::1] df, double[::1] dg, double[::1] sig, Py_ssize_t subseqlen, Py_ssize_t minsep, Py_ssize_t index_offset) nogil
    
cdef void ab_cmp(double[::1] mp_a, double[::1] mp_b, np.int_t[::1] mpi_a, np.int_t[::1] mpi_b, double[::1] cov, double[::1] df_a, double[::1] df_b, double[::1] dg_a, double[::1] dg_b, double[::1] sig_a, double[::1] sig_b, Py_ssize_t offset_a, Py_ssize_t offset_b) nogil
