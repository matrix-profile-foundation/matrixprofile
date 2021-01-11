import numpy as np
import matrixprofile.streamingimpl as mps


def mpx(ts, w):
    ssct = ts.shape[0] - w + 1
    mu = np.empty(ssct, dtype='d')
    mu_s = np.empty(ssct, dtype='d') # window length w - 1 skipping first and last
    invn = np.empty(ssct, dtype='d')
    minlag = w // 4

    mps.windowed_mean(ts, mu, w)
    mps.windowed_mean(ts[:-1], mu_s, w-1)
    mps.windowed_invcnorm(ts, mu, invn, w)
    rbwd = ts[:ssct-1] - mu[:-1]
    cbwd = ts[:ssct-1] - mu_s[1:]
    rfwd = ts[w:] - mu[1:]
    cfwd = ts[w:] - mu_s[1:]

    mp = np.full(ssct, -1, dtype='d')
    mpi = np.full(ssct, -1, dtype='i')

    first_row = ts[:w] - mu[0]
    cov = np.empty(ssct - minlag, dtype='d')   
 
    mps.crosscov(cov, ts[minlag:],  mu[minlag:], first_row)

    # roffset only applies when this is tiled by row, since otherwise the index would be wrong
    mps.mpx_inner(cov, rbwd, rfwd, cbwd, cfwd, invn, mp, mpi, minlag, 0) 

    return mp, mpi
