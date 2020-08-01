import numba
import numpy as np
import numpy as np
import  matrixprofile.cycore as mpcc



def mpx_exper3(timeseries, subseqlen):
    """ Experimental item, out of place so that I can build without installing """
    subseqcount = len(timeseries) - subseqlen + 1
    
    minlag = subseqlen // 4   
    mu, invn = mpcc.muinvn(timeseries, subseqlen)
    mu_s, _ = mpcc.muinvn(timeseries, subseqlen-1)

    # bwd indicates trailing difference, fwd leading difference
    dr_bwd = timeseries[:subseqcount-1] - mu[:subseqcount-1]
    dc_bwd = timeseries[:subseqcount-1] - mu_s[1:subseqcount]
    dr_fwd = timeseries[subseqlen-1:] - mu
    dc_fwd = timeseries[subseqlen-1:] - mu_s[:subseqcount]
    
    mprof = np.full(subseqcount, -1)
    mprofidx = np.full(subseqcount, np.nan)
    # numba can't create this stuff internally in most cases
    mpx_exper3_inner(timeseries, subseqlen, subseqcount, minlag, dr_bwd, dc_bwd, dr_fwd, dc_fwd, mu, invn, mprof, mprofidx)
    return mprof, mprofidx    

@numba.jit
def mpx_exper3_inner(timeseries, subseqlen, subseqcount, minlag, dr_bwd, dc_bwd, dr_fwd, dc_fwd, mu, invn, mprof, mprofidx):
    for diag in range(minlag, subseqcount):
        cov_ = np.dot(timeseries[diag:diag+subseqlen] - mu[diag], timeseries[:subseqlen] - mu[0])
        for row, (rbwd, cbwd, rfwd, cfwd, invnr, invnc, mpr, mpc) in enumerate(zip(dr_bwd, dc_bwd[diag:], dr_fwd, dc_fwd[diag:], invn, invn[diag:], mprof, mprof[diag:])):
            col = diag + row
            corr_ = cov_ * invnr * invnc
            if corr_ > mpr:
                mprof[row] = corr_
                mprofidx[row] = col
            if corr_ > mpc:
                mprof[col] = corr_
                mprofidx[col] = row
            # This creates minor problems with optimization that are difficult for compilers
            # for a compiler to schedule this well, it needs to load cov either before the first iteration
            # or here, then use those values the next time the loop is traversed. Otherwise you load a stream of values
            # twice instead of once. Common avx unrolling factor is 32, common neon is 16, this can impact runtime since this 
            # is the main loop
            if col < subseqcount - 1:
                cov_ -= rbwd * cbwd
                cov_ += rfwd * cfwd
          

def mpx_exper4(timeseries, subseqlen):
    subseqcount = len(timeseries) - subseqlen + 1
    minlag = subseqlen // 4
    mu, invnorm = mpcc.muinvn(timeseries, subseqlen)
    mu_s, _ = mpcc.muinvn(timeseries[:-1], subseqlen-1)

    # skip the leading 0, just avoid making an update step on initialization
    dr_bwd = timeseries[: subseqcount - 1] - mu[:-1]
    dc_bwd = timeseries[: subseqcount - 1] - mu_s[1:]
    dr_fwd = timeseries[subseqlen:] - mu[1:]
    dc_fwd = timeseries[subseqlen:] - mu_s[1:]

    mprof = np.full(subseqcount, -1.0, dtype='d')
    mprofidx = np.full(subseqcount, -1, dtype='i')
    first_row = timeseries[:subseqlen] - mu[0]  

    for diag in range(minlag, subseqcount):
        cov_ = np.dot(timeseries[diag:diag+subseqlen] - mu[diag], first_row)
        for row in range(subseqcount - diag):
            col = diag + row
            if row > 0: # for tiled versions, this changes from 0 to whatever first row
                cov_ -= dr_bwd[row-1] * dc_bwd[col-1]
                cov_ += dr_fwd[row-1] * dc_fwd[col-1]
            corr_ = cov_ * invnorm[row] * invnorm[col]
            if corr_ > mprof[row]:
                mprof[row] = corr_
                mprofidx[row] = col 
            if corr_ > mprof[col]:
                mprof[col] = corr_
                mprofidx[col] = row
    return mprof, mprofidx


def mpx_exper3(timeseries, subseqlen):
    """ Experimental item, out of place so that I can build without installing """
    subseqcount = len(timeseries) - subseqlen + 1
    
    minlag = subseqlen // 4   
    mu, invn = mpcc.muinvn(timeseries, subseqlen)
    mu_s, _ = mpcc.muinvn(timeseries, subseqlen-1)

    # bwd indicates trailing difference, fwd leading difference
    dr_bwd = timeseries[:subseqcount-1] - mu[:subseqcount-1]
    dc_bwd = timeseries[:subseqcount-1] - mu_s[1:subseqcount]
    dr_fwd = timeseries[subseqlen-1:] - mu
    dc_fwd = timeseries[subseqlen-1:] - mu_s[:subseqcount]
    
    mprof = np.full(subseqcount, -1.0, dtype='d')
    mprofidx = np.full(subseqcount, -1, dtype='i')
 
    for diag in range(minlag, subseqcount):
        cov_ = np.dot(timeseries[diag:diag+subseqlen] - mu[diag], timeseries[:subseqlen] - mu[0])
        for row, (rbwd, cbwd, rfwd, cfwd, invnr, invnc, mpr, mpc) in enumerate(zip(dr_bwd, dc_bwd[diag:], dr_fwd[1:], dc_fwd[diag+1:], invn, invn[diag:], mprof, mprof[diag:])):
            col = diag + row
            corr_ = cov_ * invnr * invnc
            if corr_ > mpr:
                mprof[row] = corr_
                mprofidx[row] = col
            if corr_ > mpc:
                mprof[col] = corr_
                mprofidx[col] = row
            # This creates minor problems with optimization that are difficult for compilers
            # for a compiler to schedule this well, it needs to load cov either before the first iteration
            # or here, then use those values the next time the loop is traversed. Otherwise you load a stream of values
            # twice instead of once. Common avx unrolling factor is 32, common neon is 16, this can impact runtime since this 
            # is the main loop
            cov_ -= rbwd * cbwd
            cov_ += rfwd * cfwd
        col = subseqcount - 1
        row = col - row
        corr_ = cov_ * invn[row] * invn[col]
        if corr_ > mprof[row]:
            mprof[row] = corr_
            mprofidx[row] = col
        if corr_ > mprof[col]:
            mprof[col] = corr_
            mprofidx[col] = row
    return mprof, mprofidx



def mpx_exper2(timeseries, subseqlen):
    """ Experimental item, out of place so that I can build without installing """
    subseqcount = len(timeseries) - subseqlen + 1
    
    minlag = subseqlen // 4   
    mu, invn = mpcc.muinvn(timeseries, subseqlen)
    mu_s, _ = mpcc.muinvn(timeseries, subseqlen-1)

    # bwd indicates trailing difference, fwd leading difference
    dr_bwd = timeseries[:subseqcount-1] - mu[:subseqcount-1]
    dc_bwd = timeseries[:subseqcount-1] - mu_s[1:subseqcount]
    dr_fwd = timeseries[subseqlen-1:] - mu
    dc_fwd = timeseries[subseqlen-1:] - mu_s[:subseqcount]
    
    mprof = np.full(subseqcount, -1.0, dtype='d')
    mprofidx = np.full(subseqcount, -1, dtype='i')
 
    for diag in range(minlag, subseqcount):
        cov_ = np.dot(timeseries[diag:diag+subseqlen] - mu[diag], timeseries[:subseqlen] - mu[0])
        for row, (rbwd, cbwd, rfwd, cfwd, invnr, invnc, mpr, mpc) in enumerate(zip(dr_bwd, dc_bwd[diag:], dr_fwd[1:], dc_fwd[diag+1:], invn, invn[diag:], mprof, mprof[diag:])):
            col = diag + row
            corr_ = cov_ * invnr * invnc
            if corr_ > mpr:
                mprof[row] = corr_
                mprofidx[row] = col
            if corr_ > mpc:
                mprof[col] = corr_
                mprofidx[col] = row
            # This creates minor problems with optimization that are difficult for compilers
            # for a compiler to schedule this well, it needs to load cov either before the first iteration
            # or here, then use those values the next time the loop is traversed. Otherwise you load a stream of values
            # twice instead of once. Common avx unrolling factor is 32, common neon is 16, this can impact runtime since this 
            # is the main loop
            cov_ -= rbwd * cbwd
            cov_ += rfwd * cfwd
        col = subseqcount - 1
        row = col - row
        corr_ = cov_ * invn[row] * invn[col]
        if corr_ > mprof[row]:
            mprof[row] = corr_
            mprofidx[row] = col
        if corr_ > mprof[col]:
            mprof[col] = corr_
            mprofidx[col] = row
    return mprof, mprofidx

