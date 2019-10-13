#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.cympx import mpx as cympx


def mpx(ts, w, cross_correlation=False, n_jobs=-1):
    """
    The MPX algorithm computes the matrix profile without using the FFT. Right
    now it only supports single dimension self joins.

    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix profile for.
    w : int
        The window size.
    cross_correlation : bool, Default=False
        Setermine if cross_correlation distance should be returned. It defaults
        to Euclidean Distance.
    n_jobs : int, Default all
        Number of cpu cores to use.
    
    Returns
    -------
    (array_like, array_like) :
        The matrix profile (distance profile, profile index).
    """
    ts = core.to_np_array(ts).astype('d')
    n_jobs = core.valid_n_jobs(n_jobs)
    mp, mpi = cympx(ts, w, int(cross_correlation), n_jobs)

    return (np.asarray(mp), np.asarray(mpi))