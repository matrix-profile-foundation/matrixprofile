#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np

from matrixprofile import core


def idealized_arc_curve(width, index):
    """
    Returns the value at x for the parabola of width n and height n / 2.
    Formula taken from https://www.desmos.com/calculator/awtnrxh6rk.

    Parameters
    ----------
    width : int
        Length of the time series to calculate the parabola for.
    index : int
        location to compute the parabola value at.

    Returns
    -------
    float : y
        The value at index for the parabola.

    """
    height = width / 2
    c = width / 2
    b = height
    a = height / (width / 2) ** 2
    y = -(a * (index - c) ** 2) + b

    return y


def fluss(profile):
    """
    Computes the corrected arc curve (CAC) for the MatrixProfile index. This
    algorithm is provides Fast Low-cost Unipotent Semantic Segmantation.

    Parameters
    ----------
    profile : dict
        Data structure from a MatrixProfile algorithm.

    Returns
    -------
    array_like : corrected_arc_curve
        The corrected arc curve for the profile.

    """
    if not core.is_mp_obj(profile):
        raise ValueError('profile must be a MatrixProfile structure')

    mpi = profile.get('pi')
    w = profile.get('w')

    n = len(mpi)
    nnmark = np.zeros(n)

    # find the number of additional arcs starting to cross over each index
    for i in range(n):
        mpi_val = mpi[i]
        small = int(min(i, mpi_val))
        large = int(max(i, mpi_val))
        nnmark[small + 1] = nnmark[small + 1] + 1
        nnmark[large] = nnmark[large] - 1

    # cumulatively sum all crossing arcs at each index
    cross_count = np.cumsum(nnmark)

    # compute ideal arc curve for all indices
    idealized = np.apply_along_axis(lambda i: idealized_arc_curve(n, i), 0, np.arange(0, n))
    idealized = cross_count / idealized

    # correct the arc curve so that it is between 0 and 1
    idealized[idealized > 1] = 1
    corrected_arc_curve = idealized

    # correct the head and tail with the window size
    corrected_arc_curve[:w] = 1
    corrected_arc_curve[-w:] = 1

    return corrected_arc_curve


def extract_regimes(profile, num_regimes=3):
    """
    Given a MatrixProfile, compute the corrected arc curve and extract
    the desired number of regimes. Regimes are computed with an exclusion
    zone of 5 * window size per the authors. 

    The author states:
        This exclusion zone is based on an assumption that regimes will have
        multiple repetitions; FLUSS is not able to segment single gesture 
        patterns.

    Parameters
    ----------
    profile : dict
        Data structure from a MatrixProfile algorithm.
    num_regimes : int
        The desired number of regimes to find.

    Returns
    -------
    dict : profile
        The original MatrixProfile object with additional keys containing.

        >>> {
        >>> 	'cac': The corrected arc curve
        >>> 	'cac_ez': The exclusion zone used
        >>> 	'regimes': Array of starting indices indicating a regime.
        >>> }

    """
    if not core.is_mp_obj(profile):
        raise ValueError('profile must be a MatrixProfile structure')

    cac = profile.get('cac')
    window_size = profile.get('w')
    ez = window_size * 5

    # compute the CAC if needed
    if isinstance(cac, type(None)):
        cac = fluss(profile)
        profile['cac'] = cac

    regimes = []
    tmp = np.copy(cac)
    n = len(tmp)
    
    for _ in range(num_regimes):
        min_index = np.argmin(tmp)
        regimes.append(min_index)
        
        # apply exclusion zone
        ez_start = np.max([0, min_index - ez])
        ez_end = np.min([n, min_index + ez])
        tmp[ez_start:ez_end] = np.inf

    profile['regimes'] = np.array(regimes, dtype=int)
    profile['cac_ez'] = ez

    return profile
