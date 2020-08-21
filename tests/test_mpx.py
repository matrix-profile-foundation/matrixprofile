#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import os

import pytest

import numpy as np

from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.cympx import mpx_ab_parallel
import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]


def test_mpx_small_series_self_join_euclidean_single_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([1.9550, 1.9550, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0])
    desired_pi = np.array([4, 5, 6, 7, 8, 1, 2, 3, 4])

    profile = mpx(ts, w, cross_correlation=False, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_small_series_self_join_euclidean_multi_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([1.9550, 1.9550, 0.8739, 0, 0, 1.9550, 0.8739, 0, 0])
    desired_pi = np.array([4, 5, 6, 7, 8, 1, 2, 3, 4])

    profile = mpx(ts, w, cross_correlation=False, n_jobs=-1)
    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_small_series_self_join_pearson_single_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([0.522232967867094, 0.522232967867094, 0.904534033733291, 1, 1, 0.522232967867094, 0.904534033733291, 1, 1])
    desired_pi = np.array([4, 5, 6, 7, 8, 1, 2, 3, 4])

    profile = mpx(ts, w, cross_correlation=True, n_jobs=1)
    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_small_series_self_join_pearson_multi_threaded():
    ts = np.array([0, 1, 1, 1, 0, 0, 2, 1, 0, 0, 2, 1])
    w = 4
    desired = np.array([0.522232967867094, 0.522232967867094, 0.904534033733291, 1, 1, 0.522232967867094, 0.904534033733291, 1, 1])
    desired_pi = np.array([4, 5, 6, 7, 8, 1, 2, 3, 4])

    profile = mpx(ts, w, cross_correlation=True, n_jobs=-1)
    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_small_series_similarity_join_single_threaded():
    ts = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2, 4, 5, 1, 1, 9]).astype('d')
    query = np.array([0, 0, 1, 1, 2, 2, 4, 5]).astype('d')
    w = 4

    desired = np.array([
        2.36387589e+00, 2.82842712e+00, 2.17957574e+00, 6.40728972e-01,
        6.40728972e-01, 6.40728972e-01, 3.26103392e+00, 3.61947699e+00,
        3.39984131e+00, 0.00000000e+00, 4.21468485e-08, 0.00000000e+00,
        4.21468485e-08, 0.00000000e+00, 2.82842712e+00, 3.57109342e+00,
        1.73771570e+00
    ])
    desired_pi = np.array([0, 1, 4, 1, 1, 1, 2, 1, 4, 2, 1, 2, 3, 4, 2, 1, 3])

    profile = mpx(ts, w, cross_correlation=False, query=query, n_jobs=1)

    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_small_series_similarity_join_multi_threaded():
    ts = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2, 4, 5, 1, 1, 9]).astype('d')
    query = np.array([0, 0, 1, 1, 2, 2, 4, 5]).astype('d')
    w = 4

    desired = np.array([
        2.36387589e+00, 2.82842712e+00, 2.17957574e+00, 6.40728972e-01,
        6.40728972e-01, 6.40728972e-01, 3.26103392e+00, 3.61947699e+00,
        3.39984131e+00, 0.00000000e+00, 4.21468485e-08, 0.00000000e+00,
        4.21468485e-08, 0.00000000e+00, 2.82842712e+00, 3.57109342e+00,
        1.73771570e+00
    ])
    desired_pi = np.array([0, 1, 4, 1, 1, 1, 2, 1, 4, 2, 1, 2, 3, 4, 2, 1, 3])

    profile = mpx(ts, w, cross_correlation=False, query=query, n_jobs=-1)

    np.testing.assert_almost_equal(profile['mp'], desired, decimal=4)
    np.testing.assert_almost_equal(profile['pi'], desired_pi)


def test_mpx_similarity_join_matlab():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    tsb = ts[199:300]
    w = 32

    ml_mpa = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpa.txt'))
    ml_mpb = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpb.txt'))

    mpa, mpia, mpb, mpib = mpx_ab_parallel(ts, tsb, w, 0, 1)

    np.testing.assert_almost_equal(ml_mpa, mpa, decimal=4)
    np.testing.assert_almost_equal(ml_mpb, mpb, decimal=4)


def test_mpx_similarity_join_parallel_matlab():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    tsb = ts[199:300]
    w = 32

    ml_mpa = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpa.txt'))
    ml_mpb = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'mpx_ab_mpb.txt'))

    mpa, mpia, mpb, mpib = mpx_ab_parallel(ts, tsb, w, 0, 2)

    np.testing.assert_almost_equal(ml_mpa, mpa, decimal=4)
    np.testing.assert_almost_equal(ml_mpb, mpb, decimal=4)