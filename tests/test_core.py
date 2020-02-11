#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

"""Tests for `mass_ts` package."""

import os

import pytest

import numpy as np

from matrixprofile import core


def test_is_array_like_invalid():
    assert(core.is_array_like(1) == False)
    assert(core.is_array_like('adf') == False)
    assert(core.is_array_like({'a': 1}) == False)
    assert(core.is_array_like(set([1, 2, 3])) == False)


def test_is_array_like_valid():
    assert(core.is_array_like(np.array([1])) == True)
    assert(core.is_array_like([1, ]) == True)
    assert(core.is_array_like((1, 2,)) == True)


def test_is_one_dimensional_invalid():
    a = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])
    assert(core.is_one_dimensional(a) == False)


def test_is_one_dimensional_valid():
    a = np.array([1, 2, 3, 4])
    assert(core.is_one_dimensional(a) == True)


def test_to_np_array_exception():
    with pytest.raises(ValueError) as excinfo:
        core.to_np_array('s')
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        core.to_np_array(1)
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        core.to_np_array(set([1, 2, 3]))
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)


def test_to_np_array_valid():
    actual = core.to_np_array([1, 2, 3])
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = core.to_np_array((1, 2, 3,))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)

    actual = core.to_np_array(np.array([1, 2, 3]))
    desired = np.array([1, 2, 3])
    np.testing.assert_equal(actual, desired)


def test_precheck_series_and_query_1d_valid():
    ts = [1, 2, 3, 4, 5, 6, 7, 8]
    q = [1, 2, 3, 4]

    actual_ts, actual_q = core.precheck_series_and_query_1d(ts, q)
    np.testing.assert_equal(actual_ts, np.array(ts))
    np.testing.assert_equal(actual_q, np.array(q))


def test_precheck_series_and_query_1d_invalid():
    with pytest.raises(ValueError) as excinfo:
        core.precheck_series_and_query_1d('1', [1, 2, 3])
        assert 'Invalid ts value given. Must be array_like!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        core.precheck_series_and_query_1d([1, 2, 3], '1')
        assert 'Invalid query value given. Must be array_like!' \
            in str(excinfo.value)


def test_is_similarity_join():
    a = [1, 2, 3]
    b = [4, 5, 6, 7]

    assert(core.is_similarity_join(a, b))
    assert(not core.is_similarity_join(a, None))


def test_get_profile_length():
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [1, 2, 3, 4, 5]
    m = 4
    desired = 8 - 4 + 1
    actual = core.get_profile_length(a, b, m)

    assert(desired == actual)


def test_find_skip_locations():
    a = np.array([1, np.inf, np.nan, 4, 5, 6, 7, 8])
    desired = np.array([True, True, True, False, False])
    actual = core.find_skip_locations(a, 5, 4)

    np.testing.assert_equal(actual, desired)


def test_clean_nan_inf():
    a = np.array([np.nan, 1, np.inf, 2])
    desired = np.array([0, 1, 0, 2])
    actual = core.clean_nan_inf(a)

    np.testing.assert_equal(actual, desired)

    with pytest.raises(ValueError) as excinfo:
        core.clean_nan_inf(None)
        assert 'Unable to convert to np.ndarray!' in str(excinfo.value)


def test_rolling_window():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = core.rolling_window(a, 3)
    desired = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6]
    ])

    np.testing.assert_equal(actual, desired)


def test_moving_average():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = core.moving_average(a, 3)
    desired = np.array([2., 3., 4., 5.])

    np.testing.assert_equal(actual, desired)


def test_moving_std():
    a = np.array([1, 2, 3, 4, 5, 6])
    actual = core.moving_std(a, 3)
    desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(actual, desired)


def test_moving_avg_std():
    a = np.array([1, 2, 3, 4, 5, 6])
    mu, std = core.moving_avg_std(a, 3)
    mu_desired = np.array([2., 3., 4., 5.])
    std_desired = np.array([0.81649658, 0.81649658, 0.81649658, 0.81649658])

    np.testing.assert_almost_equal(mu, mu_desired)
    np.testing.assert_almost_equal(std, std_desired)


def test_fft_convolve():
    query = np.array([1, 2, 3, 4])
    ts = np.array([4, 5, 6, 1, 2, 3, 8, 9, 1, 7, 8, 15, 20])

    dp = core.fft_convolve(ts, query)
    dp_desired = np.array([36, 28, 26, 46, 68, 50, 57, 64, 99, 148])

    np.testing.assert_almost_equal(dp, dp_desired)


def test_sliding_dot_product():
    query = np.array([1, 2, 3, 4])
    ts = np.array([4, 5, 6, 1, 2, 3, 8, 9, 1, 7, 8, 15, 20])

    dp = core.sliding_dot_product(ts, query)
    dp_desired = np.array([36, 28, 26, 46, 68, 50, 57, 64, 99, 148])

    np.testing.assert_almost_equal(dp, dp_desired)


def test_generate_batch_jobs_single_job():
    profile_length = 9
    n_jobs = 1

    desired = [[0, 9]]
    actual = list(core.generate_batch_jobs(profile_length, n_jobs))

    np.testing.assert_equal(actual, desired)


def test_generate_batch_jobs_small_ts_many_jobs():
    profile_length = 9
    n_jobs = 12

    desired = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    actual = list(core.generate_batch_jobs(profile_length, n_jobs))

    np.testing.assert_equal(actual, desired)


def test_generate_batch_jobs_multiple_jobs():
    profile_length = 9
    n_jobs = 4

    desired = [(0, 3), (3, 6), (6, 9)]
    actual = list(core.generate_batch_jobs(profile_length, n_jobs))

    np.testing.assert_equal(actual, desired)


def test_is_nan_inf():
    assert(core.is_nan_inf(np.inf) == True)
    assert(core.is_nan_inf(np.nan) == True)
    assert(core.is_nan_inf(100) == False)


def test_is_not_nan_inf():
    assert(core.is_not_nan_inf(np.inf) == False)
    assert(core.is_not_nan_inf(np.nan) == False)
    assert(core.is_not_nan_inf(100) == True)


def test_nan_inf_indices_all_inf():
    a = np.array([np.inf, np.inf, np.inf])
    desired = np.array([True, True, True])
    actual = core.nan_inf_indices(a)

    np.testing.assert_equal(desired, actual)


def test_nan_inf_indices_all_nan():
    a = np.array([np.nan, np.nan, np.nan])
    desired = np.array([True, True, True])
    actual = core.nan_inf_indices(a)

    np.testing.assert_equal(desired, actual)


def test_nan_inf_indices_some_true():
    a = np.array([1, np.inf, np.nan])
    desired = np.array([False, True, True])
    actual = core.nan_inf_indices(a)

    np.testing.assert_equal(desired, actual)


def test_nan_inf_indices_all_false():
    a = np.array([1, 1, 1])
    desired = np.array([False, False, False])
    actual = core.nan_inf_indices(a)

    np.testing.assert_equal(desired, actual)


def test_pearson_to_euclidean_1d():
    a = np.array([0.23, 0.5, 0.34, 0.67, 0.88])
    w = [4,]
    desired = np.array([2.48193473, 2, 2.29782506, 1.62480768, 0.9797959])
    actual = core.pearson_to_euclidean(a, w)

    np.testing.assert_almost_equal(desired, actual)


def test_pearson_to_euclidean_2d():
    a = np.array([[0.23, 0.5, 0.34, 0.67, 0.88],
                  [0.23, 0.5, 0.34, 0.67, 0.88]])
    w = [4, 4]
    desired = np.array([[2.48193473, 2, 2.29782506, 1.62480768, 0.9797959],
                        [2.48193473, 2, 2.29782506, 1.62480768, 0.9797959]])
    actual = core.pearson_to_euclidean(a, w)

    np.testing.assert_almost_equal(desired, actual)


def test_is_mp_obj():
    assert(True == core.is_mp_obj({'class': 'MatrixProfile'}))
    assert(False == core.is_mp_obj('s'))
    assert(False == core.is_mp_obj({}))


def test_is_pmp_obj():
    assert(True == core.is_pmp_obj({'class': 'PMP'}))
    assert(False == core.is_pmp_obj('s'))
    assert(False == core.is_pmp_obj({}))


def test_is_mp_or_pmp_obj():
    assert(True == core.is_mp_or_pmp_obj({'class': 'PMP'}))
    assert(True == core.is_mp_or_pmp_obj({'class': 'MatrixProfile'}))
    assert(False == core.is_mp_or_pmp_obj('s'))
    assert(False == core.is_mp_or_pmp_obj({}))


def test_moving_min():
    a = np.array([1, 1, 1, 2, 0, 2])
    desired = np.array([1, 0, 0])
    actual = core.moving_min(a, window=4)

    np.testing.assert_equal(desired, actual)

    a = np.array([1, 0, 1, 2, 0, 2])
    desired = np.array([0, 0, 0])
    actual = core.moving_min(a, window=4)

    np.testing.assert_equal(desired, actual)

    a = np.array([1, 1, 1, 2, 0, 2])
    desired = np.array([1, 1, 1, 0, 0])
    actual = core.moving_min(a, window=2)

    np.testing.assert_equal(desired, actual)


def test_moving_max():
    a = np.array([1, 1, 1, 2, 0, 2])
    desired = np.array([2, 2, 2])
    actual = core.moving_max(a, window=4)

    np.testing.assert_equal(desired, actual)

    a = np.array([1, 0, 1, 2, 0, 2])
    desired = np.array([2, 2, 2])
    actual = core.moving_max(a, window=4)

    np.testing.assert_equal(desired, actual)

    a = np.array([1, 1, 1, 2, 0, 2])
    desired = np.array([1, 1, 2, 2, 2])
    actual = core.moving_max(a, window=2)

    np.testing.assert_equal(desired, actual)