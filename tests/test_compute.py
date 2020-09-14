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

from matrixprofile import compute

import matrixprofile
MODULE_PATH = matrixprofile.__path__[0]


def test_compute_mp_exact_no_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile = compute(ts, windows=m)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_mp_exact_with_query():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    query = ts[100:200]
    m = 32

    profile = compute(ts, windows=m, query=query)
    assert(profile['algorithm'] == 'mpx')
    assert(profile['w'] == 32)
    np.testing.assert_equal(profile['data']['query'], query)
    assert(profile['join'] == True)
    assert(profile['sample_pct'] == 1)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_mp_approximate():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    m = 32

    profile = compute(ts, windows=m, sample_pct=0.5)
    assert(profile['algorithm'] == 'scrimp++')
    assert(profile['w'] == 32)
    assert(profile['data']['query'] == None)
    assert(profile['join'] == False)
    assert(profile['sample_pct'] == 0.5)
    assert(profile['class'] == 'MatrixProfile')


def test_compute_pmp_no_sample_pct_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    windows = np.arange(8, 32)

    profile = compute(ts, windows=windows)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    np.testing.assert_equal(profile['windows'], windows)


def test_compute_pmp_sample_pct_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    windows = np.arange(8, 32)

    profile = compute(ts, windows=windows, sample_pct=1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')
    assert(profile['sample_pct'] == 1)
    np.testing.assert_equal(profile['windows'], windows)


def test_compute_pmp_no_windows():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile = compute(ts)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')

    # sample pct is ignored when windows are provided and defaults to 1
    assert(profile['sample_pct'] == 1)


def test_compute_pmp_no_windows_sample_pct():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))

    profile = compute(ts, sample_pct=0.1)
    assert(profile['algorithm'] == 'skimp')
    assert(profile['class'] == 'PMP')

    # sample pct is ignored when windows are provided and defaults to 1
    assert(profile['sample_pct'] == 0.1)


def test_compute_mp_invalid_windows():
    ts = [3., 3., 3., 3., 3., 3., 3., 3.]

    with pytest.raises(ValueError) as excinfo:
        w = 0
        compute(ts, windows=w)
        assert 'Compute requires all window sizes to be greater than 3!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        w = 3
        compute(ts, windows=w)
        assert 'Compute requires all window sizes to be greater than 3!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        w = [4, 0]
        compute(ts, windows=w)
        assert 'Compute requires all window sizes to be greater than 3!' \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        w = [4, 3]
        compute(ts, windows=w)
        assert 'Compute requires all window sizes to be greater than 3!' \
            in str(excinfo.value)


def test_preprocess():
    ts = np.array([np.nan, np.inf, np.inf, np.nan, np.inf, 2, 3, 2, 3, 1, 2, 3, 4, 2,
                   np.nan, np.inf, 4, 2, 3, 4, 5, 6, 7, 8, 3, 4, 2, 3, 4, 5, 6, 7, 6,
                   5, 4, 3, np.nan, np.nan, np.inf, np.nan, np.inf, np.nan])
    m = 6
    preprocessing_kwargs = {
        'window': 5,
        'impute_method': 'median',
        'impute_direction': 'backward',
        'add_noise': False
    }

    profile = compute(ts, windows=m, preprocessing_kwargs=preprocessing_kwargs)
    preprocessed_ts = profile['data']['ts']
    assert(np.any(np.isnan(preprocessed_ts)) == False)
    assert(np.any(np.isinf(preprocessed_ts)) == False)

    # if preprocessing_kwargs=None, we disable the preprocessing procedure.
    profile = compute(ts, windows=m, preprocessing_kwargs=None)
    unprocessed_ts = profile['data']['ts']
    assert(np.any(np.isnan(unprocessed_ts)) == True)
    assert(np.any(np.isinf(unprocessed_ts)) == True)

    # check if preprocessing_kwargs is None by default.
    profile = compute(ts, windows=m)
    unprocessed_ts = profile['data']['ts']
    assert(np.any(np.isnan(unprocessed_ts)) == True)
    assert(np.any(np.isinf(unprocessed_ts)) == True)

    with pytest.raises(ValueError) as excinfo:
        compute(ts, windows=m, preprocessing_kwargs=1)
        assert "The parameter 'preprocessing_kwargs' is not dict like!" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {
            'win': 5,
            'impute_dir': 'backward',
        }
        compute(ts, windows=m, preprocessing_kwargs=preprocessing_kwargs)
        assert "invalid key(s) for preprocessing_kwargs! valid key(s) should include " \
               "{'impute_direction', 'add_noise', 'impute_method', 'window'}" \
            in str(excinfo.value)