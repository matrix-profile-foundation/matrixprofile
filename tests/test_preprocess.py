#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import numpy as np
import pytest

from matrixprofile.preprocess import preprocess
from matrixprofile.preprocess import is_subsequence_constant
from matrixprofile.preprocess import add_noise_to_series
from matrixprofile.preprocess import impute_missing
from matrixprofile.preprocess import validate_preprocess_kwargs


def test_valid_preprocess_kwargs():
    preprocessing_kwargs = {
        'window': 5,
        'impute_method': 'median',
        'impute_direction': 'backward',
        'add_noise': False
    }

    valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
    assert(valid_kwargs['window'] == 5)
    assert(valid_kwargs['impute_method'] == 'median')
    assert(valid_kwargs['impute_direction'] == 'backward')
    assert(valid_kwargs['add_noise'] == False)

    preprocessing_kwargs = {
        'window': 5,
        'add_noise': False
    }

    valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
    assert(valid_kwargs['window'] == 5)
    assert(valid_kwargs['impute_method'] == 'mean')
    assert(valid_kwargs['impute_direction'] == 'forward')
    assert(valid_kwargs['add_noise'] == False)

    valid_kwargs = validate_preprocess_kwargs(None)
    assert (valid_kwargs == None)


def test_invalid_preprocess_kwargs():
    with pytest.raises(ValueError) as excinfo:
        validate_preprocess_kwargs(preprocessing_kwargs = 1)
        assert "The parameter 'preprocessing_kwargs' is not dict like!" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {
            'win': 5,
            'impute_dir': 'backward',
        }
        validate_preprocess_kwargs(preprocessing_kwargs)
        assert "invalid key(s) for preprocessing_kwargs! valid key(s) should include " \
               "{'impute_direction', 'add_noise', 'impute_method', 'window'}" \
            in str(excinfo.value)


def test_is_subsequence_constant():
    ts = np.array([1, 1, 1, 1, 1, 1])
    assert (is_subsequence_constant(ts) == True)

    ts = np.array([1, 2, 1, 1, 1, 1])
    assert (is_subsequence_constant(ts) == False)


def test_add_noise_to_series():
    ts = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    temp = add_noise_to_series(ts)
    assert (np.all((temp - ts) >= 0) and np.all((temp - ts) < 0.0000009))


def test_impute_missing():
    ts = np.array([np.nan, np.nan, np.inf, np.nan, np.inf, np.inf, 4, 5, np.nan,
                   np.inf, np.nan, np.inf, np.inf, np.inf, np.inf, np.nan, 2])

    ts = impute_missing(ts, window=4, direction='b')
    assert (np.any(np.isnan(ts)) == False)
    assert (np.any(np.isinf(ts)) == False)


def test_preprocess():
    ts = np.array([np.nan, np.inf, np.inf, np.nan, np.inf, 2, 3, 2, 3, 1, 2, 3, 4, 2,
                   np.nan, np.inf, 4, 2, 3, 4, 5, 6, 7, 8, 3, 4, 2, 3, 4, 5, 6, 7, 6,
                   5, 4, 3, np.nan, np.nan, np.inf, np.nan, np.inf, np.nan])

    ts = preprocess(ts, window=4)
    assert(np.any(np.isnan(ts)) == False)
    assert(np.any(np.isinf(ts)) == False)