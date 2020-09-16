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
    assert(valid_kwargs == None)


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

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {'window': 'str'}
        valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
        assert "The value for preprocessing_kwargs['window'] is not an integer!" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {'impute_method': False}
        valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
        assert "invalid imputation method! valid include options: mean, median, min, max" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {'impute_direction': 5}
        valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
        assert "invalid imputation direction! valid include options: " \
               "forward, fwd, f, backward, bwd, b" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        preprocessing_kwargs = {'add_noise': 'str'}
        valid_kwargs = validate_preprocess_kwargs(preprocessing_kwargs)
        assert "The value for preprocessing_kwargs['add_noise'] is not a boolean value!" \
            in str(excinfo.value)


def test_is_subsequence_constant():
    with pytest.raises(ValueError) as excinfo:
        ts = 1
        is_subsequence_constant(ts)
        assert "subsequence is not array like!" \
            in str(excinfo.value)

    ts = np.array([1, 1, 1, 1, 1, 1])
    assert(is_subsequence_constant(ts) == True)

    ts = np.array([1, 2, 1, 1, 1, 1])
    assert(is_subsequence_constant(ts) == False)


def test_add_noise_to_series():
    with pytest.raises(ValueError) as excinfo:
        ts = 1
        temp = add_noise_to_series(ts)
        assert "series is not array like!" \
            in str(excinfo.value)

    ts = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    temp = add_noise_to_series(ts)
    assert(np.all((temp - ts) >= 0) and np.all((temp - ts) < 0.0000009))


def test_impute_missing():
    with pytest.raises(ValueError) as excinfo:
        ts = 1
        ts = impute_missing(ts, window=4)
        assert "ts is not array like!" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ts = np.array([1, 2, 3])
        ts = impute_missing(ts, window=4, method=False)
        assert "invalid imputation method! valid include options: mean, median, min, max" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ts = np.array([1, 2, 3])
        ts = impute_missing(ts, window=4, direction='a')
        assert "invalid imputation direction! valid include options: " \
               "forward, fwd, f, backward, bwd, b" \
            in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ts = np.array([1, 2, 3])
        ts = impute_missing(ts, window='str')
        assert "window is not an integer!" \
            in str(excinfo.value)

    ts = np.array([np.nan, np.nan, np.inf, np.nan, np.inf, np.inf, 4, 5, np.nan,
                   np.inf, np.nan, np.inf, np.inf, np.inf, np.inf, np.nan, 2])

    ts = impute_missing(ts, window=4, direction='b')
    assert(np.any(np.isnan(ts)) == False)
    assert(np.any(np.isinf(ts)) == False)


def test_preprocess():
    with pytest.raises(ValueError) as excinfo:
        ts = 1
        ts = preprocess(ts, window=4)
        assert "ts is not array like!" \
            in str(excinfo.value)

    ts = np.array([np.nan, np.inf, np.inf, np.nan, np.inf, 2, 3, 2, 3, 1, 2, 3, 4, 2,
                   np.nan, np.inf, 4, 2, 3, 4, 5, 6, 7, 8, 3, 4, 2, 3, 4, 5, 6, 7, 6,
                   5, 4, 3, np.nan, np.nan, np.inf, np.nan, np.inf, np.nan])

    ts = preprocess(ts, window=4)
    assert(np.any(np.isnan(ts)) == False)
    assert(np.any(np.isinf(ts)) == False)