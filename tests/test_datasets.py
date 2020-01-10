#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import pytest

import numpy as np

from matrixprofile.datasets.datasets import load
from matrixprofile.datasets.datasets import fetch_available


def test_load_valid():
    dataset = load('motifs-discords-small')
    assert(isinstance(dataset['data'], np.ndarray) == True)
    assert('description' in dataset)
    assert('name' in dataset)
    assert('category' in dataset)


def test_load_not_found():  
    with pytest.raises(ValueError) as excinfo:
        data = load('alksdfasdf')
        assert('Could not find dataset alksdfasdf' in str(excinfo.value))


def test_fetch_available_all():
    datasets = fetch_available()
    assert(isinstance(datasets, list) == True)
    assert(len(datasets) > 0)


def test_fetch_available_category_valid():
    datasets = fetch_available(category='real')
    assert(isinstance(datasets, list) == True)
    assert(len(datasets) > 0)


def test_fetch_available_category_invalid():
    with pytest.raises(ValueError) as excinfo:
        fetch_available('alksdsfldfsd')
        assert('category alksdsfldfsd is not a valid option.' in str(excinfo.value))