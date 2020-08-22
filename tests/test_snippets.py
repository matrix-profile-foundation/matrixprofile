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

from matrixprofile.algorithms.snippets import snippets

import matrixprofile

MODULE_PATH = matrixprofile.__path__[0]

def test_snippets():
    ts = np.loadtxt(os.path.join(MODULE_PATH, '..', 'tests', 'sampledata.txt'))
    w = 32
    snippet_size = 64

    result = snippets(ts, snippet_size, window_size=w)
    assert(result[0]['index'] == 384)
    assert(result[1]['index'] == 704)
    assert(sum(result[0]['neighbors']) == 191408)
    assert(sum(result[1]['neighbors']) == 190967)

    # test inferred window size of snippet size / 2
    result = snippets(ts, snippet_size)
    assert(result[0]['index'] == 384)
    assert(result[1]['index'] == 704)
    assert(sum(result[0]['neighbors']) == 191408)
    assert(sum(result[1]['neighbors']) == 190967)

    snippet_size = 128
    result = snippets(ts, snippet_size, window_size=w)
    assert(result[0]['index'] == 384)
    assert(result[1]['index'] == 640)
    assert(sum(result[0]['neighbors']) == 227661)
    assert(sum(result[1]['neighbors']) == 154714)

    snippet_size = 8
    result = snippets(ts, snippet_size, window_size=snippet_size / 2)
    assert(result[0]['index'] == 72)
    assert(result[1]['index'] == 784)
    assert(sum(result[0]['neighbors']) == 149499)
    assert(sum(result[1]['neighbors']) == 232876)

def test_invalid_snippet_size():
    ts = np.arange(100)
    ss = 2

    error = 'snippet_size must be an integer >= 4'
    with pytest.raises(ValueError) as excinfo:
        snippets(ts, ss)
        assert(error == str(excinfo.value))

    with pytest.raises(ValueError) as excinfo:
        snippets(ts, '232')
        assert(error == str(excinfo.value))


def test_invalid_snippet_size_and_ts():
    ts = np.arange(100)
    ss = 75

    error = 'Time series is too short relative to snippet length'
    with pytest.raises(ValueError) as excinfo:
        snippets(ts, ss)
        assert(error == str(excinfo.value))


def test_window_size_greater_snippet_size():
    ts = np.arange(100)
    ss = 25
    w = 30

    error = 'window_size must be smaller than snippet_size'
    with pytest.raises(ValueError) as excinfo:
        snippets(ts, ss, window_size=w)
        assert(error == str(excinfo.value))
