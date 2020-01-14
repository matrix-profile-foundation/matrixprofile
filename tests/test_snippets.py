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
    assert(result[0]['index'] == 192)
    assert(result[1]['index'] == 704)

    snippet_size = 128
    result = snippets(ts, snippet_size, window_size=w)
    assert(result[0]['index'] == 384)
    assert(result[1]['index'] == 640)