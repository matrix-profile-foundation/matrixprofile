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

from matrixprofile.algorithms.stomp import stomp
from matrixprofile.algorithms.skimp import skimp
from matrixprofile.visualize import visualize
from matrixprofile.visualize import plot_snippets
from matrixprofile.algorithms.snippets import snippets

def test_catch_all_visualize_invalid_structure():
    data = {}
    with pytest.raises(Exception) as e:
        visualize(data)
        assert('MatrixProfile, Pan-MatrixProfile or Statistics data structure expected!' == str(e.value))


def test_catch_all_visualize_mp_only():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)

    # expect only the matrix profile plot
    figures = visualize(profile)
    assert(len(figures) == 1)


def test_catch_all_visualize_mp_cmp():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['cmp'] = np.arange(len(ts) - w + 1)

    figures = visualize(profile)
    assert(len(figures) == 2)


def test_catch_all_visualize_mp_av():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['av'] = np.arange(len(ts) - w + 1)

    figures = visualize(profile)
    assert(len(figures) == 2)


def test_catch_all_visualize_mp_cmp_av():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['cmp'] = np.arange(len(ts) - w + 1)
    profile['av'] = np.arange(len(ts) - w + 1)

    figures = visualize(profile)
    assert(len(figures) == 3)


def test_catch_all_visualize_mp_discords():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['discords'] = [0, 1]

    figures = visualize(profile)
    assert(len(figures) == 2)


def test_catch_all_visualize_mp_motifs():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['motifs'] = [{'motifs': [1, 1], 'neighbors': []}]

    figures = visualize(profile)
    assert(len(figures) == 3)


def test_catch_all_visualize_mp_motifs_discords():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = 4

    profile = stomp(ts, w, n_jobs=1)
    profile['discords'] = [0, 1]
    profile['motifs'] = [{'motifs': [1, 1], 'neighbors': []}]

    figures = visualize(profile)
    assert(len(figures) == 4)


def test_catch_all_visualize_pmp_only():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = [4, 5, 6]

    profile = skimp(ts, w, n_jobs=1)

    # expect only the matrix profile plot
    figures = visualize(profile)
    assert(len(figures) == 1)


def test_catch_all_visualize_pmp_discords():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = [4, 5, 6]

    profile = skimp(ts, w, n_jobs=1)
    profile['discords'] = [(0, 1), (0, 2)]

    figures = visualize(profile)
    assert(len(figures) == 3)


def test_catch_all_visualize_pmp_motifs():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = [4, 5, 6]

    profile = skimp(ts, w, n_jobs=1)
    profile['motifs'] = [{'motifs': [(1, 1)], 'neighbors': []}]

    figures = visualize(profile)
    assert(len(figures) == 3)

def test_catch_all_visualize_pmp_motifs_discords():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    w = [4, 5, 6]

    profile = skimp(ts, w, n_jobs=1)
    profile['discords'] = [(0, 1), (0, 2)]
    profile['motifs'] = [{'motifs': [(1, 1)], 'neighbors': []}]

    figures = visualize(profile)
    assert(len(figures) == 5)


def test_catch_all_stats():
    profile = {
        'class': 'Statistics',
        'ts': np.array([]),
        'window_size': 100
    }

    figures = visualize(profile)
    assert(len(figures) == 1)


def test_catch_all_visualize_snippets():
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    snippet_size = 4
    snippet_num = 1

    snippet_list = snippets(ts, snippet_size, snippet_num)

    figures = plot_snippets(snippet_list, ts)
    assert (len(figures) == snippet_num)