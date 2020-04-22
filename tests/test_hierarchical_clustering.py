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

from matrixprofile.algorithms.hierarchical_clustering import (
    compute_dist,
    pairwise_dist,
    hierarchical_clusters
)

def test_hierarchical_clusters_valid_simple():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    ts2 = np.random.uniform(size=2**10)
    ts3 = np.random.uniform(size=2**10)
    X = [
        ts,
        ts,
        ts2,
        ts2,
        ts3
    ]
    w = 2**6
    t = 2

    clusters = hierarchical_clusters(X, w, t)

    # evaluate keys
    expected_keys = set([
        'pairwise_distances',
        'linkage_matrix',
        'inconsistency_statistics',
        'assignments',
        'cophenet',
        'cophenet_distances',
        'class'
    ])
    actual_keys = set(clusters.keys())
    assert(expected_keys == actual_keys)
    assert(clusters['class'] == 'hclusters')

    # evaluate cluster assignments
    expected_assignments = np.array([1, 1, 2, 2, 3])
    np.testing.assert_equal(clusters['assignments'], expected_assignments)

    # evaluate cophenet score
    expected_cophenet = 0.9999870997174531
    np.testing.assert_almost_equal(clusters['cophenet'], expected_cophenet)

    # evaluate pairwise distances
    expected_distances = np.array([0, 8.2299501, 8.2299501, 8.29915377, 
    8.2299501, 8.2299501, 8.29915377, 0, 8.2558308, 8.2558308])
    np.testing.assert_almost_equal(
        clusters['pairwise_distances'], expected_distances)


def test_hierarchical_clusters_valid_simple_parallel():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    ts2 = np.random.uniform(size=2**10)
    ts3 = np.random.uniform(size=2**10)
    X = [
        ts,
        ts,
        ts2,
        ts2,
        ts3
    ]
    w = 2**6
    t = 2

    clusters = hierarchical_clusters(X, w, t, n_jobs=2)

    # evaluate keys
    expected_keys = set([
        'pairwise_distances',
        'linkage_matrix',
        'inconsistency_statistics',
        'assignments',
        'cophenet',
        'cophenet_distances',
        'class'
    ])
    actual_keys = set(clusters.keys())
    assert(expected_keys == actual_keys)
    assert(clusters['class'] == 'hclusters')

    # evaluate cluster assignments
    expected_assignments = np.array([1, 1, 2, 2, 3])
    np.testing.assert_equal(clusters['assignments'], expected_assignments)

    # evaluate cophenet score
    expected_cophenet = 0.9999870997174531
    np.testing.assert_almost_equal(clusters['cophenet'], expected_cophenet)

    # evaluate pairwise distances
    expected_distances = np.array([0, 8.2299501, 8.2299501, 8.29915377, 
    8.2299501, 8.2299501, 8.29915377, 0, 8.2558308, 8.2558308])
    np.testing.assert_almost_equal(
        clusters['pairwise_distances'], expected_distances)


def test_hierarchical_clusters_invalid_params():
    np.random.seed(9999)
    ts = np.random.uniform(size=2**10)
    ts2 = np.random.uniform(size=2**10)
    ts3 = np.random.uniform(size=2**10)
    X = [
        ts,
        ts,
        ts2,
        ts2,
        ts3
    ]
    w = 2**6
    t = 2

    # invalid X
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters('', w, t)
        assert('X must be array_like!' == str(excinfo.value))

    # invalid t
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, '')
        assert('t must be a scalar (int or float)' == str(excinfo.value))
    
    # invalid threshold 0
    error = 'threshold must be a float greater than 0 and less than 1'
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, threshold=0)
        assert(error == str(excinfo.value))
    
    # invalid threshold 1
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, threshold=1)
        assert(error == str(excinfo.value))

    # invalid threshold not numeric
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, threshold='')
        assert(error == str(excinfo.value))

    # invalid depth < 1
    error = 'depth must be an integer greater than 0'
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, depth=0)
        assert(error == str(excinfo.value))

    # invalid depth not int
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, depth='')
        assert(error == str(excinfo.value))

    # invalid method
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, method='')
        assert('method may be only one of' in str(excinfo.value))

    # invalid criterion
    with pytest.raises(ValueError) as excinfo:
        clusters = hierarchical_clusters(X, w, t, criterion='')
        assert('criterion may be only one of' in str(excinfo.value))


def test_pairwise_dist_valid_simple():
    X = [
        np.arange(100),
        np.arange(100),
        np.ones(100),
        np.zeros(100)
    ]
    w = 8
    dists = pairwise_dist(X, w)
    expected = np.array([ 0, np.inf, np.inf, np.inf, np.inf, np.inf])
    np.testing.assert_equal(dists, expected)

    # test with MxN np.ndarray
    X = np.array(X)
    dists = pairwise_dist(X, w)
    expected = np.array([ 0, np.inf, np.inf, np.inf, np.inf, np.inf])
    np.testing.assert_equal(dists, expected)


def test_pairwise_dist_invalid_params():
    X = [
        np.arange(10),
        np.arange(20)
    ]
    w = 4
    threshold = 0.05
    n_jobs = 1
    with pytest.raises(ValueError) as excinfo:
        pairwise_dist('', w, threshold=threshold, n_jobs=n_jobs)
        assert('X must be array_like!' == str(excinfo.value))

    error = 'threshold must be a float greater than 0 and less'\
                ' than 1'
    with pytest.raises(ValueError) as excinfo:
        pairwise_dist(X, w, threshold=1, n_jobs=n_jobs)
        assert(error == str(excinfo.value))


def test_compute_dist_valid():
    ts = np.arange(100)
    w = 8
    k = 0
    threshold = 0.05
    args = (k, ts, ts, w, threshold)
    result = compute_dist(args)

    assert(result[0] == k)
    assert(result[0] == 0)