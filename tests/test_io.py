#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import os
import tempfile

import numpy as np

import matrixprofile as mp


def test_disk_to_json_and_from_json_mp():
    ts = np.random.uniform(size=1024)
    w = 32

    profile = mp.algorithms.mpx(ts, w)
    out = os.path.join(tempfile.gettempdir(), 'mp.json')
    mp.io.to_disk(profile, out)

    dprofile = mp.io.from_disk(out)

    keys = set(profile.keys())
    keysb = set(dprofile.keys())

    assert(keys == keysb)

    # check values same
    for k, v in profile.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, dprofile[k])
        elif k == 'data':
            pass
        else:
            assert(v == dprofile[k])

    np.testing.assert_equal(profile['data']['ts'], dprofile['data']['ts'])
    np.testing.assert_equal(profile['data']['query'], dprofile['data']['query'])


def test_disk_to_json_and_from_json_pmp():
    ts = np.random.uniform(size=1024)

    profile = mp.algorithms.skimp(ts)
    out = os.path.join(tempfile.gettempdir(), 'pmp.json')
    mp.io.to_disk(profile, out)

    dprofile = mp.io.from_disk(out)

    keys = set(profile.keys())
    keysb = set(dprofile.keys())

    assert(keys == keysb)

    # check values same
    for k, v in profile.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, dprofile[k])
        elif k == 'data':
            pass
        else:
            assert(v == dprofile[k])

    np.testing.assert_equal(profile['data']['ts'], dprofile['data']['ts'])


def test_disk_to_mpf_and_from_mpf_mp():
    ts = np.random.uniform(size=1024)
    w = 32

    profile = mp.algorithms.mpx(ts, w)
    out = os.path.join(tempfile.gettempdir(), 'mp.mpf')
    mp.io.to_disk(profile, out, format='mpf')

    dprofile = mp.io.from_disk(out)

    keys = set(profile.keys())
    keysb = set(dprofile.keys())

    assert(keys == keysb)

    # check values same
    for k, v in profile.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, dprofile[k])
        elif k == 'data':
            pass
        else:
            assert(v == dprofile[k])

    np.testing.assert_equal(profile['data']['ts'], dprofile['data']['ts'])
    np.testing.assert_equal(profile['data']['query'], dprofile['data']['query'])


def test_disk_to_mpf_and_from_mpf_pmp():
    ts = np.random.uniform(size=1024)

    profile = mp.algorithms.skimp(ts)
    out = os.path.join(tempfile.gettempdir(), 'pmp.mpf')
    mp.io.to_disk(profile, out, format='mpf')

    dprofile = mp.io.from_disk(out)

    keys = set(profile.keys())
    keysb = set(dprofile.keys())

    assert(keys == keysb)

    # check values same
    for k, v in profile.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(v, dprofile[k])
        elif k == 'data':
            pass
        else:
            assert(v == dprofile[k])

    np.testing.assert_equal(profile['data']['ts'], dprofile['data']['ts'])