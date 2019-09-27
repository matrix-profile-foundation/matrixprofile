# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import math

from matplotlib import pyplot as plt
import numpy as np

from matrixprofile import core
from matrixprofile.algorithms.mpx import mpx


def split(lower_bound, upper_bound, middle):
    if lower_bound == middle:
        L = None
        R = [middle + 1, upper_bound]
    elif upper_bound == middle:
        L = [lower_bound, middle - 1]
        R = None
    else:
        L = [lower_bound, middle - 1]
        R = [middle + 1, upper_bound]
    
    return (L, R)


def binary_split(n):
    index = []
    intervals = []
    
    # always begin by exploring the first integer
    index.append(0)
    
    # after exploring first integer, split interval 2:n
    intervals.append([1, n - 1])
    
    while len(intervals) > 0:
        interval = intervals.pop(0)
        lower_bound = interval[0]
        upper_bound = interval[1]
        middle = int(math.floor((lower_bound + upper_bound) / 2))
        index.append(middle)
        
        if lower_bound == upper_bound:
            continue
        else:
            L, R = split(lower_bound, upper_bound, middle)
            
            if L is not None:
                intervals.append(L)
            
            if R is not None:
                intervals.append(R)
    
    return index


def plot_pmp(data, cmap=None):
    plt.figure(figsize = (15,15))
    depth = 256
    test = np.ceil(data * depth) / depth
    test[test > 1] = 1
    plt.imshow(test, cmap=cmap)
    plt.gca().invert_yaxis()
    plt.title('PMP')
    plt.xlabel('Profile Index')
    plt.ylabel('Window Size')
    plt.show()


def skimp(ts, windows=None, show_progress=False):
    ts = core.to_np_array(ts)
    n = len(ts)
    
    if windows == None:
        start = 4
        end = int(math.floor(len(ts) / 2))
        windows = range(start, end + 1)
    
    r = len(windows)

    split_index = binary_split(r)
    pmp = np.full((r, n), np.inf)
    idx = np.full(r, -1)
    
    pct_shown = {}
    
    for i in range(len(split_index)):
        window_size = windows[split_index[i]]
        mp, pi = mpx(ts, window_size)
        pmp[split_index[i], 0:len(mp)] = mp
        
        j = split_index[i]
        while j < len(split_index) and idx[j] != j:
            idx[j] = split_index[i]
            j = j + 1
        
        if show_progress:
            pct_complete = round((i / (len(split_index) - 1)) * 100, 2)
            int_pct = math.floor(pct_complete)
            if int_pct % 5 == 0 and int_pct not in pct_shown:
                print('{}% complete'.format(int_pct))
                pct_shown[int_pct] = 1
    
    return (pmp, idx)