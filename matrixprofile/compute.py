# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

# Python native imports
import math
import logging

logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np

# Project imports
from matrixprofile import core
from matrixprofile.preprocess import preprocess
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.scrimp import scrimp_plus_plus
from matrixprofile.algorithms.skimp import skimp
from matrixprofile.algorithms.skimp import maximum_subsequence


def compute(ts, windows=None, query=None, sample_pct=1, threshold=0.98,
            n_jobs=1, preprocessing_args = { 'window': 4, 'impute_method': 'mean',
            'impute_direction': 'forward', 'add_noise': True }):
    """
    Computes the exact or approximate MatrixProfile based on the sample percent
    specified. Currently, MPX and SCRIMP++ is used for the exact and
    approximate algorithms respectively. When multiple windows are passed, the
    Pan-MatrixProfile is computed and returned.

    By default, only passing in a time series (ts), the Pan-MatrixProfile is
    computed based on the maximum upper window algorithm with a correlation
    threshold of 0.98.

    Notes
    -----
    When multiple windows are passed and the Pan-MatrixProfile is computed, the
    query is ignored!

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    windows : int, array_like
        The window(s) to compute the MatrixProfile. Note that it may be an int
        for a single matrix profile computation or an array of ints for
        computing the pan matrix profile.
    query : array_like, optional
        The query to analyze. Note that when computing the PMP the query is
        ignored!
    sample_pct : float, default 1
        A float between 0 and 1 representing how many samples to compute for
        the MP or PMP. When it is 1, the exact algorithm is used.
    threshold : float, default 0.98
        The correlation coefficient used as the threshold. It should be between
        0 and 1. This is used to compute the upper window size when no
        window(s) is given.
    n_jobs : int, default = 1
        Number of cpu cores to use.
    preprocessing_args : dict, default = { 'window': 4, 'impute_method': 'mean', 'impute_direction': 'forward', 'add_noise': True }
        A dictionary object to sets parameters for preprocess function.
        To disable preprocessing procedure, set the preprocessing_args to
        None/False/""/{}.

    Returns
    -------
    dict : profile
        The profile computed.

    """
    result = None
    multiple_windows = core.is_array_like(windows) and len(windows) > 1
    no_windows = isinstance(windows, type(None))
    has_threshold = isinstance(threshold, float)

    if no_windows and not has_threshold:
        raise ValueError('compute requires a threshold or window(s) to be set!')

    # Check to make sure all window sizes are greater than 3, return a ValueError if not.
    if (isinstance(windows, int) and windows < 4) or (multiple_windows and np.any(np.unique(windows) < 4)):
        raise ValueError('Compute requires all window sizes to be greater than 3!')

    if core.is_array_like(windows) and len(windows) == 1:
        windows = windows[0]

    # preprocess the time series
    if preprocessing_args != None and preprocessing_args:

        valid_preprocessing_args_keys = {'window', 'impute_method', 'impute_direction', 'add_noise'}

        if isinstance(preprocessing_args,dict) == False:
            raise ValueError("The parameter 'preprocessing_args' is not dict like!")

        elif set(preprocessing_args.keys()).issubset(valid_preprocessing_args_keys):
            window = 4
            impute_method = 'mean'
            impute_direction = 'forward'
            add_noise = True

            if 'window' in preprocessing_args.keys():
                window = preprocessing_args['window']

            if 'impute_method' in preprocessing_args.keys():
                impute_method = preprocessing_args['impute_method']

            if 'impute_direction' in preprocessing_args.keys():
                impute_direction = preprocessing_args['impute_direction']

            if 'add_noise' in preprocessing_args.keys():
                add_noise = preprocessing_args['add_noise']

            ts = preprocess(ts,
                            window=window,
                            impute_method=impute_method,
                            impute_direction=impute_direction,
                            add_noise=add_noise)

        else:
            raise ValueError('invalid key(s) for preprocessing_args! '
                             'valid key(s) should include '+ str(valid_preprocessing_args_keys))

    # compute the upper window and pmp
    if no_windows and has_threshold:
        profile = maximum_subsequence(ts, threshold, include_pmp=True)

        # determine windows to be computed
        # from 8 in steps of 2 until upper w
        start = 8
        windows = range(start, profile['upper_window'] + 1)

        # compute the pmp
        result = skimp(ts, windows=windows, sample_pct=sample_pct,
                       pmp_obj=profile)

    # compute the pmp
    elif multiple_windows:
        if core.is_array_like(query):
            logger.warn('Computing PMP - query is ignored!')

        result = skimp(ts, windows=windows, sample_pct=1,
                       n_jobs=n_jobs)

    # compute exact mp
    elif sample_pct >= 1:
        result = mpx(ts, windows, query=query, n_jobs=n_jobs)

    # compute approximate mp
    else:
        result = scrimp_plus_plus(ts, windows, query=query, n_jobs=n_jobs,
                                  sample_pct=sample_pct)

    return result