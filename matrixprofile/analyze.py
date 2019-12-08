# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math

from matrixprofile import core

from matrixprofile.algorithms.top_k_discords import top_k_discords
from matrixprofile.algorithms.top_k_motifs import top_k_motifs
from matrixprofile.algorithms import skimp
from matrixprofile.algorithms.mpx import mpx
from matrixprofile.algorithms.scrimp import scrimp_plus_plus
from matrixprofile import visualize


def analyze_pmp(ts, query, sample_pct, threshold, windows=None, n_jobs=-1):
    """
    Computes the Pan-MatrixProfile, top 3 motifs and top 3 discords for the
    provided time series and query. Additionally, plots for the PMP, motifs
    and discords is provided.

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    query : array_like
        The query to analyze.
    sample_pct : float
        A float between 0 and 1 representing how many samples to compute for
        the PMP.
    threshold : float
        A correlation threshold between 0 and 1 that is used to compute the
        upper window. Note that this is used only when the windows is None.
    windows : array_like, default None
        Integers representing the desired windows to use during the
        computation of the PMP.
    n_jobs : int, default -1 (all cpu cores)
        The number of cpu cores to use when computing the PMP.
    
    Returns
    -------
    tuple : (profile, figures)
        A tuple with the first item being the profile and the second being an
        array of matplotlib figures.
    """
    ts = core.to_np_array(ts)

    if isinstance(threshold, type(None)):
        threshold = 0.98

    # when a threshold is passed, we compute the upper window
    profile = None
    if isinstance(windows, type(None)):
        profile = skimp.maximum_subsequence(ts, threshold, include_pmp=True)

        # determine windows to be computed
        # from 8 in steps of 2 until upper w
        start = 8
        end = int(math.floor(len(ts) / 2))
        windows = range(start, profile['upper_window'] + 1)

    # compute the pmp
    profile = skimp.skimp(ts, windows=windows, sample_pct=sample_pct,
        pmp_obj=profile)

    # extract top motifs
    profile = top_k_motifs(profile)

    # extract top discords
    profile = top_k_discords(profile)

    # plot pmp
    figures = visualize(profile)

    return (profile, figures)


def analyze_mp_exact(ts, query, window, n_jobs=-1):
    """
    Computes the exact MatrixProfile, top 3 motifs and top 3 discords for the
    provided time series and query. Additionally, the MatrixProfile, discords
    and motifs are visualized.

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    query : array_like
        The query to analyze.
    window : int
        The window size to compute the MatrixProfile.
    n_jobs : int, default -1 (all cpu cores)
        The number of cpu cores to use when computing the MP.
    
    Returns
    -------
    tuple : (profile, figures)
        A tuple with the first item being the profile and the second being an
        array of matplotlib figures.
    """
    ts = core.to_np_array(ts)

    # compute mp
    profile = mpx(ts, window, query=query, n_jobs=n_jobs)

    # extract top motifs
    profile = top_k_motifs(profile)

    # extract top discords
    profile = top_k_discords(profile)

    # plot mp
    figures = visualize(profile)

    return (profile, figures)


def analyze_mp_approximate(ts, query, window, sample_pct, n_jobs=-1):
    """
    Computes the exact MatrixProfile, top 3 motifs and top 3 discords for the
    provided time series and query. Additionally, the MatrixProfile, discords
    and motifs are visualized.

    Parameters
    ----------
    ts : array_like
        The time series to analyze.
    query : array_like
        The query to analyze.
    window : int
        The window size to compute the MatrixProfile.
    sample_pct : float
        A float between 0 and 1 representing how many samples to compute for
        the MP. When it is 1, it is the same as using the exact algorithm.
    n_jobs : int, default -1 (all cpu cores)
        The number of cpu cores to use when computing the MP.
    
    Returns
    -------
    tuple : (profile, figures)
        A tuple with the first item being the profile and the second being an
        array of matplotlib figures.
    """
    ts = core.to_np_array(ts)

    # compute mp
    profile = scrimp_plus_plus(ts, query, window, sample_pct=sample_pct, n_jobs=n_jobs)

    # extract top motifs
    profile = top_k_motifs(profile)

    # extract top discords
    profile = top_k_discords(profile)

    # plot mp
    figures = visualize(profile)

    return (profile, figures)


def analyze(ts, query=None, windows=None, sample_pct=1.0, threshold=0.98, n_jobs=-1):
    """
    Runs an appropriate workflow based on the parameters passed in. The goal
    of this function is to compute all fundamental algorithms on the provided
    time series data. For now the following is computed:

    1. Matrix Profile - exact or approximate based on sample_pct given that a
       window is provided. By default is the exact algorithm.
    2. Top Motifs - The top 3 motifs are found.
    3. Top Discords - The top 3 discords are found.
    4. Plot MP, Motifs and Discords

    When a window is not provided or more than a single window is provided,
    the PMP is computed:

    1. Compute UPPER window when no window(s) is provided
    2. Compute PMP for all windows
    3. Top Motifs
    4. Top Discords
    5. Plot PMP, motifs and discords.

    Returns
    -------
    tuple : (profile, figures)
        The appropriate PMP or MP profile object and associated figures.
    """
    result = None

    # determine proper number of jobs
    n_jobs = core.valid_n_jobs(n_jobs)

    # determine what algorithm to use based on params
    no_window = isinstance(windows, type(None))
    many_windows = core.is_array_like(windows) and len(windows) > 1
    single_window = isinstance(windows, int) or \
        (core.is_array_like(windows) and len(windows) == 1)
    is_exact = sample_pct >= 1
    is_approx = sample_pct > 0 and sample_pct < 1

    # use PMP with no window provided
    if no_window or many_windows:
        result = analyze_pmp(ts, query, sample_pct, threshold, windows=windows, n_jobs=n_jobs)
    elif single_window and is_exact:
        result = analyze_mp_exact(ts, query, windows, n_jobs=n_jobs)
    elif single_window and is_approx:
        result = analyze_mp_approximate(ts, query, windows, sample_pct, n_jobs=n_jobs)
    else:
        raise RuntimeError('Param combination resulted in an uknown operation')

    return result