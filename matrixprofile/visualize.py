# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

# handle Python 2/3 Iterable import
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import os

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from matrixprofile import core


def __combine(a, b):
    """
    Helper function to combine lists or a list and an object.
    """
    output = []
    if isinstance(a, list) and isinstance(b, list):
        output = a + b
    elif isinstance(a, list):
        output = a
        output.append(b)
    elif isinstance(b, list):
        output = b
        output.append(a)

    return output


def is_visualizable(obj):
    """
    Helper function to determine if the passed in object can be visualized or
    not based on the data structure.

    Parameters
    ----------
    obj : Object
        The object to test.

    Returns
    -------
    list : figures
        A list of matplotlib figures.

    """
    return core.is_mp_obj(obj) or core.is_pmp_obj(obj) or core.is_stats_obj(obj)


def visualize(profile):
    """
    Automatically creates plots for the provided data structure. In some cases
    many plots are created. For example, when a MatrixProfile is passed with
    corresponding motifs and discords, the matrix profile, discords and motifs
    will be plotted.

    Parameters
    ----------
    profile : dict_like
        A MatrixProfile, Pan-MatrixProfile or Statistics data structure.

    Returns
    -------
    list : figures
        A list of matplotlib figures.

    """
    figures = []

    if not is_visualizable(profile):
        raise ValueError('MatrixProfile, Pan-MatrixProfile or Statistics data structure expected!')

    # plot MP
    if core.is_mp_obj(profile):
        figures = __combine(figures, plot_mp(profile))

        if 'cmp' in profile and len(profile['cmp']) > 0:
            figures = __combine(figures, plot_cmp_mp(profile))

        if 'av' in profile and len(profile['av']) > 0:
            figures = __combine(figures, plot_av_mp(profile))

        if 'motifs' in profile and len(profile['motifs']) > 0:
            figures = __combine(figures, plot_motifs_mp(profile))

        if 'discords' in profile and len(profile['discords']) > 0:
            figures = __combine(figures, plot_discords_mp(profile))

    # plot PMP
    if core.is_pmp_obj(profile):
        figures = __combine(figures, plot_pmp(profile))

        if 'motifs' in profile and len(profile['motifs']) > 0:
            figures = __combine(figures, plot_motifs_pmp(profile))

        if 'discords' in profile and len(profile['discords']) > 0:
            figures = __combine(figures, plot_discords_pmp(profile))

    # plot stats
    if core.is_stats_obj(profile):
        figures = __combine(figures, plot_stats(profile))


    return figures


def plot_stats(profile):
    """
    Plots the given Statistics data structure provided.

    Parameters
    ----------
    profile : dict
        The dict structure from a Statistics algorithm.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    ts = profile.get('ts')
    ax[0].plot(ts, label='Time Series', c='black')

    for k, v in profile.items():
        if k.startswith('moving'):
            ax[1].plot(v, label=k)

    fig.legend(loc="upper right", bbox_to_anchor=(1.11, 0.97))
    fig.tight_layout()

    return fig


def plot_pmp(profile):
    """
    Plots the given Pan-MatrixProfile data structure provided.

    Parameters
    ----------
    profile : dict
        The dict structure from a PMP algorithm.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    pmp = profile.get('pmp', None)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    depth = 256
    test = np.ceil(pmp * depth) / depth
    test[test > 1] = 1

    ax.imshow(test, interpolation=None, aspect='auto')
    ax.invert_yaxis()
    ax.set_title('Pan-MatrixProfile')
    ax.set_xlabel('Profile Index')
    ax.set_ylabel('Window Size')
    ax.set_aspect(1, 'box')

    fig.tight_layout()

    return fig


def plot_mp(profile):
    """
    Plots the matrix profile given the appropriate data structure.

    Parameters
    ----------
    profile : dict_like
        The matrix profile to plot.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    plot_count = 0
    data = profile.get('data', None)
    ts = None
    query = None
    if data:
        ts = data.get('ts', None)
        query = data.get('query', None)

    mp = profile.get('mp', None)
    lmp = profile.get('lmp', None)
    rmp = profile.get('rmp', None)

    for val in [ts, query, mp, lmp, rmp]:
        if core.is_array_like(val):
            plot_count += 1

    if plot_count < 1:
        raise ValueError("Object passed has nothing to plot!")

    w = profile.get('w', None)
    if not isinstance(w, int):
        raise ValueError("Expecting window size!")

    current = 0

    fig, axes = plt.subplots(plot_count, 1, sharex=True, figsize=(15, 7))

    if not isinstance(axes, Iterable):
        axes = [axes,]

    # plot the original ts
    if core.is_array_like(ts):
        axes[current].plot(np.arange(len(ts)), ts)
        axes[current].set_ylabel('Data')
        current += 1

    # plot the original query
    if core.is_array_like(query):
        axes[current].plot(np.arange(len(query)), query)
        axes[current].set_ylabel('Query')
        current += 1

    # plot matrix profile
    if core.is_array_like(mp):
        mp_adj = np.append(mp, np.zeros(w - 1) + np.nan)
        axes[current].plot(np.arange(len(mp_adj)), mp_adj)
        axes[current].set_ylabel('Matrix Profile')
        axes[current].set_title('Window Size {}'.format(w))
        current += 1

    # plot left matrix profile
    if core.is_array_like(lmp):
        mp_adj = np.append(lmp, np.zeros(w - 1) + np.nan)
        axes[current].plot(np.arange(len(mp_adj)), mp_adj)
        axes[current].set_ylabel('Left Matrix Profile')
        axes[current].set_title('Window Size {}'.format(w))
        current += 1

    # plot left matrix profile
    if core.is_array_like(rmp):
        mp_adj = np.append(rmp, np.zeros(w - 1) + np.nan)
        axes[current].plot(np.arange(len(mp_adj)), mp_adj)
        axes[current].set_ylabel('Right Matrix Profile')
        axes[current].set_title('Window Size {}'.format(w))
        current += 1

    fig.tight_layout()

    return fig


def plot_cmp_mp(profile):
    """
    Plot corrected matrix profile for a MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The matrix profile object to plot.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    cmp = profile['cmp']
    w = profile['w']

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    cmp_adj = np.append(cmp, np.zeros(w - 1) + np.nan)
    ax.plot(np.arange(len(cmp_adj)), cmp_adj)
    ax.set_ylabel('Corrected Matrix Profile')
    ax.set_title('Window Size {}'.format(w))

    fig.tight_layout()

    return fig


def plot_av_mp(profile):
    """
    Plot the annotation vector for a MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The matrix profile object to plot.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    av = profile['av']
    w = profile['w']

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    av_adj = np.append(av, np.zeros(w - 1) + np.nan)
    ax.plot(np.arange(len(av_adj)), av_adj)
    ax.set_ylabel('Annotation Vector')
    ax.set_title('Window Size {}'.format(w))

    fig.tight_layout()

    return fig


def plot_discords_mp(profile):
    """
    Plot discords for a MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The matrix profile object to plot.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    mp = profile['mp']
    w = profile['w']
    discords = profile['discords']
    data = profile.get('data', None)
    if data:
        ts = data.get('ts', None)

    mp_adjusted = np.append(mp, np.full(w + 1, np.nan))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 7), gridspec_kw={'height_ratios': [25, 5, 25]})

    pos = axes[1].imshow([mp_adjusted,], aspect='auto', cmap='coolwarm')
    axes[1].set_yticks([])
    axes[0].plot(np.arange(len(ts)), ts)
    axes[0].set_ylabel('Data')

    axes[2].plot(np.arange(len(mp_adjusted)), mp_adjusted)
    axes[2].set_ylabel('Matrix Profile')
    axes[2].set_title('Window Size {}'.format(w))

    for idx in discords:
        axes[2].plot(idx, mp_adjusted[idx], c='r', marker='*', lw=0, markersize=10)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1, 0.46, 0.01, 0.1])
    fig.colorbar(pos, orientation='vertical', cax=cbar_ax, use_gridspec=True)

    lines = [
        Line2D([0], [0], color='red', marker='*', lw=0),
        Line2D([0], [0], color='blue'),
    ]
    fig.legend(lines, ['Discord', 'MP'], bbox_to_anchor=(1.07, 0.44))

    fig.tight_layout()

    return fig


def plot_discords_pmp(profile):
    """
    Plot discords for the given Pan-MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The pmp object to plot.

    Returns
    -------
    matplotlib.Figure : figure
        The matplotlib figure object.

    """
    discord_figures = []

    for discord in profile['discords']:
        mp_idx = discord[0]
        idx = discord[1]
        w = profile['windows'][mp_idx]

        data = profile.get('data', None)
        if data:
            ts = data.get('ts', None)

        mp_adjusted = profile['pmp'][mp_idx]
        # mp_adjusted = np.append(mp, np.full(w + 1, np.nan))

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 7), gridspec_kw={'height_ratios': [25, 5, 25]})

        pos = axes[1].imshow([mp_adjusted,], aspect='auto', cmap='coolwarm')
        axes[1].set_yticks([])
        axes[0].plot(np.arange(len(ts)), ts)
        axes[0].set_ylabel('Data')

        axes[2].plot(np.arange(len(mp_adjusted)), mp_adjusted)
        axes[2].set_ylabel('Matrix Profile')

        # for idx in discords:
        axes[2].plot(idx, mp_adjusted[idx], c='r', marker='*', lw=0, markersize=10)
        axes[2].set_title('Window Size = {}'.format(w))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.46, 0.01, 0.1])
        fig.colorbar(pos, orientation='vertical', cax=cbar_ax, use_gridspec=True)

        lines = [
            Line2D([0], [0], color='red', marker='*', lw=0),
            Line2D([0], [0], color='blue'),
        ]
        fig.legend(lines, ['Discord', 'MP'], bbox_to_anchor=(1.07, 0.44))

        fig.tight_layout()

        discord_figures.append(fig)

    return discord_figures


def plot_motifs_mp(profile):
    """
    Plot motifs given a MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The matrix profile object to plot.

    Returns
    -------
    list : figures
        A list of matplotlib figure objects.

    """
    figures = []

    w = profile['w']
    motifs = profile['motifs']
    data = profile.get('data', None)
    if data:
        ts = data.get('ts', None)

    fig, axes = plt.subplots(len(motifs), 2, figsize=(15, 7), sharey='row', sharex='col')
    if len(motifs) == 1:
        axes = [axes,]

    pair_num = 1
    for ax_row, motif in zip(axes, motifs):
        first = True
        for ax, idx in zip(ax_row, motif['motifs']):
            subquery = ts[idx:idx + w]
            indices = np.arange(len(subquery))
            ax.plot(indices, subquery)
            ax.set_title('Index Start {}'.format(idx))
            if first:
                ax.set_ylabel('Motif {}'.format(pair_num))
                first = False

        pair_num += 1

    fig.tight_layout()
    figures.append(fig)

    fig, axes = plt.subplots(len(motifs), 1, figsize=(15, 7), sharey='row', sharex='col')
    if len(motifs) == 1:
        axes = [axes,]

    pair_num = 1
    for ax, motif in zip(axes, motifs):
        ax.plot(np.arange(len(ts)), ts)
        for idx in motif['motifs']:
            subquery = ts[idx:idx + w]
            indices = np.arange(idx, idx + w)
            ax.plot(indices, subquery, c='r')
            ax.set_ylabel('Motif {}'.format(pair_num))

        for idx in motif['neighbors']:
            subquery = ts[idx:idx + w]
            indices = np.arange(idx, idx + w)
            ax.plot(indices, subquery, c='black')

        pair_num += 1

    lines = [
        Line2D([0], [0], color='blue'),
        Line2D([0], [0], color='red'),
        Line2D([0], [0], color='black')
    ]
    fig.legend(lines, ['Data', 'Motif', 'Neighbor'], bbox_to_anchor=(1.08, 0.975))
    fig.tight_layout()

    figures.append(fig)

    return figures


def plot_motifs_pmp(profile):
    """
    Plot motifs given a Pan-MatrixProfile data structure.

    Parameters
    ----------
    profile : dict_like
        The pan matrix profile object to plot.

    Returns
    -------
    list : figures
        A list of matplotlib figure objects.

    """
    motif_figures = []
    motifs = profile.get('motifs')
    data = profile.get('data', None)
    windows = profile.get('windows', None)
    if data:
        ts = data.get('ts', None)

    fig, axes = plt.subplots(len(motifs), 2, figsize=(15, 7), sharey='row', sharex='col')
    if len(motifs) == 1:
        axes = [axes,]

    pair_num = 1
    for ax_row, motif in zip(axes, motifs):
        first = True
        for ax, motif_loc in zip(ax_row, motif['motifs']):
            w = windows[motif_loc[0]]
            idx = motif_loc[1]
            subquery = ts[idx:idx + w]
            indices = np.arange(len(subquery))
            ax.plot(indices, subquery)
            ax.set_title('Index Start {}, Window Size {}'.format(idx, w))
            if first:
                ax.set_ylabel('Motif {}'.format(pair_num))
                first = False

        pair_num += 1

    fig.tight_layout()
    motif_figures.append(fig)

    fig, axes = plt.subplots(len(motifs), 1, figsize=(15, 7), sharey='row', sharex='col')
    if len(motifs) == 1:
        axes = [axes,]

    pair_num = 1
    for ax, motif in zip(axes, motifs):
        ax.plot(np.arange(len(ts)), ts)
        for motif_loc in motif['motifs']:
            w = windows[motif_loc[0]]
            idx = motif_loc[1]
            subquery = ts[idx:idx + w]
            indices = np.arange(idx, idx + w)
            ax.plot(indices, subquery, c='r')
            ax.set_title('Window Size {}'.format(w))
            ax.set_ylabel('Motif {}'.format(pair_num))

        for neigh_loc in motif['neighbors']:
            w = windows[neigh_loc[0]]
            idx = neigh_loc[1]
            subquery = ts[idx:idx + w]
            indices = np.arange(idx, idx + w)
            ax.plot(indices, subquery, c='black')

        pair_num += 1

    lines = [
        Line2D([0], [0], color='blue'),
        Line2D([0], [0], color='red'),
        Line2D([0], [0], color='black')
    ]
    fig.legend(lines, ['Data', 'Motif', 'Neighbor'], bbox_to_anchor=(1.08, 0.975))
    fig.tight_layout()

    motif_figures.append(fig)

    return motif_figures


def plot_snippets(snippets, ts):
    """
    Plot snippets for the given Snippets data structure.
    
    Parameters
    ----------
    snippets : list
        A list of snippets as dictionary objects to plot.
    ts : array_like
        The time series.
        
    Returns
    -------
    list : figures
        A list of matplotlib figures.
        
    """
    figures = []
    
    for i in range(len(snippets)):
        snippet_id = str(i+1)
        snippet_start = snippets[i]['index']
        snippet_end = snippets[i]['index']+len(snippets[i]['snippet'])
        snippet_data = snippets[i]['snippet']
        
        # Create a plot for current snippets
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15,5))
        ax.plot(ts)
        ax.set_title('Snippet-'+snippet_id, size=12)
        ax.set_ylabel('Data')
        flag = 1
    
        # Get intervals for the given neighboring snippet indices
        neighbors = snippets[i]['neighbors']
        intervals = []
        for i in range(len(neighbors)):
            if i == 0:
                intervals.append(neighbors[i])
            if i == len(neighbors)-1:
                intervals.append(neighbors[i])
                break
            if (neighbors[i+1] - neighbors[i]) != 1:
                intervals.append(neighbors[i])
                intervals.append(neighbors[i+1])
        step = 2
        intervals = [intervals[i:i+step] for i in range(0, len(intervals), step)]
    
        # Plot the neighboring snippets
        for interval in intervals:
            start = interval[0]
            end = interval[1]
            if flag:
                ax.plot(np.arange(start,end+1),ts[start:end+1], c = "orange"
                        ,label = "Subsequences Represented by Snippet-"+ snippet_id)
                flag = 0
            else:
                ax.plot(np.arange(start,end+1),ts[start:end+1],"orange")
    
        # Plot the snippet
        ax.plot(np.arange(snippet_start,snippet_end), snippet_data, c = "red"
                , label = 'Snippet-'+ snippet_id)
        
        plt.legend(loc="upper right")       
        fig.tight_layout()       
        figures.append(fig)
    
    return figures

