# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from collections.abc import Iterable

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


def visualize(obj):
	"""
	Plots an object generated from one of the algorithms. In some cases
	multiple plots will be generated.

	Parameters
	----------
	obj : dict_like
		The object to plot.

	Returns
	-------
	A list of matplotlib figures.
	"""
	figures = []

	mp_class = obj.get('class', None)

	if mp_class == 'MatrixProfile':
		figures = __combine(figures, plot_mp(obj))
	
	if 'motifs' in obj:
		figures = __combine(figures, plot_motifs(obj))
	
	if 'discords' in obj:
		figures = __combine(figures, plot_discords(obj))

	return figures


def plot_mp(obj):
	"""
	Plots a matrix profile object.

	Parameters
	----------
	obj : dict_like
		The matrix profile object to plot.

	Returns
	-------
	The matplotlib figure object.
	"""
	plot_count = 0
	data = obj.get('data', None)
	ts = None
	query = None
	if data:
		ts = data.get('ts', None)
		query = data.get('query', None)

	mp = obj.get('mp', None)
	lmp = obj.get('lmp', None)
	rmp = obj.get('rmp', None)

	for val in [ts, query, mp, lmp, rmp]:
		if core.is_array_like(val):
			plot_count += 1

	if plot_count < 1:
		raise ValueError("Object passed has nothing to plot!")

	w = obj.get('w', None)
	if not isinstance(w, int):
		raise ValueError("Expecting window size!")

	current = 0

	fig, axes = plt.subplots(plot_count, 1, sharex=True, figsize=(15, 7))

	if not isinstance(axes, Iterable):
		axes = [axes,]

	# plot the original ts
	if core.is_array_like(ts):
		axes[current].plot(np.arange(len(ts)), ts)
		axes[current].set_ylabel('Data', size=14)
		current += 1

	# plot the original query
	if core.is_array_like(query):
		axes[current].plot(np.arange(len(query)), query)
		axes[current].set_ylabel('Query', size=14)
		current += 1

	# plot matrix profile
	if core.is_array_like(mp):
		mp_adj = np.append(mp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_ylabel('Matrix Profile', size=14)
		current += 1

	# plot left matrix profile
	if core.is_array_like(lmp):
		mp_adj = np.append(lmp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_ylabel('Left Matrix Profile', size=14)
		current += 1

	# plot left matrix profile
	if core.is_array_like(rmp):
		mp_adj = np.append(rmp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_ylabel('Right Matrix Profile', size=14)
		current += 1

	fig.tight_layout()

	return fig


def plot_discords(obj):
	"""
	Plot discords.

	Parameters
	----------
	obj : dict_like
		The matrix profile object to plot.

	Returns
	-------
	The matplotlib figure object.
	"""
	mp = obj['mp']
	w = obj['w']
	discords = obj['discords']
	data = obj.get('data', None)
	if data:
		ts = data.get('ts', None)

	mp_adjusted = np.append(mp, np.full(w + 1, np.nan))

	fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 7), gridspec_kw={'height_ratios': [25, 5, 25]})

	pos = axes[1].imshow([mp_adjusted,], aspect='auto', cmap='coolwarm')
	axes[1].set_yticks([])
	axes[0].plot(np.arange(len(ts)), ts)
	axes[0].set_ylabel('Data', size=14)

	axes[2].plot(np.arange(len(mp_adjusted)), mp_adjusted)
	axes[2].set_ylabel('Matrix Profile', size=14)

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


def plot_motifs(obj):
	"""
	Plot motifs.

	Parameters
	----------
	obj : dict_like
		The matrix profile object to plot.

	Returns
	-------
	A list of matplotlib figure objects.
	"""
	figures = []

	mp = obj['mp']
	w = obj['w']
	motifs = obj['motifs']
	data = obj.get('data', None)
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