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


def visualize(obj, data=None):
	"""
	Plots an object generated from one of the algorithms. In some cases
	multiple plots will be generated.

	Parameters
	----------
	obj : dict_like
		The object to plot.
	data : array_like
		The original data used in computation.

	Returns
	-------
	Nothing at the moment. It is purely for display.
	"""
	mp_class = obj.get('class', None)
	func = None

	if mp_class == 'MatrixProfile':
		func = plot_mp

	return func(obj, data=data)


def plot_mp(obj, data=None):
	"""
	Plots a matrix profile object.

	Parameters
	----------
	obj : dict_like
		The matrix profile object to plot.
	data : array_like, Optional
		The original data used to compute the matrix profile.

	Returns
	-------
	Nothing at the moment. It is purely for display.
	"""
	plot_count = 0
	mp = obj.get('mp', None)
	lmp = obj.get('lmp', None)
	rmp = obj.get('rmp', None)

	for val in [data, mp, lmp, rmp]:
		if core.is_array_like(val):
			plot_count += 1

	if plot_count < 1:
		raise ValueError("Object passed has nothing to plot!")

	w = obj.get('w', None)
	if not isinstance(w, int):
		raise ValueError("Expecting window size!")

	current = 0

	fig, axes = plt.subplots(plot_count, 1, sharex=True, figsize=(20,10))

	if not isinstance(axes, Iterable):
		axes = [axes,]

	# plot the original data
	if core.is_array_like(data):
		axes[current].plot(np.arange(len(data)), data)
		axes[current].set_title('Data')
		current += 1

	# plot matrix profile
	if core.is_array_like(mp):
		mp_adj = np.append(mp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_title('Matrix Profile')
		current += 1

	# plot left matrix profile
	if core.is_array_like(lmp):
		mp_adj = np.append(lmp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_title('Left Matrix Profile')
		current += 1

	# plot left matrix profile
	if core.is_array_like(rmp):
		mp_adj = np.append(rmp, np.zeros(w - 1) + np.nan)
		axes[current].plot(np.arange(len(mp_adj)), mp_adj)
		axes[current].set_title('Right Matrix Profile')
		current += 1


def plot_discords(obj, data):
	"""
	Plot discords.
	"""
	mp = obj['mp']
	w = obj['w']
	discords = obj['discords']
	ts = data

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
	fig.legend(lines, ['Discord', 'MP'], bbox_to_anchor=(1.06, 0.44))


	fig.tight_layout()


def plot_motifs(obj, data):
	"""
	Plot motifs.
	"""
	mp = obj['mp']
	w = obj['w']
	motifs = obj['motifs']
	ts = data

	fig, axes = plt.subplots(len(motifs), 2, figsize=(15, 7), sharey='row', sharex='col')
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

	fig, axes = plt.subplots(len(motifs), 1, figsize=(15, 7), sharey='row', sharex='col')
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