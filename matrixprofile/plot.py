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

from matrixprofile import core


def plot(obj, data=None):
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
