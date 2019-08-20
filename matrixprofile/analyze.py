# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

from matrixprofile.algorithms.top_k_discords import top_k_discords
from matrixprofile.algorithms.top_k_motifs import top_k_motifs