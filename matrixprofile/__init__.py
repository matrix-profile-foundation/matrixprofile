name = "matrixprofile"

import sys
sys.path.append('../')

from matrixprofile.version import __version__, __version_info__

from matrixprofile.compute import compute
from matrixprofile.visualize import visualize
from matrixprofile.analyze import analyze
from matrixprofile import discover
from matrixprofile import transform
from matrixprofile import utils
from matrixprofile import io
from matrixprofile import algorithms
from matrixprofile import datasets