import setuptools

from setuptools import dist
dist.Distribution().fetch_build_eggs(['cython>=0.x', 'numpy>=1.16.2', 'wheel'])

from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

import os, sys
from glob import glob

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR_PATH)

import version

SOURCE_URL = 'https://github.com/matrix-profile-foundation/matrixprofile'
README = os.path.join(DIR_PATH, 'README.rst')

# manual list of files to be compiled
extensions = []
extensions.append(Extension(
    'matrixprofile.algorithms.cympx',
    [os.path.join(DIR_PATH, 'matrixprofile', 'algorithms', 'cympx.pyx')],
    extra_compile_args = ["-O2", "-fopenmp" ],
    extra_link_args = ['-fopenmp'],
    include_dirs=[numpy.get_include()],
))

extensions.append(Extension(
    'matrixprofile.cycore',
    [os.path.join(DIR_PATH, 'matrixprofile', 'cycore.pyx')],
    extra_compile_args = ["-O2",],
    include_dirs=[numpy.get_include()],
))

matplot = 'matplotlib>=3.0.3'
scipy = 'scipy>=1.3.2,<2.0.0'
if sys.version_info.major == 3:
    with open(README, 'r', encoding='utf-8') as fh:
        long_description = fh.read()
elif sys.version_info.major == 2:
    matplot = 'matplotlib'
    scipy = 'scipy<2.0.0'
    with open(README, 'r') as fh:
        long_description = fh.read()

# copy version file over
with open(os.path.join(DIR_PATH, 'version.py')) as fh:
    with open(os.path.join(DIR_PATH, 'matrixprofile', 'version.py'), 'w') as out:
        out.write(fh.read())

setuptools.setup(
    name="matrixprofile",
    version=version.__version__,
    author="Matrix Profile Foundation",
    author_email="tylerwmarrs@gmail.com",
    description="An open source time series data mining library based on Matrix Profile algorithms.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url=SOURCE_URL,
    project_urls={
        'Matrix Profile Foundation': 'https://matrixprofile.org',
        'Source Code': SOURCE_URL,
    },
    packages = setuptools.find_packages(),
    setup_requires=['cython>=0.x', 'wheel'],
    install_requires=['numpy>=1.16.2', matplot, 'protobuf==3.11.2', scipy],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Developers",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    keywords="matrix profile time series discord motif analysis data science anomaly pattern",
)
