import setuptools

from setuptools import dist
dist.Distribution().fetch_build_eggs(['cython>=0.x', 'numpy>=1.16.2', 'wheel'])

from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

import os, sys
from glob import glob

SOURCE_URL = 'https://github.com/target/matrixprofile-ts'

# manual list of files to be compiled
extensions = []
extensions.append(Extension(
    'matrixprofile.algorithms.cympx',
    ['matrixprofile/algorithms/cympx.pyx'],
    extra_compile_args = ["-O3", "-march=native", "-fopenmp" ],
    extra_link_args = ['-fopenmp'],
    include_dirs=[numpy.get_include()],
))

extensions.append(Extension(
    'matrixprofile.cycore',
    ['matrixprofile/cycore.pyx'],
    extra_compile_args = ["-O3", "-march=native"],
    include_dirs=[numpy.get_include()],
))

matplot = 'matplotlib>=3.0.3'
if sys.version_info.major == 3:
    with open('README.rst', 'r', encoding='utf-8') as fh:
        long_description = fh.read()
elif sys.version_info.major == 2:
    matplot = 'matplotlib'
    with open('README.rst', 'r') as fh:
        long_description = fh.read()

setuptools.setup(
    name="matrixprofile-ts",
    version="1.0.0",
    author="Matrix Profile Foundation",
    author_email="avbs89@gmail.com, tylerwmarrs@gmail.com",
    description="An Open Source Python Time Series Library For Motif Discovery using Matrix Profile",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url=SOURCE_URL,
    project_urls={
        'Matrix Profile Foundation': 'https://matrixprofile.org',
        'Source Code': SOURCE_URL,
    },
    packages = setuptools.find_packages(),
    setup_requires=['cython>=0.x', 'wheel'],
    install_requires=['numpy>=1.16.2', matplot, 'protobuf==3.11.2'],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
