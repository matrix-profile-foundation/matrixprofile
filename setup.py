import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
from glob import glob

SOURCE_URL = 'https://github.com/target/matrixprofile-ts'

# manual list of files to be compiled
extensions = []
extensions.append(Extension(
    'matrixprofile.algorithms.cympx',
    ['matrixprofile/algorithms/cympx.pyx'],
    extra_compile_args = ["-O3", "-march=native", "-fopenmp" ],
))

extensions.append(Extension(
    'matrixprofile.cycore',
    ['matrixprofile/cycore.pyx'],
    extra_compile_args = ["-O3", "-march=native"],
))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrixprofile-ts",
    version="0.1.0",
    author="Andrew Van Benschoten, Tyler Marrs (Matrix Profile Foundation)",
    author_email="avbs89@gmail.com, tylerwmarrs@gmail.com",
    description="An Open Source Python Time Series Library For Motif Discovery using Matrix Profile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=SOURCE_URL,
    project_urls={
        'Matrix Profile Foundation': 'https://matrixprofile.org',
        'Source Code': SOURCE_URL,
    },
    packages = setuptools.find_packages(),
    setup_requires=['cython>=0.x',],
    install_requires=['numpy>=1.11.3', 'matplotlib==3.0.3', 'ray==0.7.2'],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
