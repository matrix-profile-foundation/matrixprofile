import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
from glob import glob

SOURCE_URL = 'https://github.com/target/matrixprofile-ts'

# Glob compliant file paths used to find Cython files for compilation
EXTENSION_PATHS = [
    'matrixprofile/algorithms/*.pyx',
    'matrixprofile/*.pyx'
]

# Skip these extensions explicitly
SKIP_EXTENSIONS = [
    'matrixprofile/algorithms/cympx.pyx',
    'matrixprofile/algorithms/cympx_ab.pyx',
]

def skip_extension(path):
    skip = False
    for skip in SKIP_EXTENSIONS:
        if skip in path:
            skip = True
    
    return skip

def find_extensions():
    """Utility script that finds Cython files to be compiled. It makes use of
    EXTENSION_PATHS global variable.

    Note
    ----
    The namespace of the compiled files will be the path of the directory in
    which it was found in. For example, matrixprofile/algorithms/mpx.pyx
    would be compiled into the namespace "matrixprofile.algorithms.mpx".

    Returns
    -------
    list(Extension) :
        A list of Extension object instances for Cython compilation.
    """
    extensions = []

    for ep in EXTENSION_PATHS:
        for fp in glob(ep):
            if not skip_extension(fp):
                module_path = fp.replace(os.path.sep, '.').replace('.pyx', '')
                extensions.append(
                    Extension(module_path, [fp,])
                )
    
    return extensions

extensions = find_extensions()
extensions.append(Extension(
    'matrixprofile.algorithms.cympx',
    ['matrixprofile/algorithms/cympx.pyx'],
    extra_compile_args = ["-O3", "-march=native", "-fopenmp" ],
))
extensions.append(Extension(
    'matrixprofile.algorithms.cympx_ab',
    ['matrixprofile/algorithms/cympx_ab.pyx'],
    extra_compile_args = ["-O3", "-march=native", "-fopenmp" ],
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
