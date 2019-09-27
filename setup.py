import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
from glob import glob

EXTENSION_PATHS = [
    'matrixprofile/algorithms/*.pyx',
    'matrixprofile/*.pyx'
]

def find_extensions():
    extensions = []

    for ep in EXTENSION_PATHS:
        for fp in glob(ep):
            module_path = fp.replace(os.path.sep, '.').replace('.pyx', '')
            extensions.append(
                Extension(module_path, [fp,])
            )
    
    return extensions

extensions = find_extensions()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matrixprofile-ts",
    version="0.0.6",
    author="Andrew Van Benschoten",
    author_email="avbs89@gmail.com",
    description="An Open Source Python Time Series Library For Motif Discovery using Matrix Profile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/target/matrixprofile-ts",
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
