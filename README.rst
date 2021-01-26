.. image:: https://matrixprofile.org/static/img/mpf-logo.png
    :target: https://matrixprofile.org
    :height: 300px
    :scale: 50%
    :alt: MPF Logo
|
|
.. image:: https://img.shields.io/pypi/v/matrixprofile.svg
    :target: https://pypi.org/project/matrixprofile/
    :alt: PyPI Version
.. image:: https://pepy.tech/badge/matrixprofile
    :target: https://pepy.tech/project/matrixprofile
    :alt: PyPI Downloads
.. image:: https://img.shields.io/conda/vn/conda-forge/matrixprofile.svg
    :target: https://anaconda.org/conda-forge/matrixprofile
    :alt: Conda Version
.. image:: https://img.shields.io/conda/dn/conda-forge/matrixprofile.svg
    :target: https://anaconda.org/conda-forge/matrixprofile
    :alt: Conda Downloads
.. image:: https://codecov.io/gh/matrix-profile-foundation/matrixprofile/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/matrix-profile-foundation/matrixprofile
    :alt: Code Coverage
.. image:: https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/matrixprofile-feedstock?branchName=master
    :target: https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=11637&branchName=master
    :alt: Azure Pipelines
.. image:: https://api.travis-ci.com/matrix-profile-foundation/matrixprofile.svg?branch=master
    :target: https://travis-ci.com/matrix-profile-foundation/matrixprofile
    :alt: Build Status
.. image:: https://img.shields.io/conda/pn/conda-forge/matrixprofile.svg
    :target: https://anaconda.org/conda-forge/matrixprofile
    :alt: Platforms
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: License
.. image:: https://img.shields.io/twitter/follow/matrixprofile.svg?style=social
    :target: https://twitter.com/matrixprofile
    :alt: Twitter
.. image:: https://img.shields.io/discord/589321741277462559?logo=discord
    :target: https://discordapp.com/invite/sBhDNXT
    :alt: Discord
.. image:: https://joss.theoj.org/papers/10.21105/joss.02179/status.svg
   :target: https://doi.org/10.21105/joss.02179
   :alt: JOSSDOI
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3789780.svg
   :target: https://doi.org/10.5281/zenodo.3789780
   :alt: ZenodoDOI

MatrixProfile
----------------
MatrixProfile is a Python 3 library, brought to you by the `Matrix Profile Foundation <https://matrixprofile.org>`_, for mining time series data. The Matrix Profile is a novel data structure with corresponding algorithms (stomp, regimes, motifs, etc.) developed by the `Keogh <https://www.cs.ucr.edu/~eamonn/MatrixProfile.html>`_ and `Mueen <https://www.cs.unm.edu/~mueen/>`_ research groups at UC-Riverside and the University of New Mexico. The goal of this library is to make these algorithms accessible to both the novice and expert through standardization of core concepts, a simplistic API, and sensible default parameter values.

In addition to this Python library, the Matrix Profile Foundation, provides implementations in other languages. These languages have a pretty consistent API allowing you to easily switch between them without a huge learning curve.

* `tsmp <https://github.com/matrix-profile-foundation/tsmp>`_ - an R implementation
* `go-matrixprofile <https://github.com/matrix-profile-foundation/go-matrixprofile>`_ - a Golang implementation

Python Support
----------------
Currently, we support the following versions of Python:

* 3.5
* 3.6
* 3.7
* 3.8
* 3.9

Python 2 is no longer supported. There are earlier versions of this library that support Python 2.

Installation
------------
The easiest way to install this library is using pip or conda. If you would like to install it from source, please review the `installation documentation <http://matrixprofile.docs.matrixprofile.org/install.html>`_ for your platform.

Installation with pip

.. code-block:: bash

   pip install matrixprofile

Installation with conda

.. code-block:: bash

   conda config --add channels conda-forge
   conda install matrixprofile

Getting Started
---------------
This article provides introductory material on the Matrix Profile:
`Introduction to Matrix Profiles  <https://towardsdatascience.com/introduction-to-matrix-profiles-5568f3375d90>`_


This article provides details about core concepts introduced in this library:
`How To Painlessly Analyze Your Time Series  <https://towardsdatascience.com/how-to-painlessly-analyze-your-time-series-f52dab7ea80d>`_

Our documentation provides a `quick start guide <http://matrixprofile.docs.matrixprofile.org/Quickstart.html>`_, `examples <http://matrixprofile.docs.matrixprofile.org/examples.html>`_ and `api <http://matrixprofile.docs.matrixprofile.org/api.html>`_ documentation. It is the source of truth for getting up and running.

Algorithms
----------
For details about the algorithms implemented, including performance characteristics, please refer to the `documentation <http://matrixprofile.docs.matrixprofile.org/Algorithms.html>`_.
            
------------
Getting Help
------------
We provide a dedicated `Discord channel <https://discordapp.com/invite/sBhDNXT>`_ where practitioners can discuss applications and ask questions about the Matrix Profile Foundation libraries. If you rather not join Discord, then please open a `Github issue <https://github.com/matrix-profile-foundation/matrixprofile/issues>`_.

------------
Contributing
------------
Please review the `contributing guidelines <http://matrixprofile.docs.matrixprofile.org/contributing.html>`_ located in our documentation.

---------------
Code of Conduct
---------------
Please review our `Code of Conduct documentation <http://matrixprofile.docs.matrixprofile.org/code_of_conduct.html>`_.

---------
Citations
---------
All proper acknowledgements for works of others may be found in our `citation documentation <http://matrixprofile.docs.matrixprofile.org/citations.html>`_.

------
Citing
------
Please cite this work using the `Journal of Open Source Software article <https://joss.theoj.org/papers/10.21105/joss.02179>`_.

    Van Benschoten et al., (2020). MPA: a novel cross-language API for time series analysis. Journal of Open Source Software, 5(49), 2179, https://doi.org/10.21105/joss.02179

.. code:: bibtex

    @article{Van Benschoten2020,
        doi = {10.21105/joss.02179},
        url = {https://doi.org/10.21105/joss.02179},
        year = {2020},
        publisher = {The Open Journal},
        volume = {5},
        number = {49},
        pages = {2179},
        author = {Andrew Van Benschoten and Austin Ouyang and Francisco Bischoff and Tyler Marrs},
        title = {MPA: a novel cross-language API for time series analysis},
        journal = {Journal of Open Source Software}
    }
