# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import csv
import gzip
import json
import os

# load urlretrieve for python2 and python3
try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

import numpy as np
    
DATA_LISTING_URL = 'https://raw.githubusercontent.com/matrix-profile-foundation/mpf-datasets/master/listings.json'
DATA_URL = 'https://raw.githubusercontent.com/matrix-profile-foundation/mpf-datasets/master/{}/{}'
DATA_DIR = os.path.expanduser(os.path.join('~', '.mpf-datasets'))


def create_dirs(path):
    """
    Python 2 and 3 compatible function to make directories. Python 3 has the
    exist_ok option in makedirs, but Python 2 does not.

    Parameters
    ----------
    path : str
        The path to create directories for.

    """
    try:
        os.makedirs(path)
    except:
        pass

    if not os.path.exists(path):
        raise OSError('Unable to create path: {}'.format(path))


def fetch_available(category=None):
    """
    Fetches the available datasets found in
    github.com/matrix-profile-foundation/mpf-datasets github repository.
    Providing a category filters the datasets.
    
    Parameters
    ----------
    category : str, Optional
        The desired category to retrieve datasets by.
    
    Returns
    -------
    list :
        A list of dictionaries containing details about each dataset.
    
    Raises
    ------
    ValueError:
        When a category is provided, but is not found in the listing.

    """
    # download the file and load it
    create_dirs(DATA_DIR)
    output_path = os.path.join(DATA_DIR, 'listings.json')
    result = urlretrieve(DATA_LISTING_URL, output_path)
    
    with open(output_path) as f:
        datasets = json.load(f)
    
    # filter with category
    if category:
        category_found = False
        filtered = []
        
        for dataset in datasets:
            if dataset['category'] == category.lower():
                filtered.append(dataset)
                category_found = True
        
        datasets = filtered
        if not category_found:
            raise ValueError('category {} is not a valid option.'.format(category))
    
    return datasets


def get_csv_indices(fp, is_gzip=False):
    """
    Utility function to provide indices of the datetime dimension and the
    real valued dimensions.
    
    Parameters
    ----------
    fp : str
        The filepath to load.
    is_gzip : boolean, Default False
        Flag to tell if the csv is gzipped.
    
    Returns
    -------
    (dt_index, real_indices) :
        The datetime index and real valued indices.

    """
    first_line = None
    if is_gzip:
        with gzip.open(fp, 'rt') as f:
            first_line = f.readline()
    else:
        with open(fp) as f:
            first_line = f.readline()
    
    dt_index = None
    real_indices = []
    for index, label in enumerate(first_line.split(',')):
        if 'date' in label.lower() or 'time' in label.lower():
            dt_index = index
        else:
            real_indices.append(index)
    
    return dt_index, real_indices


def load(name):
    """
    Loads a MPF dataset by base file name or file name. The match is case 
    insensitive.

    Note
    ----
    An internet connection is required to fetch the data.

    Returns
    -------
    dict :
        The dataset and metadata.

        >>> {
        >>>     'name': The file name loaded,
        >>>     'category': The category the file came from,
        >>>     'description': A description,
        >>>     'data': The real valued data as an np.ndarray,
        >>>     'datetime': The datetime as an np.ndarray
        >>> }

    """
    datasets = fetch_available()
    
    # find the filename in datasets matching either on filename provided or
    # the base name
    filename = None
    category = None
    description = None
    for dataset in datasets:
        base_name = dataset['name'].split('.')[0]
        
        if name.lower() == base_name or name.lower() == dataset['name']:
            filename = dataset['name']
            category = dataset['category']
            description = dataset['description']
    
    if not filename:
        raise ValueError('Could not find dataset {}'.format(name))
        
    # download the file
    output_dir = os.path.join(DATA_DIR, category)
    create_dirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(output_path):
        url = DATA_URL.format(category, filename)
        urlretrieve(url, output_path)

    # load the file based on type
    is_txt = filename.endswith('.txt')    
    is_txt_gunzip = filename.endswith('.txt.gz')
    is_csv = filename.endswith('.csv')
    is_csv_gunzip = filename.endswith('.csv.gz')
    
    data = None
    dt_data = None
    if is_txt or is_txt_gunzip:
        data = np.loadtxt(output_path)
    elif is_csv or is_csv_gunzip:
        dt_index, real_indices = get_csv_indices(
            output_path, is_gzip=is_csv_gunzip)

        if isinstance(dt_index, int):
            dt_data = np.genfromtxt(
                output_path,
                dtype='datetime64',
                delimiter=',',
                skip_header=True,
                usecols=[dt_index,]
            )

        data = np.genfromtxt(
            output_path,
            delimiter=',',
            dtype='float64',
            skip_header=True,
            usecols=real_indices
        )
    
    return {
        'name': filename,
        'category': category,
        'description': description,
        'data': data,
        'datetime': dt_data
    }
