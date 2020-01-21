# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

__all__ = [
    'to_json',
    'from_json',
    'to_disk',
    'from_disk',
]

import json.tool

import numpy as np

from matrixprofile import core
from matrixprofile.io.protobuf.protobuf_utils import (
    to_mpf,
    from_mpf
)


# Supported file extensions
SUPPORTED_EXTS = set([
    'json',
    'mpf',
])

# Supported file formats
SUPPORTED_FORMATS = set([
    'json',
    'mpf',
])

def JSONSerializer(obj):
    """
    Default JSON serializer to write numpy arays and other non-supported
    data types.

    Borrowed from:
    https://stackoverflow.com/a/52604722
    """
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()

    raise TypeError('Unknown type:', type(obj))


def from_json(profile):
    """
    Converts a JSON formatted string into a profile data structure.

    Parameters
    ----------
    profile : str
        The profile as a JSON formatted string.

    Returns
    -------
    profile : dict_like
        A MatrixProfile or Pan-MatrixProfile data structure.
    """
    dct = json.load(profile)

    # handle pmp and convert to appropriate types
    if core.is_pmp_obj(dct):
        dct['pmp'] = np.array(dct['pmp'], dtype='float64')
        dct['pmpi'] = np.array(dct['pmpi'], dtype=int)
        dct['data']['ts'] = np.array(dct['data']['ts'], dtype='float64')
        dct['windows'] = np.array(dct['windows'], dtype=int)

    # handle mp
    elif core.is_mp_obj(dct):
        dct['mp'] = np.array(dct['mp'], dtype='float64')
        dct['pi'] = np.array(dct['pi'], dtype=int)

        has_l = isinstance(dct['lmp'], list)
        has_l = has_l and isinstance(dct['lpi'], list)

        if has_l:
            dct['lmp'] = np.array(dct['lmp'], dtype='float64')
            dct['lpi'] = np.array(dct['lpi'], dtype=int)

        has_r = isinstance(dct['rmp'], list)
        has_r = has_r and isinstance(dct['rpi'], list)
        
        if has_r:
            dct['rmp'] = np.array(dct['rmp'], dtype='float64')
            dct['rpi'] = np.array(dct['rpi'], dtype=int)
        
        dct['data']['ts'] = np.array(dct['data']['ts'], dtype='float64')

        if isinstance(dct['data']['query'], list):
            dct['data']['query'] = np.array(dct['data']['query'], dtype='float64')
    else:
        raise ValueError('File is not of type profile!')

    return dct


def to_json(profile):
    """
    Converts a given profile object into JSON format.

    Parameters
    ----------
    profile : dict_like
        A MatrixProfile or Pan-MatrixProfile data structure.

    Returns
    -------
    str :
        The profile as a JSON formatted string.
    """
    if not core.is_mp_or_pmp_obj(profile):
        raise ValueError('profile is expected to be of type MatrixProfile or PMP')

    return json.dumps(profile, default=JSONSerializer)


def add_extension_to_path(file_path, extension):
    """
    Utility function to add the file extension when it is not provided by the
    user in the file path.

    Parameters
    ----------
    file_path : str
        The file path.

    Returns
    -------
    str :
        The file path with the extension appended.
    str :
        The file format extension.
    """
    end = '.{}'.format(extension)
    if not file_path.endswith(end):
        file_path = '{}{}'.format(file_path, end)

    return file_path


def infer_file_format(file_path):
    """
    Attempts to determine the file type based on the extension. The extension
    is assumed to be the last dot suffix.

    Parameters
    ----------
    file_path : str
        The file path to infer the file format of.
    
    Returns
    -------
    str :
        A label described the file extension.
    """
    pieces = file_path.split('.')
    extension = pieces[-1].lower()

    if extension not in SUPPORTED_EXTS:
        raise RuntimeError('Unsupported file type with extension {}'.format(extension))

    return extension


def to_disk(profile, file_path, format='json'):
    """
    Writes a profile object of type MatrixProfile or PMP to disk as a JSON
    formatted file by default.

    Note
    ----
    The JSON format is human readable where as the mpf format is binary and
    cannot be read when opened in a text editor. When the file path does not
    include the extension, it is appended for you.

    Parameters
    ----------
    profile : dict_like
        A MatrixProfile or Pan-MatrixProfile data structure.
    file_path : str
        The path to write the file to.
    format : str, default json
        The format of the file to be written. Options include json, mpf
    """
    if not core.is_mp_or_pmp_obj(profile):
        raise ValueError('profile is expected to be of type MatrixProfile or PMP')

    if format not in SUPPORTED_FORMATS:
        raise ValueError('Unsupported file format {} given.'.format(format))

    file_path = add_extension_to_path(file_path, format)

    if format == 'json':
        with open(file_path, 'w') as out:
            out.write(to_json(profile))
    elif format == 'mpf':
        with open(file_path, 'wb') as out:
            out.write(to_mpf(profile))


def from_disk(file_path, format='infer'):
    """
    Reads a profile object of type MatrixProfile or PMP from disk into the
    respective object type. By default the type is inferred by the file
    extension.

    Parameters
    ----------
    file_path : str
        The path to read the file from.
    format : str, default infer
        The file format type to read from disk. Options include:
        infer, json, mpf
    
    Returns
    -------
    profile : dict_like, None
        A MatrixProfile or Pan-MatrixProfile data structure.
    """
    if format != 'infer':
        if format not in SUPPORTED_FORMATS:
            raise ValueError('format supplied {} is not supported'.format(format))
    else:
        format = infer_file_format(file_path)
    
    profile = None
    if format == 'json':
        with open(file_path) as f:
            profile = from_json(f)
    elif format == 'mpf':
        with open(file_path, 'rb') as f:
            profile = from_mpf(f.read())
    
    return profile