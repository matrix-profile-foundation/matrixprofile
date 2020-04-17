# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate


import numpy as np

from matrixprofile import core
from matrixprofile.io.protobuf.proto_messages_pb2 import (
    Location, Motif, MPFOutput
)


def get_matrix_attributes(matrix):
    """
    Utility function to extract the rows, cols and flattened array from a
    numpy array so it can be stored in the MPFOutput protobuf message.

    Parameters
    ----------
    matrix : np.ndarray
        The numpy array to extract the attributes from.

    Returns
    -------
    tuple :
        A tuple containing the rows, cols and flattened array.
    """
    if not core.is_array_like(matrix) or len(matrix) < 1:
        return None, None, None

    rows = matrix.shape[0]
    cols = 0
    if len(matrix.shape) > 1:
        cols = matrix.shape[1]

    return rows, cols, matrix.flatten()


def get_windows(profile):
    """
    Utility function to format the windows from a profile structure ensuring
    that the windows are in an array.

    Parameters
    ----------
    profile : dict
        The MatrixProfile or PMP profile.

    Returns
    -------
    list :
        The window(s) in a list.
    """
    windows = []

    if core.is_mp_obj(profile):
        windows.append(profile.get('w'))
    elif core.is_pmp_obj(profile):
        windows = profile.get('windows')

    return windows


def get_proto_motif(motif):
    """
    Utility function to convert a motif from a MatrixProfile or PMP structure
    ensuring that it is compatible with the MPFOutput message.

    Note
    ----
    A single dimensional motif location will only have a row index and
    a column index of 0.

    Parameters
    ----------
    motif : dict
        The motif to convert.

    Returns
    -------
    Motif :
        The motif object for MPFOutput message.
    """
    out_motif = Motif()

    for indices in motif['motifs']:
        tmp = Location()
        tmp.row = 0
        tmp.col = 0

        # handle single integer location
        if core.is_array_like(indices):
            tmp.row = indices[0]
            tmp.col = indices[1]
        else:
            tmp.row = indices

        out_motif.motifs.append(tmp)

    for neighbor in motif['neighbors']:
        tmp = Location()
        tmp.row = 0
        tmp.col = 0

        # handle single integer location
        if core.is_array_like(neighbor):
            tmp.row = neighbor[0]
            tmp.col = neighbor[1]
        else:
            tmp.row = neighbor

        out_motif.neighbors.append(tmp)

    return out_motif


def get_proto_discord(discord):
    """
    Utility function to convert a discord into the MPFOutput message
    format.

    Note
    ----
    A single dimensional discord location will only have a row index and
    a column index of 0.

    Parameters
    ----------
    discord : int or tuple
        The discord with row, col index or single index.

    Returns
    -------
    Location :
        The Location message used in the MPFOutput protobuf message.
    """
    out_discord = Location()
    out_discord.row = 0
    out_discord.col = 0

    if core.is_array_like(discord):
        out_discord.row = discord[0]
        out_discord.col = discord[1]
    else:
        out_discord.row = discord

    return out_discord


def profile_to_proto(profile):
    """
    Utility function that takes a MatrixProfile or PMP profile data structure
    and converts it to the MPFOutput protobuf message object.

    Parameters
    ----------
    profile : dict
        The profile to convert.

    Returns
    -------
    MPFOutput :
        The MPFOutput protobuf message object.
    """
    output = MPFOutput()

    # add higher level attributes that work for PMP and MP
    output.klass = profile.get('class')
    output.algorithm = profile.get('algorithm')
    output.metric = profile.get('metric')
    output.sample_pct = profile.get('sample_pct')

    # add time series data
    ts = profile.get('data').get('ts')
    query = profile.get('data').get('query')
    rows, cols, data = get_matrix_attributes(ts)
    output.ts.rows = rows
    output.ts.cols = cols
    output.ts.data.extend(data)

    # add query data
    query = profile.get('data').get('query')
    rows, cols, data = get_matrix_attributes(query)

    if rows and cols and core.is_array_like(data):
        output.query.rows = rows
        output.query.cols = cols
        output.query.data.extend(data)

    # add window(s)
    output.windows.extend(get_windows(profile))

    # add motifs
    motifs = profile.get('motifs')
    if not isinstance(motifs, type(None)):
        for motif in motifs:
            output.motifs.append(get_proto_motif(motif))

    # add discords
    discords = profile.get('discords')
    if not isinstance(discords, type(None)):
        for discord in discords:
            output.discords.append(get_proto_discord(discord))

    # add cmp
    cmp = profile.get('cmp')
    if not isinstance(cmp, type(None)):
        rows, cols, data = get_matrix_attributes(cmp)

        output.cmp.rows = rows
        output.cmp.cols = cols
        output.cmp.data.extend(data)

    # add av
    av = profile.get('av')
    if not isinstance(av, type(None)):
        rows, cols, data = get_matrix_attributes(av)

        output.av.rows = rows
        output.av.cols = cols
        output.av.data.extend(data)

    # add av_type
    av_type = profile.get('av_type')
    if not isinstance(av_type, type(None)) and len(av_type) > 0:
        output.av_type = av_type

    # add the matrix profile specific attributes
    if core.is_mp_obj(profile):
        output.mp.ez = profile.get('ez')
        output.mp.join = profile.get('join')

        # add mp
        rows, cols, data = get_matrix_attributes(profile.get('mp'))
        output.mp.mp.rows = rows
        output.mp.mp.cols = cols
        output.mp.mp.data.extend(data)

        # add pi
        rows, cols, data = get_matrix_attributes(profile.get('pi'))
        output.mp.pi.rows = rows
        output.mp.pi.cols = cols
        output.mp.pi.data.extend(data)

        # add lmp
        rows, cols, data = get_matrix_attributes(profile.get('lmp'))
        if rows and cols and core.is_array_like(data):
            output.mp.lmp.rows = rows
            output.mp.lmp.cols = cols
            output.mp.lmp.data.extend(data)

        # add lpi
        rows, cols, data = get_matrix_attributes(profile.get('lpi'))
        if rows and cols and core.is_array_like(data):
            output.mp.lpi.rows = rows
            output.mp.lpi.cols = cols
            output.mp.lpi.data.extend(data)

        # add rmp
        rows, cols, data = get_matrix_attributes(profile.get('rmp'))
        if rows and cols and core.is_array_like(data):
            output.mp.rmp.rows = rows
            output.mp.rmp.cols = cols
            output.mp.rmp.data.extend(data)

        # add rpi
        rows, cols, data = get_matrix_attributes(profile.get('rpi'))
        if rows and cols and core.is_array_like(data):
            output.mp.rpi.rows = rows
            output.mp.rpi.cols = cols
            output.mp.rpi.data.extend(data)

    # add the pan matrix profile specific attributes
    elif core.is_pmp_obj(profile):
        # add pmp
        rows, cols, data = get_matrix_attributes(profile.get('pmp'))
        output.pmp.pmp.rows = rows
        output.pmp.pmp.cols = cols
        output.pmp.pmp.data.extend(data)

        # add pmpi
        rows, cols, data = get_matrix_attributes(profile.get('pmpi'))
        output.pmp.pmpi.rows = rows
        output.pmp.pmpi.cols = cols
        output.pmp.pmpi.data.extend(data)

    else:
        raise ValueError('Expecting Pan-MatrixProfile or MatrixProfile!')

    return output


def to_mpf(profile):
    """
    Converts a given profile object into MPF binary file format.

    Parameters
    ----------
    profile : dict_like
        A MatrixProfile or Pan-MatrixProfile data structure.

    Returns
    -------
    str :
        The profile as a binary formatted string.
    """
    obj = profile_to_proto(profile)
    return obj.SerializeToString()


def from_proto_to_array(value):
    """
    Utility function to convert a protobuf array back into the correct
    dimensions.

    Parameters
    ----------
    value : array_like
        The array to transform.

    Returns
    -------
    np.ndarray :
        The transformed array.
    """
    if isinstance(value, type(None)) or len(value.data) < 1:
        return None

    shape = (value.rows, value.cols)
    out = np.array(value.data)

    if shape[1] > 0:
        out = out.reshape(shape)

    return out


def discords_from_proto(discords, is_one_dimensional=False):
    """
    Utility function to transform discord locations back to single dimension
    or multi-dimension location.

    Parameter
    ---------
    discords : array_like
        The protobuf formatted array.
    is_one_dimensional : boolean
        A flag to indicate if the original locations should be 1D.

    Returns
    -------
    np.ndarray :
        The transformed discord locations.
    """
    out = []

    for discord in discords:
        if is_one_dimensional:
            out.append(discord.row)
        else:
            out.append((discord.row, discord.col))

    return np.array(out, dtype=int)


def motifs_from_proto(motifs, is_one_dimensional=False):
    """
    Utility function to transform motif locations back to single dimension
    or multi-dimension location.

    Parameter
    ---------
    motifs : array_like
        The protobuf formatted array.
    is_one_dimensional : boolean
        A flag to indicate if the original locations should be 1D.

    Returns
    -------
    list :
        The transformed motif locations.
    """
    out = []

    for motif in motifs:
        tmp = {'motifs': [], 'neighbors': []}

        for location in motif.motifs:
            if is_one_dimensional:
                tmp['motifs'].append(location.row)
            else:
                tmp['motifs'].append((location.row, location.col))

        for neighbor in motif.neighbors:
            if is_one_dimensional:
                tmp['neighbors'].append(neighbor.row)
            else:
                tmp['neighbors'].append((neighbor.row, neighbor.col))

        out.append(tmp)

    return out


def from_mpf(profile):
    """
    Converts binary formatted MPFOutput message into a profile data structure.

    Parameters
    ----------
    profile : str
        The profile as a binary formatted MPFOutput message.

    Returns
    -------
    profile : dict_like
        A MatrixProfile or Pan-MatrixProfile data structure.
    """
    obj = MPFOutput()
    obj.ParseFromString(profile)

    out = {}
    is_one_dimensional = False

    # load in all higher level attributes
    out['class'] = obj.klass
    out['algorithm'] = obj.algorithm
    out['metric'] = obj.metric
    out['sample_pct'] = obj.sample_pct
    out['data'] = {
        'ts': from_proto_to_array(obj.ts),
        'query': from_proto_to_array(obj.query)
    }

    if obj.klass == 'MatrixProfile':
        out['mp'] = from_proto_to_array(obj.mp.mp)
        out['pi'] = from_proto_to_array(obj.mp.pi)
        out['lmp'] = from_proto_to_array(obj.mp.lmp)
        out['lpi'] = from_proto_to_array(obj.mp.lpi)
        out['rmp'] = from_proto_to_array(obj.mp.rmp)
        out['rpi'] = from_proto_to_array(obj.mp.rpi)
        out['ez'] = obj.mp.ez
        out['join'] = obj.mp.join
        out['w'] = obj.windows[0]

        is_one_dimensional = len(out['mp'].shape) == 1

    elif obj.klass == 'PMP':
        out['pmp'] = from_proto_to_array(obj.pmp.pmp)
        out['pmpi'] = from_proto_to_array(obj.pmp.pmpi)
        out['windows'] = np.array(obj.windows)

    if not isinstance(obj.discords, type(None)) and len(obj.discords) > 0:
        out['discords'] = discords_from_proto(
            obj.discords, is_one_dimensional=is_one_dimensional)

    if not isinstance(obj.motifs, type(None)) and len(obj.motifs) > 0:
        out['motifs'] = motifs_from_proto(
            obj.motifs, is_one_dimensional=is_one_dimensional)

    if not isinstance(obj.cmp, type(None)) and len(obj.cmp.data) > 0:
        out['cmp'] = from_proto_to_array(obj.cmp)

    if not isinstance(obj.av, type(None)) and len(obj.av.data) > 0:
        out['av'] = from_proto_to_array(obj.av)

    if not isinstance(obj.av_type, type(None)) and len(obj.av_type) > 0:
        out['av_type'] = obj.av_type

    return out
