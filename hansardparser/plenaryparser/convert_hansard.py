""" Contains methods for converting a list of Entry objects (output from use of hansard_parser.py methods) into a dictionary or Pandas DataFrame.

"""

import os
import re
import collections
import copy
import warnings
import numpy as np
import pandas as pd

from hansardparser.plenaryparser import utils


def convert_contents(contents, metadata, attributes, to_format, verbose=False):
    """converts raw contents to a specified format.

    Arguments:
        contents : list of Entries
            list of entries as output by HansardParser._process_transcript.
        attributes : list of str
            list of Entry attributes that are desired for the output.
        metadata : Sitting obj.
            Sitting object as defined in Sitting.py.
        to_format : str
            Desired output format. Either 'list', 'df-raw', or 'df-long'.
            'df-raw' is a pandas dataframe where each row is an entry. This format is more useful for comparing the parsed transcript to the original pdf for errors.
            'df-long' is a multi-index pandas dataframe (where header and subheader are the indices). This format is more useful for analysis since speeches are organized under headers and subheaders.
            'list' is a 2d list.
        verbose : bool
            False by default. Set to True if more detailed output is desired.
    """
    metadata_values = [metadata.__dict__[k] for k in sorted(metadata.__dict__)]
    metadata_names = sorted(metadata.__dict__.keys())
    if to_format not in ['list', 'df-raw', 'df-long']:
        raise RuntimeError('to_format must be either \'list\', \'df-raw\', or \'df-long\'.')
    if to_format == 'list':
        result = contents_to_2darray(contents, attributes, metadata_values, verbose)
    if to_format == 'df-raw':
        result = contents_to_df_raw(contents, attributes, metadata_names, metadata_values)
    if to_format == 'df-long':
        result = contents_to_df_long(contents, attributes, metadata_names, metadata_values, verbose)
    return result


def contents_to_df_raw(contents, attributes, metadata_names, metadata_values):
    """Converts ontents to raw dataframe. """
    series = []
    for entry in contents:
        entry_list = [entry.__dict__[k] for k in sorted(entry.__dict__) if k in attributes]
        series.append(pd.Series(entry_list + metadata_values))
    df = pd.concat(series, axis=1).T
    # colnames = copy.deepcopy(contents[0].__dict__.keys())
    attributes = sorted(attributes)
    colnames = attributes + metadata_names
    df.columns = colnames
    df['date'] = df['date'].apply(utils.str_from_date)  # FIXME: this ignores time information.
    return df


def contents_to_df_long(contents, attributes, metadata_names, metadata_values, verbose):
    """converts a dictionary of contents (produced by contents_to_dict) to a
    pandas DataFrame. """
    contents_2d = contents_to_2darray(contents, attributes, metadata_values, verbose)
    # contents_dict_mod = collections.OrderedDict()
    columns = ['header', 'subheader', 'subsubheader'] + attributes + metadata_names
    df = pd.DataFrame(contents_2d, columns=columns)
    df['date'] = df['date'].apply(utils.str_from_date)
    return df


def contents_to_2darray(contents, attributes, metadata_values, verbose):
    """Converts a list of entries to a 2d array."""
    # transcript_dict = collections.OrderedDict()
    # NOTE TO SELF: this first while loop is a temporary block to pop entries until the first header is encountered. This may lose some valuable information at the beginning of the transript if for some reason the first header does not appear for a while or was not entered correctly in contents.
    # contents = copy.deepcopy(contents)

    # prelim = []
    # entry = contents.pop(0)
    # while entry.entry_type != 'header':
    #     prelim.append(entry)
    #     entry = contents.pop(0)
    #     if verbose and len(prelim) > 5:
    #         warnings.warn('More than 5 entries encountered before first header', RuntimeWarning)

    # contents.insert(0, entry)  # re-insert first header back into contents.
    current_header = None
    current_subheader = None
    current_subsubheader = None
    data = []
    # transcript_dict[(i, current_header, current_subheader, current_subsubheader)] = []
    # transcript_dict[current_header][current_subheader] = collections.OrderedDict()
    # transcript_dict[current_header][current_subheader][current_subsubheader] = []
    for entry in contents:
        # entry = contents.pop(0)
        if entry.entry_type in ['header', 'subheader', 'subsubheader']:
            if len(data) > 0 and data[-1][:3] != [current_header, current_subheader, current_subsubheader]:
                data.append([current_header, current_subheader, current_subsubheader] + [None]*len(attributes) + metadata_values)
            if entry.entry_type == 'header':
                current_header = entry.text
                current_subheader = None
                current_subsubheader = None
            elif entry.entry_type == 'subheader':
                current_subheader = entry.text
                current_subsubheader = None
            elif entry.entry_type == 'subsubheader':
                current_subsubheader = entry.text
        elif entry.entry_type == 'scene':
            current_scene = [getattr(entry, attr) for attr in attributes]
            data.append([current_header, current_subheader, current_subsubheader] + current_scene + metadata_values)
        elif entry.entry_type in ['speech_new', 'speech_ctd']:
            current_speech = [getattr(entry, attr) for attr in attributes]
            data.append([current_header, current_subheader, current_subsubheader] + current_speech  + metadata_values)
    # if verbose and len(transcript_dict['preliminary']['no_subheading']['no_subsubheading']) > 5:
    #     warnings.warn('More than 5 entries encountered before first header', RuntimeWarning)
    return data


def export_contents(filename, contents, output_dir, input_format, output_format, suffix=None):
    """Exports transcript contents to output_dir using filename.

    Arguments:

        filename :

        contents :

        output_dir :

        input_format :

        output_format :
    """
    if suffix is None:
        suffix = output_format
    output_filepath = '%s/%s.%s' % (output_dir, filename, output_format)
    if os.path.isfile(output_filepath):
        raise RuntimeError('File already exists.')
    if output_format == 'hdf':
        # print(contents)
        # contents.position = contents.position.astype('str')
        contents.to_hdf(output_filepath, key='table', format='table')
    elif output_format in ['csv', 'txt']:
        if input_format in ['dict']:
            raise RuntimeError('Capability to export dict to csv or txt not yet implemented.')
        delim = ',' if output_format == 'csv' else '|'
        index = False
        if input_format == 'df-long':
            index = True
        contents.to_csv(output_filepath, index=index, encoding='utf-8', sep=delim, na_rep='None')
    # if output_format == 'txt':
        # np.savetxt(output_dir + filename + '.txt', contents.to_records(), fmt='%s', delimiter='|')
    elif output_format == 'json':
        if input_format in ['dict']:
            raise RuntimeError('Capability to export dict to json not yet implemented.')
        if input_format =='df-long':
            contents.to_csv(output_filepath, index=True, encoding='utf-8')
        if input_format =='df-raw':
            contents.to_csv(output_filepath, index=False, encoding='utf-8')
    else:
        raise RuntimeError('output_format "%s" not permitted.' % output_format)
    return 0


