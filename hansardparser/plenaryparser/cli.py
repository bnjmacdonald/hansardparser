"""Implements a command-line application for parsing a corpus of transcripts.

Retrieves txt files from Box.com and parses them using the `TxtParser` class.

Usage::

    Parse a single txt transcript::
        
        python -m hansardparser.plenaryparser.cli -v 1 --filetype txt \
            --inpaths /Users/bnjmacdonald/Documents/current/projects/hansardparser/data/tests/hansards/txt/1985/1985.txt \
            --outpath data/raw/hansards


Todos:

    TODO: refactor in combination with `parse_one_transcript` so that a single
        module exposes an API that can be used for any transcript.
    
    TODO: convert to Flask app that can be deployed in Docker container.
"""

import os
import re
import json
import argparse
import numpy as np
import subprocess
import warnings

from hansardparser.plenaryparser.TxtParser import TxtParser
from hansardparser.plenaryparser.models import Entry, Sitting
from hansardparser import settings

# KLUDGE: for builder pickle imports.
from hansardparser.plenaryparser.TxtParser.LineLabeler.SupervisedLineLabeler import *  

DEFAULT_OUTPATH = os.path.join(settings.DATA_ROOT, 'temp', 'plenaryparser', 'txt')
DEFAULT_FILETYPE = 'txt'

# NOTE: builder and classifier should be set in environment by a config file.
BUILDER_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'plenaryparser', 'text2vec', 'builders', '2018-07-15T084305')
CLASSIFIER_PATH = os.path.join(settings.EXPERIMENTS_ROOT, 'plenaryparser', 'line_classifier', 'classifier0')

# TODO: allow user to specify path to label codes.
LINE_LABEL_CODES_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'plenaryparser', 'text2vec', 'corpora', '2018-07-15T084305', 'label_codes.json')
with open(LINE_LABEL_CODES_PATH, 'r') as f:
    LINE_LABEL_CODES = json.load(f)
    assert len(set(LINE_LABEL_CODES.values())) == len(LINE_LABEL_CODES), \
        'values must be unique for dict to be reversed.'
    LINE_LABEL_CODES = {v: k for k, v in LINE_LABEL_CODES.items()}

# TODO: ?? also allow these options to be specified by the user?
MERGE = True
TO_FORMAT = 'df-long'
PREDICT_BATCH_SIZE = 100

FILES_META_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'txt-files-meta.json')
with open(FILES_META_PATH, 'r') as f:
    FILES_META = json.load(f)


def parse_args(args: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-i', '--inpaths', nargs="+", required=True,
        help='path to input file to parse. If multiple files are given, randomly '
             'selects one. If directories are provided, randomly selects one PDF '
             'from within the directory.')
    parser.add_argument('-o', '--outpath', type=str, default=DEFAULT_OUTPATH,
        help="Path to output directory.")
    parser.add_argument('--filetype', choices=['txt', 'xml'], default=DEFAULT_FILETYPE,
        help='File type of input file(s).')
    args = parser.parse_args(args)
    return args


def main(inpaths: List[str],
         outpath: str = DEFAULT_OUTPATH,
         filetype: str = DEFAULT_FILETYPE,
         verbosity: int = 0) -> None:
    # reads in text file.
    for inpath in inpaths:
        if args.verbosity > 1:
            print(f'Parsing {input}...')
        with open(inpath, 'r') as f:
            text = f.readlines()
        # reads in file metadata.
        # TODO: ?? separate this logic into another script. It is too specific
        # to the Box.com files.
        with open(inpath.replace('.txt', '_meta.json'), 'r') as f:
            meta = json.load(f)
        year_str = meta['folder']['name']
        filename = meta['name']
        try:
            start_line = FILES_META[year_str][filename]['start_line']
            end_line = FILES_META[year_str][filename]['end_line']
            text = '\n'.join(text[start_line-1:end_line])
        except:
            warnings.warn(f'Start/end lines not found for "{year_str}/{filename}".')
            text = '\n'.join(text)
        if filetype == 'txt':
            parser = TxtParser(
                line_labeler='supervised',
                line_predict_kws={'builder_path': BUILDER_PATH,
                                'clf_path': CLASSIFIER_PATH,
                                'batch_size': PREDICT_BATCH_SIZE},
                line_label_codes=LINE_LABEL_CODES,
                speaker_parser='rule',
                verbosity=verbosity
            )
            parsed_transcripts = parser.parse_hansards(
                text=text,
                merge=MERGE,
                to_format=TO_FORMAT,
            )
        elif filetype == 'xml':
            raise NotImplementedError('TODO: add XmlParser call here.')
        else:
            raise NotImplementedError(f'Transcript not implemented for "{filetype}" file type.')
        out_fname = re.sub(r'\..{1,6}$', '', inpath.split('/')[-1])
        for i, (metadata, entries) in enumerate(parsed_transcripts):
            out_fname = f'{out_fname}_{i}.json'
            _save_transcript(entries, metadata, fname=out_fname, outpath=outpath, verbosity=verbosity)
    return None


def _save_transcript(entries: List[Entry],
                     metadata: Sitting,
                     fname: str,
                     outpath: str,
                     verbosity: int = 0) -> None:
    """saves parsed transcripts to disk.
    """
    metadata.date = metadata.date.isoformat() if metadata.date else None
    if isinstance(entries, list):
        entries_dump = [e.__dict__ for e in entries]
    else:
        entries_dump = entries.to_dict(orient='records')
    with open(os.path.join(outpath, fname), 'w') as f:
        json.dump({'meta': metadata.__dict__, 'entries': entries_dump}, f, indent=4)
    if verbosity > 1:
        print(f'Parsed transcript(s) were saved to {os.path.join(outpath, fname)}.')
    return None


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
