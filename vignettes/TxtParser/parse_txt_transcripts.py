"""an example using the TxtParser to parse txt files of Kenyan Hansards.

Retrieves txt files from Box.com and parses them using the `TxtParser` class.

Usage::

    `python -m vignettes.TxtParser.parse_txt_transcripts`

Todos:

    TODO: add `argparse` arguments so that user can select which transcript to
        parse. Or maybe revise `parse_one_transcript` module so that it works
        with FlatWorld files too?
"""

import os
import json
import requests
import numpy as np
import subprocess
import warnings

from hansardparser.plenaryparser.TxtParser import TxtParser
from hansardparser import settings

# KLUDGE: for builder pickle imports.
from hansardparser.plenaryparser.TxtParser.LineLabeler.SupervisedLineLabeler import *  

INPATH = '/Users/bnjmacdonald/Documents/current/projects/hansardparser/data/tests/hansards/txt/1985/1985.txt'
OUTPATH = os.path.join(settings.DATA_ROOT, 'temp', 'plenaryparser', 'txt')
VERBOSITY = 1
MERGE = True
TO_FORMAT = 'df-long'
FOLDERS = ['1985', '1987', '1990', '1992']

PREDICT_BATCH_SIZE = 100

FILES_META_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'txt-files-meta.json')
with open(FILES_META_PATH, 'r') as f:
    FILES_META = json.load(f)

BUILDER_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'plenaryparser', 'text2vec', 'builders', '2018-07-15T084305')
CLASSIFIER_PATH = os.path.join(settings.EXPERIMENTS_ROOT, 'plenaryparser', 'line_classifier', 'classifier0')
LINE_LABEL_CODES_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'plenaryparser', 'text2vec', 'corpora', '2018-07-15T084305', 'label_codes.json')
with open(LINE_LABEL_CODES_PATH, 'r') as f:
    LINE_LABEL_CODES = json.load(f)
    assert len(set(LINE_LABEL_CODES.values())) == len(LINE_LABEL_CODES), \
        'values must be unique for dict to be reversed.'
    LINE_LABEL_CODES = {v: k for k, v in LINE_LABEL_CODES.items()}


def main():
    # reads in text file.
    with open(INPATH, 'r') as f:
        text = f.readlines()
    # reads in file metadata.
    with open(INPATH.replace('.txt', '_meta.json'), 'r') as f:
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
    parser = TxtParser(
        line_labeler='supervised',
        line_predict_kws={'builder_path': BUILDER_PATH, 'clf_path': CLASSIFIER_PATH, 'batch_size': PREDICT_BATCH_SIZE},
        line_label_codes=LINE_LABEL_CODES,
        speaker_parser='rule',
        verbosity=VERBOSITY
    )
    # self = parser
    parsed_transcripts = parser.parse_hansards(
        text=text,
        merge=MERGE,
        to_format=TO_FORMAT,
    )
    for i, (metadata, entries) in enumerate(parsed_transcripts):
        path = os.path.join(OUTPATH, meta['folder']['name'] + '_' + meta['name'].replace('.txt', '') + '_' + str(i))
        metadata.date = metadata.date.isoformat() if metadata.date else None
        if isinstance(entries, list):
            json_dump = {'meta': metadata.__dict__, 'entries': [e.__dict__ for e in entries]}
        else:
            json_dump = {'meta': metadata.__dict__, 'entries': entries.to_dict(orient='records')}
        with open(path + '.json', 'w') as f:
            json.dump(json_dump, f, indent=4)
        if VERBOSITY:
            print(f'Success! Parsed transcript(s) was saved to {path}')
    return None


if __name__ == '__main__':
    main()
