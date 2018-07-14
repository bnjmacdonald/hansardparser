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


INPATH = '/Users/bnjmacdonald/Documents/current/projects/hansardparser/data/tests/hansards/txt/1985/8th Oct 1985 - 11th dec 1985.txt'
OUTPATH = os.path.join(settings.DATA_ROOT, 'temp', 'plenaryparser', 'txt')
VERBOSITY = 1
MERGE = True
TO_FORMAT = 'df-long'
FOLDERS = ['1985', '1987', '1990', '1992']

FILES_META_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'txt-files-meta.json')
with open(FILES_META_PATH, 'r') as f:
    FILES_META = json.load(f)


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
        text = '\n'.join(text[start_line:end_line+1])
    except:
        warnings.warn(f'Start/end lines not found for "{year_str}/{filename}".')
        text = '\n'.join(text)
    parser = TxtParser(verbose=VERBOSITY)
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
