"""Implements a command-line application for parsing a corpus of transcripts.
Retrieves txt files from Box.com and parses them using the `TxtParser` class.
Usage::
    Parse a single txt transcript::
        
        python scripts/parse_and_save_transcripts.py -v 1 \
            --inpaths /Users/bnjmacdonald/Documents/current/projects/hansardparser/data/tests/hansards/txt/1985/1985.txt \
            --outpath generated/plenaryparser/parsed \
            --gcs
Todos:
    TODO: refactor in combination with `parse_one_transcript` so that a single
        module exposes an API that can be used for any transcript.
"""

import os
import re
import json
import argparse
import requests
from typing import List
import numpy as np
from google.cloud import storage


PARSER_URL = 'https://hansardparser.appspot.com'
PARAMS = {'line_labeler': 'rule'}

# config for saving to google cloud storage.
BUCKET_NAME = 'hansardparser-data'

# FILES_META_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'txt-files-meta.json')
# with open(FILES_META_PATH, 'r') as f:
#     FILES_META = json.load(f)


def parse_args(args: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-i', '--inpaths', nargs="+", required=True,
        help='path to input files to parse.')
    parser.add_argument('-o', '--outpath', type=str, help="Path to output directory.")
    parser.add_argument('--gcs', action='store_true', default=False,
        help='Save parsed transcript(s) to google cloud storage.')
    args = parser.parse_args(args)
    return args


def main(inpaths: List[str],
         outpath: str,
         gcs: bool,
         verbosity: int = 0) -> None:
    # reads in text file.
    for inpath in inpaths:
        if args.verbosity > 1:
            print(f'Parsing {input}...')
        with open(inpath, 'r') as f:
            unparsed_transcript = f.read()
            unparsed_transcript = unparsed_transcript[:50000]
        # reads in file metadata.
        # TODO: read file metadata.
        # with open(inpath.replace('.txt', '_meta.json'), 'r') as f:
        #     meta = json.load(f)
        # year_str = meta['folder']['name']
        # filename = meta['name']
        # try:
        #     params['start_line'] = FILES_META[year_str][filename]['start_line']
        #     params['end_line'] = FILES_META[year_str][filename]['end_line']
        # except:
        #     warnings.warn(f'Start/end lines not found for "{year_str}/{filename}".')
        #     text = '\n'.join(text)
        # retrieves parsed transcripts from within the file.
        headers = {'Content-Type': 'application/json'}
        if verbosity > 0:
            print(f'Parsing {inpath}...')
        resp = requests.post(PARSER_URL, params=PARAMS, headers=headers,
            data=json.dumps({'transcript': unparsed_transcript}))
        if resp.status_code != 200:
            raise requests.HTTPError('failed to parse transcript. '
                f'Reason: {resp.reason}. Response content: {resp.content}')
        parsed_transcripts = json.loads(resp.content)
        out_fname = re.sub(r'\..{1,6}$', '', inpath.split('/')[-1])
        for i, parsed_transcript in enumerate(parsed_transcripts):
            out_fname = f'{out_fname}_{i}.json'
            _save_parsed_transcript(parsed_transcript,
                path=os.path.join(outpath, out_fname), gcs=gcs, verbosity=verbosity)
    return None


def _save_parsed_transcript(parsed_transcript: dict,
                            path: str,
                            gcs: bool,
                            verbosity: int = 0) -> None:
    """saves a parsed transcript to disk or to google cloud storage.
    """
    if gcs:
        # saves parsed transcript to google cloud storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(path)
        blob.upload_from_string(
            json.dumps(parsed_transcript, indent=4),
            content_type='application/json'
        )
        if verbosity > 0:
            print(f'A parsed transcript was saved to gs://{os.path.join(BUCKET_NAME, path)}.')
    else:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            json.dump(parsed_transcript, f, indent=4)
        if verbosity > 0:
            print(f'A parsed transcript was saved to {path}.')
    return None


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
