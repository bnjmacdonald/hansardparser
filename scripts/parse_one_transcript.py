"""parses a single transcript.

This script is useful for debugging and manual inspection.

Usage
-----

Parse a single transcript::
    
    python -m api.parse_one_transcript -i '../data/raw/transcripts/Hansard_Report_-_Wednesday__18th_November_2015A.pdf'
    python -m api.parse_one_transcript -i "/Volumes/Transcend/HANSARDS/2000/July/19th July, 2000P.pdf"

Randomly select a single transcript::

    python -m api.parse_one_transcript -i /Volumes/Transcend/HANSARDS/{1998..2013} ../data/raw/transcripts

"""

import os
import random
import subprocess
import argparse

import settings
from hansardparser.plenaryparser.XmlParser import XmlParser
from hansardparser.plenaryparser.utils import get_file_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        dest="input",
        nargs="+",
        required=True,
        help="path to input file to parse. If multiple files are given, randomly selects one. If directories are provided, randomly selects one PDF from within the directory."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Config(args)
    # parses transcript.
    print(config.file_path)
    parser = XmlParser(verbose=config.verbose)
    metadata, contents = parser.process_transcript(
        file_path=config.file_path,
        save_soup=config.save_soup,
        path=config.out_path,
        rm_whitespace=True,
        append_meta=True,
        to_format='df-long',
    )
    lines = []
    file_info = 'File: {0}\nNumber of entries: {1}\n\n'.format(config.file_path, contents.shape[0])
    file_info += ''.join(['{0:12s}: {1}\n'.format(k, v) for k, v in metadata.__dict__.items()]) + '\n'
    lines.append(file_info)
    # creates file containing results.
    for index, row in contents.iterrows():
        line = '-'*30 + '\n'
        line += 'position: {0}\n'.format(row['position'])
        line += 'entry_type: {0}\n'.format(row['entry_type'])
        line += 'header: {0}\n'.format(row['header'])
        line += 'subheader: {0}\n'.format(row['subheader'])
        line += 'subsubheader: {0}\n'.format(row['subsubheader'])
        line += 'page_num: {0}\n'.format(row['page_number'])
        line += 'speaker: {0}\n'.format(row['speaker'])
        line += 'speaker_cleaned: {0}\n'.format(row['speaker_cleaned'])
        line += 'appointment: {0}\n'.format(row['appointment'])
        line += 'text: {0}\n\n'.format(row['text'])
        lines.append(line)

    assert len(lines) == (contents.shape[0] + 1)
    # path = get_path('query', creation_date=config.date, temp=True)
    fname = 'hansard.txt'
    with open(os.path.join(config.out_path, fname), 'w') as f:
        f.writelines(lines)
    # opens PDF, soup, and parsed soup.
    command = 'sublime -n "{0}"'.format(os.path.join(config.out_path, config.file_path.split('/')[-1].replace('.pdf', '.xml')))
    _ = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    command = 'sublime -n "{0}"'.format(os.path.join(config.out_path, fname))
    _ = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    command = 'open -a preview "{0}"'.format(config.file_path)
    _ = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # contents.to_csv(os.path.join(config.out_path, 'temp.csv'))

class Config(object):
    def __init__(self, args):
        # year = random.choice(range(1997,2013))
        # base_dir = '/Volumes/Transcend/HANSARDS'  # /Users/mounted
        # file_path = os.path.join(settings.BASE_DIR, 'data', 'raw', 'transcripts', 'Hansard_Report_-_Wednesday__18th_November_2015A.pdf')
        file_paths = []
        for path in args.input:
            # print(path)
            if os.path.isfile(path):
                file_paths.append(path)
            else:
                these_file_paths = get_file_paths(path, verbose=0)
                file_paths.extend(these_file_paths)
        self.file_path = random.choice(file_paths)
        self.verbose = 1
        self.save_soup = True
        self.out_path = os.path.join(settings.DATA_ROOT, 'temp', 'hansardparser')

if __name__ == '__main__':
    main()
