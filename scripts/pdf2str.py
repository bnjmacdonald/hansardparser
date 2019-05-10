"""converts a list of pdfs to text files.

This script has been used for:

    * generating text files to be decoded using t2t-decoder.

Example usage::

    $ python -m scripts.pdf2str --input \
        "/Users/bmacwell/Documents/current/projects/hansardlytics/data/raw/hansards/national-assembly/2005/AUGUST/2nd August, 2005P.pdf" \
        "/Users/bmacwell/Documents/current/projects/hansardlytics/data/raw/hansards/national-assembly/2005/AUGUST/10th August, 2005A.pdf" \
        --output data/temp

Example:

    This example retrieves a random PDF from a directory and converts it to a string.
    $ INPUT=`find /Users/bmacwell/Documents/current/projects/hansardlytics/data/raw/hansards/national-assembly -type f | grep .pdf | gshuf -n 1`
    $ echo $INPUT
    $ python -m scripts.pdf2str --input "$INPUT" --output data/temp

"""

import os
import re
import argparse
from hansardparser.plenaryparser.utils import pdf2str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    # print(args.input)
    for inp in args.input:
        assert inp.endswith('.pdf'), f'{inp} is not a PDF file.'
        text = pdf2str(inp)
        out_fname = re.sub(r'\.[A-z]{1,3}$', '.txt', inp.split('/')[-1])
        with open(os.path.join(args.output, out_fname), 'w') as f:
            f.write(text)
    return None

if __name__ == '__main__':
    main()
            
