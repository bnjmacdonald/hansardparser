"""Implements a simple Flask app for parsing a Hansard transcript.

Body content of PUT: file to be parsed.

Example::

    TRANSCRIPT=$(cat "data/tests/raw/transcripts/8th - 9th Dec 1987.txt") 
    curl -d '{"transcript": "$TRANSCRIPT"}' -H "Content-Type: application/json" -X POST http://localhost:5000?start_line=1
"""

import chardet
from typing import List
from flask import request, redirect, jsonify, Flask

from hansardparser.plenaryparser.TxtParser import TxtParser

app = Flask(__name__)

MERGE = True
TO_FORMAT = 'df-long'
PREDICT_BATCH_SIZE = 100


@app.route('/', methods=['GET', 'POST'])
def main():
    filetype = request.args.get('filetype', default='txt', type=str)
    start_line = request.args.get('start_line', default=1, type=int)
    end_line = request.args.get('end_line', default=None, type=int)
    if request.method == 'POST':
        data = request.get_json()
        text = data['transcript']
        parsed_transcripts = parse_file(text, filetype=filetype,
            start_line=start_line-1, end_line=end_line, verbosity=1)
        json_transcripts = []
        for metadata, entries in parsed_transcripts:
            transcript = {'meta': metadata.__dict__, 'entries': entries.to_dict(orient='records')}
            json_transcripts.append(transcript)
        return jsonify(json_transcripts)
    return ''


def parse_file(text: List[str],
               filetype: str = 'txt',
               start_line: int = 0,
               end_line: int = None,
               verbosity: int = 0):
    if start_line > 0:
        text = text.split('\n')
        if end_line is None:
            text = '\n'.join(text[start_line-1:])
        else:
            text = '\n'.join(text[start_line-1:end_line])
    if isinstance(text, bytes):
        encoding = chardet.detect(text)['encoding']
        text = text.decode(encoding)
    if filetype == 'txt':
        parser = TxtParser(
            line_labeler='rule',
            # line_predict_kws={'builder_path': BUILDER_PATH,
            #                 'clf_path': CLASSIFIER_PATH,
            #                 'batch_size': PREDICT_BATCH_SIZE},
            # line_label_codes=LINE_LABEL_CODES,
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
    return parsed_transcripts


if __name__ == '__main__':
    app.run()
