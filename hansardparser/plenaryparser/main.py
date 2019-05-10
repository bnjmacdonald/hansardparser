"""Implements a Flask app for parsing a Hansard transcript.


Python example (via Requests)::

    >>> import requests
    >>> import os
    >>> import json
    >>> import io
    >>> f = io.StringIO('''<Header>ORAL ANSWERS TO QUESTIONS </Header>\n
    >>> Question No.799\n
    >>> <Newspeech>MR. SPEAKER: Mr. Ekidoronot in? Next Question.</Newspeech>\n
    >>> Question No.780\n
    >>> DR. MANTO asked the Minister for Agriculture :-\n
    >>> (a)	 whether he is aware that the demand for sugar will be greater than its .production by 1990; and\n
    >>> (b) whether he will, therefore, reconsider\n
    >>> the suspended plan to establish an additional sugar factory in Busia District.
    >>> ''')
    >>> url = "http://localhost:8000"
    >>> res = requests.post(url, files={'file': f}, params={filetype': 'txt', 'line_type_labeler': 'supervised', 'line_speaker_span_labeler': 'hybrid'})
    >>> assert res.status_code == 200
    >>> data = json.loads(res.text)
    >>> print(len(data))
    >>> for transcript in data['parsed_transcripts']:
    >>>     print(transcript["meta"])
    >>>     print(transcript["entries"])
    

Todos:

    ...
"""

import os
import re
import json
import logging
import datetime
from typing import List
import pandas as pd
from flask import request, redirect, jsonify, Flask, abort

from TxtParser import TxtParser

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    assert 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ and 'hansardparser' in os.environ['GOOGLE_APPLICATION_CREDENTIALS'], \
        ('google application credentials are incorrect')
    if request.method == 'POST':
        verbosity = int(os.environ['VERBOSITY']) if 'VERBOSITY' in os.environ else 0
        start_line = request.args.get('start_line', default=0, type=int)
        end_line = request.args.get('end_line', default=None, type=int)
        filetype = request.args.get('filetype', default=None, type=str)
        line_type_labeler = request.args.get('line_type_labeler', default='rule', type=str)
        line_speaker_span_labeler = request.args.get('line_speaker_span_labeler', default='rule', type=str)
        try:
            assert len(request.files)
            f = request.files['file']
            # assert os.path.isfile(inpath), f'{inpath} is not a valid file.'
            parser = TxtParser(
                line_type_labeler=line_type_labeler,
                line_speaker_span_labeler=line_speaker_span_labeler,
                verbosity=verbosity
            )
            parsed_transcripts = parser.parse_hansards(
                filepath_or_buffer=f,
                start_line=start_line,
                end_line=end_line,
                filetype=filetype
            )
            result = []
            for i in range(len(parsed_transcripts)):
                d = {'parsed_transcripts': parsed_transcripts[i]}
                if verbosity > 0:
                    d.update({'sitting_texts': parser._sitting_texts[i],
                              'unmerged_parsed_transcripts': parser._unmerged_parsed_transcripts[i],
                              'line_type4_preds': parser._line_type4_preds[i],
                              'line_speaker_span_preds': parser._line_speaker_span_preds[i]})
                result.append(d)
            return jsonify(result)
        except Exception:
            logging.exception(f'Failed to parse and save transcript.')
            return abort(500)
    return ''


if __name__ == '__main__':
    app.run(host='0.0.0.0')
