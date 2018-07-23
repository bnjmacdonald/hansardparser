"""Implements a simple Flask app for parsing a Hansard transcript.

Body content of PUT: file to be parsed.

Python example (via Requests)::

    >>> import os
    >>> import requests
    >>> import json
    >>> text = '''<Header>ORAL ANSWERS TO QUESTIONS </Header>\n
    >>> Question No.799\n
    >>> <Newspeech>MR. SPEAKER: Mr. Ekidoronot in? Next Question.</Newspeech>\n
    >>> Question No.780\n
    >>> DR. MANTO asked the Minister for Agriculture :-\n
    >>> (a)	 whether he is aware that the demand for sugar will be greater than its .production by 1990; and\n
    >>> (b) whether he will, therefore, reconsider\n
    >>> the suspended plan to establish an additional sugar factory in Busia District.
    >>> '''
    >>> url = "http://localhost:5000"
    >>> res = requests.post(url, data=json.dumps({"transcript": text}), headers={"Content-Type": "application/json"})
    >>> assert res.status_code == 200
    >>> data = json.loads(res.text)
    >>> print(len(data))
    >>> for transcript in data:
    >>>     print(transcript["entries"])
    >>>     print(transcript["meta"])

Curl Example::
    
    URL=https://hansardparser.appspot.com
    URL=https://localhost:5000/
    curl -d '{"transcript": "Question No. 238\nMr. Speaker: I believe Mr. X has"}' -H "Content-Type: application/json" -X POST $URL

Todos:

    FIXME: times out with large transcripts. Re-implement so that it saves the 
        parsed transcript to disk in the background and returns a 200 OK response
        without the parsed transcript.

    TODO: ?? replace query params with body params ??

    TODO: finish writing dockerfile for this app.
"""

import chardet
from typing import List
from flask import request, redirect, jsonify, Flask

from TxtParser import TxtParser

app = Flask(__name__)

MERGE = True
TO_FORMAT = 'df-long'


@app.route('/', methods=['GET', 'POST'])
def main():
    filetype = request.args.get('filetype', default='txt', type=str)
    start_line = request.args.get('start_line', default=1, type=int)
    end_line = request.args.get('end_line', default=None, type=int)
    line_labeler = request.args.get('line_labeler', default='rule', type=str)
    speaker_parser = request.args.get('speaker_parser', default='rule', type=str)
    if request.method == 'POST':
        data = request.get_json()
        text = data['transcript']
        parsed_transcripts = parse_file(text, filetype=filetype,
            start_line=start_line-1, end_line=end_line, line_labeler=line_labeler,
            speaker_parser=speaker_parser, verbosity=1)
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
               line_labeler: str = 'rule',
               speaker_parser: str = 'rule',
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
        print(line_labeler)
        parser = TxtParser(
            line_labeler=line_labeler,
            speaker_parser=speaker_parser,
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
