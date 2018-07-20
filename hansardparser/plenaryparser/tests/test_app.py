

import os
import json
import chardet
import unittest
import requests

from hansardparser.plenaryparser.app import app
from hansardparser import settings


class AppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_get_status_code(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)
    

    def test_home_post_status_code(self):
        fpath = os.path.join(settings.DATA_ROOT, 'tests', 'raw', 'transcripts', '8th - 9th Dec 1987.txt')
        with open(fpath, 'r') as f:
            transcript = f.readlines()
            transcript = '\n'.join(transcript[0:100])
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding)
        result = self.app.post('/', data=json.dumps({"transcript": transcript}), content_type='application/json')
        self.assertEqual(result.status_code, 200)


if __name__ == '__main__':
    unittest.main()
