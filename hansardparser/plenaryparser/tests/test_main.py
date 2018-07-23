

import os
import json
import chardet
import unittest
import requests

from hansardparser.plenaryparser.main import app
from hansardparser import settings


class AppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        fpath = os.path.join(settings.DATA_ROOT, 'tests', 'raw', 'transcripts', '8th - 9th Dec 1987.txt')
        with open(fpath, 'r') as f:
            transcript = f.readlines()
            transcript = '\n'.join(transcript[0:20])
            if isinstance(transcript, bytes):
                encoding = chardet.detect(transcript)['encoding']
                transcript = transcript.decode(encoding)
            self.transcript = transcript


    def test_get_status_code(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    

    def test_post_status_code(self):
        response = self.app.post('/', data=json.dumps({"transcript": self.transcript}),
            content_type='application/json')
        self.assertEqual(response.status_code, 200)
    

    def test_post_supervised_line_labeler_return(self):
        """tests that POST request with `line_labeler=supervised` returns
        labeled lines.
        """
        resp = self.app.post('/?line_labeler=supervised',
            data=json.dumps({"transcript": self.transcript}),
            content_type='application/json')
        # test 1: response status code should equal 200.
        self.assertEqual(resp.status_code, 200)
        # test 2: response data should be a string or bytes.
        self.assertTrue(isinstance(resp.data, str) or isinstance(resp.data, bytes))
        data = json.loads(resp.data)
        # test 3: there should be only one transcript in the resp data.
        self.assertEqual(len(data), 1)
        # test 4: the transcript should only have "meta" and "entries" keys.
        self.assertTrue('meta' in data[0].keys())
        self.assertTrue('entries' in data[0].keys())
        self.assertEqual(len(data[0].keys()), 2)
        # test 5: there should be more than 3 entries in the transcript.
        # TODO: make this test more specific to the exact number of entries expected.
        self.assertGreater(len(data[0]['entries']), 3)
    

    def test_post_supervised_speaker_parser_return(self):
        """tests that POST request with `speaker_parser=supervised` returns
        labeled lines.
        """
        resp = self.app.post('/?speaker_parser=supervised',
            data=json.dumps({"transcript": self.transcript}),
            content_type='application/json')
        # test 1: response status code should equal 200.
        self.assertEqual(resp.status_code, 200)
        # test 2: response data should be a string or bytes.
        self.assertTrue(isinstance(resp.data, str) or isinstance(resp.data, bytes))
        data = json.loads(resp.data)
        # test 3: there should be only one transcript in the resp data.
        self.assertEqual(len(data), 1)
        # test 4: the transcript should only have "meta" and "entries" keys.
        self.assertTrue('meta' in data[0].keys())
        self.assertTrue('entries' in data[0].keys())
        self.assertEqual(len(data[0].keys()), 2)
        # test 5: there should be more than 3 entries in the transcript.
        # TODO: make this test more specific to the exact number of entries expected.
        self.assertGreater(len(data[0]['entries']), 3)


if __name__ == '__main__':
    unittest.main()
