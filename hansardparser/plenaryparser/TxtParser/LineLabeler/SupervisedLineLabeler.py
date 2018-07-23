
import json
import requests
from typing import List, Optional
from google.cloud import storage

from utils import extract_flatworld_tags
from TxtParser.LineLabeler import RuleLineLabeler

RM_FLATWORLD_TAGS = True
USE_CACHED_PREDICTIONS = False
BUCKET_NAME = 'hansardparser-data'
BUCKET_PREFIX = 't2t_preds/hansard_line_type4_predict'

# FIXME: don't hard code these label codes.
LABEL_CODES = {0: "header", 1: "speech", 2: "scene", 3: "garbage"}

# Google Cloud Function URL
GCF_URL = 'https://us-central1-hansardparser.cloudfunctions.net/predictHansardLineType4'


class SupervisedLineLabeler(object):
    """Invokes a trained classifier to assign a label to each line of a Hansard
    transcript.

    Label is one of: [header, subheader, subsubheader, speech, scene, garbage].

    Attributes:

        cache_filename: str. Name of cached file where predictions are located.
    """

    def __init__(self, cache_filename: str = None, verbosity: int = 0):
        self.cache_filename = cache_filename
        self.verbosity = verbosity
        # KLUDGE: for temporary use to assist where supervised line labeler needs
        # help.
        self.RuleLineLabeler = RuleLineLabeler(verbosity=verbosity)


    def label_lines(self, lines: List[str]) -> List[str]:
        """Returns the label/class of each line in the Hansard transcript.

        Possible labels: header, subheader, subsubheader, speech, scene, garbage.

        Arguments:

            lines: List[str]. List of lines in a Hansard transcript.

        Returns:

            labels: List[str]. List of line labels.
        """
        assert isinstance(lines[0], str), 'Each item in `lines` must be a string.'
        # KLUDGE: removes flatworld tag from text before making prediction.
        # TODO: this logic should happen in the tensorflow preprocessing.
        line_texts = []
        for line in lines:
            if RM_FLATWORLD_TAGS:
                line_text, _ = extract_flatworld_tags(line)
            line_texts.append(line_text)
        preds = self._get_predictions(line_texts)
        labels = [LABEL_CODES[l] for l in preds]
        # retrieves header type ('header', 'subheader', or 'subsubheader').
        for i, line in enumerate(lines):
            if labels[i] == 'header':
                labels[i] = self._get_header_type(line)
                # print(labels[i], line)
        return labels


    def _get_predictions(self, lines: List[str]) -> List[int]:
        """retrieves line predictions.
        """
        if USE_CACHED_PREDICTIONS:
            preds = self._get_cached_predictions(lines)
        else:
            preds = self._get_online_predictions(lines)
        assert len(preds) == len(lines), 'Number of predictions should equal ' \
            f'number of transcript lines, but {len(preds)} != {len(lines)}'
        return preds


    def _get_online_predictions(self, lines: List[str]) -> List[int]:
        """retrieves predictions by triggering google cloud function, which
        invokes ml-engine to make a prediction for each line.
        """
        res = requests.post(GCF_URL, data=json.dumps({"instances": lines}),
            headers={"Content-Type": "application/json"})
        assert res.status_code == 200
        preds = []
        for pred in json.loads(res.content):
            preds.append(pred['outputs'][0][0][0])
        return preds


    def _get_cached_predictions(self, lines: List[str]) -> List[int]:
        """retrieves predictions from google cloud storage.

        Todos:

            TODO: ensure that predictions will be in same order as lines.
        """
        # https://cloud.google.com/storage/docs/listing-objects
        # client = discovery.build('storage', 'v1')
        storage_client = storage.Client()
        prefix = f'{BUCKET_PREFIX}/{self.cache_filename}'
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if 'errors' not in blob.name:
                s = b'[' + blob.download_as_string().replace(b'\n', b',')[:-1] + b']'
                preds_json = json.loads(s)
                preds = []
                for pred in preds_json:
                    preds.append(pred['outputs'][0][0][0])
        return preds


    def _get_header_type(self, line: str):
        """Retrieves the header type of a header.

        Possible header types: ['header', 'subheader', 'subsubheader'].

        This method should only be called on lines that you are confident are headers.
        The method returns "header" if `self._is_header` returns true; else returns
        "subsubheader" if `self._is_subsubheader` returns true; else returns
        "subheader".

        Arguments:

            line: str. Line in a Hansard transcript.

        Returns:

            str. One of: ['header', 'subheader', 'subsubheader']. By default,
                returns 'subheader' if the criteria for 'header' or 'subsubheader'
                are not met.
        """
        line_text, flatworld_tags = extract_flatworld_tags(line)
        header_type = 'subheader'
        if self.RuleLineLabeler._is_header(line_text, flatworld_tags):
            header_type = 'header'
        elif self.RuleLineLabeler._is_subsubheader(line_text, flatworld_tags):
            header_type = 'subsubheader'
        return header_type
