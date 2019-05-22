
import os
import json
import requests
import base64
from typing import List, Optional
from tqdm import tqdm
import numpy as np
from google.cloud import storage
import googleapiclient
from tensor2tensor.data_generators.generator_utils import to_example

from utils import extract_flatworld_tags, batchify
from TxtParser.LineTypeLabeler.t2t_problem.config import LINE_LABEL_CODES, RM_FLATWORLD_TAGS, CONTEXT_N_LINES
from TxtParser.LineTypeLabeler.t2t_problem.hansard_line_type4 import HansardLineType4, HansardLineType4Char
from TxtParser.LineTypeLabeler import Rule

# load cached predictions.
USE_CACHED_PREDICTIONS = False

PROBLEM = 'hansard_line_type4_char'
PROBLEM_CLASSES = {'hansard_line_type4': HansardLineType4, 'hansard_line_type4_char': HansardLineType4Char}

LINE_CODE2LABEL = {code: label for label, code in LINE_LABEL_CODES.items()}

# number of instances to pass for online prediction per batch.
BATCH_SIZE = 50

# send request to local server, rather than cloud server.
LOCAL = True
PORT = 8501
MAX_LENGTH = 2048  # KLUDGE: how could I easily retrieve this from the exported model?
HOST = os.environ['HANSARD_LINE_TYPE4_HOST'] if 'HANSARD_LINE_TYPE4_HOST' in os.environ else PROBLEM
LOCAL_URL = f'http://{HOST}:{PORT}/v1/models/predict_{PROBLEM}:predict'

# path to vocabulary file (only used if problem.vocab_type != 'character')
DATA_DIR = f'/Users/bmacwell/Documents/current/projects/hansardparser/data/generated/plenaryparser/t2t_data/{PROBLEM}'
# model name
MODEL = "hansard_line_type4_lstm_seq2seq_attention_bidirectional_encoder_lstm_attention"
# model version
VERSION = None

# google cloud project details.
PROJECT = "hansardparser"
BUCKET_NAME = 'hansardparser-data'
BUCKET_PREFIX = 't2t_preds/hansard_line_type4_predict'

# Google Cloud Function URL
# GCF_URL = 'https://us-central1-hansardparser.cloudfunctions.net/predict_hansard_line_type4'



class Supervised(object):
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
        self.RuleLineTypeLabeler = Rule(verbosity=verbosity)


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
        if RM_FLATWORLD_TAGS:
            line_texts = []
            for line in lines:
                line_text, _ = extract_flatworld_tags(line)
                line_texts.append(line_text)
        else:
            line_texts = lines
        preds = self._get_predictions(line_texts)
        labels = [LINE_CODE2LABEL[l] for l in preds]
        # retrieves header type ('header', 'subheader', or 'subsubheader').
        for i, line in enumerate(line_texts):
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
        # constructs instances for predictions
        contexts = self._get_line_context(lines, n=CONTEXT_N_LINES)
        instances = []
        for i, line in enumerate(lines):
            context = contexts[i]
            if MAX_LENGTH > 0:
                if len(line) > MAX_LENGTH:
                    line = line[:MAX_LENGTH]
                    context = ''
                elif (len(line) + len(context)) > MAX_LENGTH:
                    context = context[:MAX_LENGTH-len(line)]
                assert (len(line) + len(context)) <= MAX_LENGTH
            instances.append({'inputs': line, 'context': context})
        if self.verbosity > 0:
            print(f'making "line type" predictions for {len(instances)} lines...')
        if self.verbosity > 1:
            raw_instances = instances.copy()
        problem_class = PROBLEM_CLASSES[PROBLEM]
        problem = problem_class()
        encoders = problem.feature_encoders(data_dir=DATA_DIR)
        instances_b64 = []
        for instance in instances:
            if 'label' not in instance:
                instance['label'] = 0
            encoded_instance = problem.encode_example(instance, encoders)
            # encoded_sample.pop('targets')
            # encoded_sample.pop('context')
            serialized_instance = to_example(encoded_instance).SerializeToString()
            instances_b64.append({"b64": base64.b64encode(serialized_instance).decode('utf-8')})
        instances = instances_b64
        preds = []
        batch_generator = batchify(instances, BATCH_SIZE)
        if self.verbosity > 0:
            batch_generator = tqdm(batch_generator, total=np.ceil(len(instances)/BATCH_SIZE).astype(int))
        for batch in batch_generator:
            if LOCAL:
                res = requests.post(LOCAL_URL, data=json.dumps({"instances": batch}),
                    headers={"Content-Type": "application/json"})
            else:
                res = self._get_cloud_predictions(project=PROJECT, model=MODEL, instances=batch, version=VERSION)
            assert res.ok, f'request failed. Reason: {res.reason}.'
            predictions = json.loads(res.content)
            predictions = predictions['predictions']
            for i, pred in enumerate(predictions):
                preds.append(pred['outputs'][0][0][0])
        if self.verbosity > 1:
            for i, pred in enumerate(preds):
                instance = raw_instances[i]
                if 'label' in instance:
                    instance.pop('label')
                print(f'INPUT: {instance}\nOUTPUT: {pred}')
        return preds


    def _get_cloud_predictions(self, project, model, instances, version=None):
        """Send json data to a deployed model for prediction.

        Args:

            project (str): project where the AI Platform Model is deployed.
            model (str): model name.
            instances ([Mapping[str: Any]]): Keys should be the names of Tensors
                your deployed model expects as inputs. Values should be datatypes
                convertible to Tensors, or (potentially nested) lists of datatypes
                convertible to tensors.
            version: str, version of the model to target.

        Returns:

            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        # Create the AI Platform service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
        service = googleapiclient.discovery.build('ml', 'v1')
        name = f'projects/{project}/models/{model}'
        if version is not None:
            name += f'/versions/{version}'
        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response


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
        if self.RuleLineTypeLabeler._is_header(line_text, flatworld_tags):
            header_type = 'header'
        elif self.RuleLineTypeLabeler._is_subsubheader(line_text, flatworld_tags):
            header_type = 'subsubheader'
        return header_type


    def _get_line_context(self, lines: List[str], n: int = 1) -> List[str]:
        """retrieves context for each line.
        """
        contexts = []
        for i in range(len(lines)):
            prev_text = '\n'.join(lines[i-n:i])
            next_text = '\n'.join(lines[i+1:i+1+n])
            # line['prev_context'] = prev_line_text
            # line['next_context'] = next_line_text
            context = '\n'.join([prev_text, next_text])
            contexts.append(context)
        return contexts
