
import os
import re
import requests
import json
import base64
import warnings
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from google.cloud import storage
import googleapiclient
from tensor2tensor.data_generators.generator_utils import to_example

from TxtParser.LineSpeakerSpanLabeler import Rule
from TxtParser.LineSpeakerSpanLabeler.t2t_problem.hansard_line_speaker_span import HansardLineSpeakerSpan, HansardLineSpeakerSpanChar
from TxtParser.LineSpeakerSpanLabeler.t2t_problem import config as config_speaker_span
from utils import extract_flatworld_tags, batchify

# load cached predictions.
USE_CACHED_PREDICTIONS = False
# only predict speaker span BIO labels for speeches.
LABEL_SPEECHES_ONLY = True

PROBLEM = 'hansard_line_speaker_span_char'
PROBLEM_CLASSES = {'hansard_line_speaker_span': HansardLineSpeakerSpan,
                   'hansard_line_speaker_span_char': HansardLineSpeakerSpanChar}

# number of instances to pass for online prediction per batch.
BATCH_SIZE = 10
MAX_LENGTH = 2048  # KLUDGE: how could I easily retrieve this from the exported model?
RM_FLATWORLD_TAGS = config_speaker_span.RM_FLATWORLD_TAGS
CONTEXT_N_LINES = config_speaker_span.CONTEXT_N_LINES

# send request to local server, rather than cloud server.
LOCAL = True
PORT = 8501
HOST = os.environ['HANSARD_LINE_SPEAKER_SPAN_HOST'] if 'HANSARD_LINE_SPEAKER_SPAN_HOST' in os.environ else PROBLEM
LOCAL_URL = f'http://{HOST}:{PORT}/v1/models/predict_{PROBLEM}:predict'

# path to vocabulary file (only used if problem.vocab_type != 'character')
DATA_DIR = f'/Users/bmacwell/Documents/current/projects/hansardparser/data/generated/plenaryparser/t2t_data/{PROBLEM}'


# model name on google cloud.
MODEL = ""
# model version
VERSION = None

# google cloud project details.
PROJECT = "hansardparser"
BUCKET_NAME = 'gs://hansarparser-data'
BUCKET_PREFIX = 't2t_preds/hansard_line_speaker_span_predict'

# Google Cloud Function URL
# GCF_URL = 'https://us-central1-hansardparser.cloudfunctions.net/predict_hansard_line_speaker_span'


class Supervised(object):
    """Invokes a trained classifier to extract the speaker name from a line in a
    Hansard transcript.

    Attributes:

        cache_filename: str. Name of cached file where predictions are located.

    """
    
    def __init__(self, cache_filename: str = None, verbosity: int = 0):
        self.cache_filename = cache_filename
        self.verbosity = verbosity
        # KLUDGE: for temporary use to assist where superivsed speaker parser needs
        # help.
        self.RuleLineSpeakerSpanLabeler = Rule(verbosity=verbosity)


    def label_speaker_spans(self,
                            lines: List[str],
                            types: List[str] = None) -> List[str]:
        """Labels the speaker span in each line using BIO tagging.

        Arguments:

            lines: List[str]. List of lines to be assigned BIO tags.

            types: List[str] = None. List of line types. Only used if
                LABEL_SPEECHES_ONLY==True. If LABEL_SPEECHES_ONLY==True, then
                the speaker span is tagged only for lines with type == 'speech'.
        

        Returns:

            preds: List[str]. BIO prediction for each line. e.g. ["BIIIIIIO", "OOOOOOOO", ...]
        """
        assert isinstance(lines[0], str), 'Each item in `lines` must be a string.'
        if RM_FLATWORLD_TAGS:
            raise NotImplementedError
        preds = self._get_predictions(lines, types=types)
        # self._line_speaker_span_preds = list(zip(lines, pred_labels))
        # picks out the speaker name from the text in each line.
        return preds


    def extract_speaker_names(self, lines: List[str], preds: List[str]) -> Tuple[List[str], List[str]]:
        speaker_names = []
        texts = []
        for i, line in enumerate(lines):
            speaker_name, text = self.RuleLineSpeakerSpanLabeler._extract_speaker_name(line, preds[i])
            speaker_names.append(speaker_name)
            texts.append(text)
        return speaker_names, texts


    def _get_predictions(self, lines: List[str], types: List[str] = None) -> List[int]:
        """retrieves line predictions.
        """
        if USE_CACHED_PREDICTIONS:
            preds = self._get_cached_predictions(lines)
        else:
            preds = self._get_online_predictions(lines, types=types)
        assert len(preds) == len(lines), 'Number of predictions should equal ' \
            f'number of transcript lines, but {len(preds)} != {len(lines)}'
        return preds


    def _get_online_predictions(self, lines: List[str], types: List[str] = None) -> List[int]:
        """retrieves predictions by triggering google cloud function, which
        invokes ml-engine to make a prediction for each line.
        """
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
        if self.verbosity > 1:
            raw_instances = instances.copy()
        if LABEL_SPEECHES_ONLY:
            assert types is not None, '`types` must be provided when LABEL_SPEECHES_ONLY == True.'
            assert len(types) == len(lines), f'types must have same length as lines, but {len(types)} != {len(lines)}.'
            speeches = []
            speeches_idx = []
            for i, instance in enumerate(instances):
                if types[i] == 'speech':
                    speeches.append(instance)
                    speeches_idx.append(i)
            instances = speeches
        if self.verbosity > 0:
            print(f'Making "speaker span" predictions for {len(instances)} lines...')
        problem_class = PROBLEM_CLASSES[PROBLEM]
        problem = problem_class()
        encoders = problem.feature_encoders(data_dir=DATA_DIR)
        instances_b64 = []
        for instance in instances:
            if 'targets' not in instance:
                instance['targets'] = ''
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
            try:
                # print([len(inst['inputs']) + len(inst['context']) for inst in raw_instances[len(preds):len(preds)+BATCH_SIZE]])
                if LOCAL:
                    res = requests.post(LOCAL_URL, data=json.dumps({"instances": batch}),
                        headers={"Content-Type": "application/json"})
                else:
                    res = self._get_cloud_predictions(project=PROJECT, model=MODEL, instances=batch, version=VERSION)
                assert res.ok, f'request failed. Reason: {res.reason}.'
                predictions = json.loads(res.content)
                predictions = predictions['predictions']
                for i, pred in enumerate(predictions):
                    pred_out = pred['outputs']
                    pred_out = encoders['targets'].decode(pred_out)
                    # removes spaces.
                    pred_out = re.sub(r'\s+', '', pred_out)
                    try:
                        eos_idx = pred_out.lower().index('<eos>')
                        pred_out = pred_out[:eos_idx]
                    except ValueError:
                        if self.verbosity > 1:
                            warnings.warn(f'<eos> not found in prediction: {pred_out}')
                    preds.append(pred_out)
                    # preds.append([token_pred[0][0] for token_pred in pred['outputs']])
            except AssertionError as e:
                print(e)
                for i in range(len(batch)):
                    preds.append(None)
        if LABEL_SPEECHES_ONLY:
            preds_all_lines = []
            for i, line in enumerate(lines):
                pred = 'O' * len(line)
                preds_all_lines.append(pred)
            n_preds = 0
            assert len(speeches_idx) == len(preds)
            for i, idx in enumerate(speeches_idx):
                preds_all_lines[idx] = preds[i]
                n_preds += 1
            # sanity check.
            assert n_preds == len(preds)
            preds = preds_all_lines
        if self.verbosity > 1:
            for i, pred in enumerate(preds):
                instance = raw_instances[i]
                if 'targets' in instance:
                    instance.pop('targets')
                if 'label' in instance:
                    instance.pop('label')
                print(f'INPUT (len={len(instance["inputs"])}): {instance}\nOUTPUT (len={len(pred) if pred is not None else None}): {pred}')
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
