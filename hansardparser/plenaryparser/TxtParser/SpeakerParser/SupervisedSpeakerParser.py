
from typing import List, Tuple
from utils import extract_flatworld_tags

from TxtParser.SpeakerParser import RuleSpeakerParser

# FIXME: don't hard-code these labels.
LABEL_CODES = {0: 'I', 1: 'O', 2: 'B'}

RM_FLATWORLD_TAGS = True
USE_CACHED_PREDICTIONS = False
BUCKET_NAME = 'gs://hansarparser-data'
BUCKET_PREFIX = 't2t_preds/hansard_line_has_speaker_predict'

# Google Cloud Function URL
GCF_URL = 'https://us-central1-hansardparser.cloudfunctions.net/predictHansardLineHasSpeaker'


class SupervisedSpeakerParser(object):
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
        self.RuleSpeakerParser = RuleSpeakerParser(verbosity=verbosity)


    def extract_speaker_names(self,
                              lines: List[str],
                              labels: List[str]
                              ) -> Tuple[List[str],
                                         List[Tuple[str, str, str]],
                                         List[str]]:
        """Extracts the speaker name from the beginning of each line.

        Extracts the speaker name from the beginning of each line. Only
        extracts speaker names where `label[i] == 'speech'`.

        Returns:

            speaker_names, texts: Tuple[List[str], List[str]].

                speaker_names: List[str]. List of speaker names, of same length
                    as `labels`. If `label[i] != 'speech'` or no speaker name
                    is found, then `speaker_name[i] = None`.

                parsed_speaker_names: List[Tuple[str, str, str]]. List of parsed
                    names. If an input speaker name is None, the parsed name will
                    be `(None, None, None)`.

                texts: List[str]. Lines of text after speaker name has been
                    extracted.
        """
        assert isinstance(lines[0], str), 'Each item in `lines` must be a string.'
        # KLUDGE: removes flatworld tag from text before making prediction.
        # TODO: this logic should happen in the tensorflow preprocessing.
        line_texts = []
        line_nums = []
        for i, line in enumerate(lines):
            if labels[i] == 'speech':
                line_text, _ = extract_flatworld_tags(line)
                line_texts.append(line_text)
                line_nums.append(i)
        # NOTE: each row in `pred_labels` is a sequence of IOB predictions (
        # each character is labeled)
        pred_labels = self._get_predictions(line_texts)
        # picks out the speaker name from the text in each line.
        speaker_names = []
        texts = []
        for i, line in enumerate(line_texts):
            speaker_name = ''
            text = ''
            # for each character, add it to `speaker_name` or `text` depending on
            # predicted label.
            for j, c in enumerate(line):
                pred_label = pred_labels[i][j]
                if LABEL_CODES[pred_label] in ['B', 'I']:
                    speaker_name += c
                elif LABEL_CODES[pred_label] in ['O']:
                    text += c
                else:
                    raise RuntimeError(f'SupervisedSpeakerParser only accepts the '
                        f'following labels: {list(LABEL_CODES.keys())}')
            speaker_names.append(speaker_name.strip())
            texts.append(text.strip())
            # prev_speaker = speaker_name
        parsed_speaker_names = self.RuleSpeakerParser._parse_speaker_names(speaker_names)
        return speaker_names, parsed_speaker_names, texts


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