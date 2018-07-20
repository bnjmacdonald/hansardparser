
from typing import List, Tuple
from hansardparser.plenaryparser.utils import extract_flatworld_tags

from hansardparser.plenaryparser.classify.tf.predict import predict_from_strings
from hansardparser.plenaryparser.TxtParser.SpeakerParser import RuleSpeakerParser

# TODO: ?? move this to a config file somewhere ??
LABEL_CODES = {0: 'I', 1: 'O', 2: 'B'}


class SupervisedSpeakerParser(object):
    """Invokes a trained classifier to extract the speaker name from a line in a
    Hansard transcript.

    Attributes:

        predict_kws: dict. Dict of keyword arguments to pass to `predict_from_strings`.

    """
    
    def __init__(self, predict_kws: dict, verbosity: int = 0):
        self.predict_kws = predict_kws
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
        _, _, pred_labels = predict_from_strings(line_texts, verbosity=self.verbosity, **self.predict_kws)
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
                        'following labels: {list(LABEL_CODES.keys())}')
            speaker_names.append(speaker_name.strip())
            texts.append(text.strip())
            # prev_speaker = speaker_name
        parsed_speaker_names = self.RuleSpeakerParser._parse_speaker_names(speaker_names)
        return speaker_names, parsed_speaker_names, texts
