
from typing import List, Optional
from hansardparser.plenaryparser.utils import extract_flatworld_tags

from hansardparser.plenaryparser.classify.tf.predict import predict_from_strings
from hansardparser.plenaryparser.TxtParser.LineLabeler import RuleLineLabeler

# KLUDGE: so that predict_from_strings works.
from text2vec.processing.preprocess import preprocess_one
from hansardparser.plenaryparser.build_training_set.utils import str2ascii_safe


class SupervisedLineLabeler(object):
    """Invokes a trained classifier to assign a label to each line of a Hansard
    transcript.
    
    Label is one of: [header, subheader, subsubheader, speech, scene, garbage].

    Attributes:

        predict_kws: dict. Dict of keyword arguments to pass to `predict_from_strings`.

        label_codes: dict. Dict of mappings from a numeric code to a string label.
            Example::

                {0: 'header', 1: 'speech', 2: 'scene', 3: 'garbage'}
    """
    
    def __init__(self, predict_kws: dict, label_codes: dict, verbosity: int = 0):
        self.predict_kws = predict_kws
        self.label_codes = label_codes
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
            line_text, _ = extract_flatworld_tags(line)
            line_texts.append(line_text)
        _, _, pred_labels = predict_from_strings(line_texts, verbosity=self.verbosity, **self.predict_kws)
        # TODO: subheaders and subsubheaders also need to be dealt with. 
        labels = [self.label_codes[l] for l in pred_labels]
        # retrieves header type ('header', 'subheader', or 'subsubheader').
        for i, line in enumerate(lines):
            if labels[i] == 'header':
                labels[i] = self._get_header_type(line)
                # print(labels[i], line)
        return labels


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
