
import os
import re
import warnings
from typing import List, Optional, Set
from google.cloud import storage

from utils import (
    is_punct,
    extract_flatworld_tags,
    is_page_number,
    is_page_heading,
    is_page_date,
    is_page_footer)

BUCKET_NAME = 'hansardparser-data'
HEADERS_FILEPATH = 'generated/plenaryparser/headers.txt'


def get_headers() -> set:
    """retrieves headers from `headers.txt` file on google cloud storage."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(HEADERS_FILEPATH)
    headers = set(blob.download_as_string().strip().decode('utf-8').split('\n'))
    return headers


class Rule(object):
    """Uses rules to assign a label to each line in a Hansard transcript.
    
    Uses rules (regular expressions and boolean tests) to assign a label to a line
    of Hansard text. Label is one of: [header, subheader, subsubheader, speech, scene, garbage]

    Attributes:

        headers: Set[str] = None. Set of strings representing known headers.
            Example:: {"bills", "motions", "oral questions"}. This set of headers
            will be used to determine whether a line should be classified as a
            "header" (as opposed to a subheader or subsubheader).
    """

    def __init__(self, headers: Set[str] = None, verbosity: int = 0):
        if headers is None:
            # headers that will be used to assign "header" vs. "subheader" label.
            headers = get_headers()
        self.headers = headers
        self.verbosity = verbosity


    def label_lines(self, lines: List[str], **kwargs) -> List[str]:
        """Returns the label/class of each line in the Hansard transcript.
        
        Possible labels: header, subheader, subsubheader, speech, scene, garbage.

        Arguments:
            
            lines: List[str]. List of lines in a Hansard transcript.
        
        Returns:

            labels: List[str]. List of line labels.
        """
        labels = [self._label_one_line(line, **kwargs) for line in lines]
        return labels


    def _label_one_line(self,
                        line: str,
                        check_if_page_header: bool = True) -> Optional[str]:
        """Returns the label of a single line, extracted using a rule-based parser.

        Arguments:

            line: str. Line in a Hansard transcript.

            check_if_page_header: bool. If True, checks if line is a page number
                or page header (and returns "garbage" label if so).

        Returns:

            label: Optional[str]. Label of line. If no label is found, returns
                None.
        
        Todos:

            TODO: lines with a speaker name in all caps are getting labeled as a header,
                but they shouldn't be.
                Examples::

                    `MR. OMYAHCHA (CTD.):`
                    `MR. BIDU (CTD):`

                One way to address this would be to try to extract a speaker name
                from the line. If a speaker name is extracted, then it is a speech.
        """
        if self._is_garbage(line, check_if_page_header):
            return 'garbage'
        if is_punct(line, True):
            return 'punct'
        line_text, flatworld_tags = extract_flatworld_tags(line)
        test_results = {
            'header': self._is_header(line_text, flatworld_tags),
            'subheader': self._is_subheader(line_text, flatworld_tags),
            'subsubheader': self._is_subsubheader(line_text, flatworld_tags),
            'speech': self._is_speech(line_text, flatworld_tags),
            'scene': self._is_scene(line_text, flatworld_tags)
        }
        if sum(test_results.values()) > 1:
            # KLUDGE: gives precedence to header over speech
            if test_results['speech'] and test_results['header']:
                test_results['speech'] = False
            # KLUDGE: gives precedence to scene over speech
            if test_results['speech'] and test_results['scene']:
                test_results['speech'] = False
            # KLUDGE: gives precedence to header over scene
            if test_results['header'] and test_results['scene']:
                test_results['scene'] = False
        if self.verbosity > 1 and sum(test_results.values()) > 1:
            warnings.warn(f'Multiple labels found for line: {line};\nLabels found: {", ".join([k for k, v in test_results.items() if v])}')
        # returns label string.
        for k, v in test_results.items():
            if v:
                return k
        if self.verbosity > 1:
            warnings.warn(f'Did not find label for line: {line}', RuntimeWarning)
        return None


    def _is_garbage(self, line: str, check_if_page_header: bool = False) -> bool:
        """checks if line fits conditions for a "garbage" label. Returns True if
        so, False otherwise.

        Arguments:

            line: bs4 tag object. A single element from body.contents.

            check_if_page_header: bool. If True, checks if line is a page number
                or page header (and returns "garbage" label if so).
        """
        # checks for page number, heading, date.
        if line is None or len(line) == 0:
            return True
        if check_if_page_header:  # if less than 10 lines from start of page...
            if is_page_number(line) or is_page_heading(line) or is_page_date(line):
                return True
        if is_page_footer(line):
            return True
        return False


    def _is_header(self, line: str, flatworld_tags: Optional[List[str]] = None) -> bool:
        """checks if line fits conditions for a "header" label. Returns True if
        so, False otherwise."""
        is_header = bool(
            line.strip().lower() in self.headers
            # tag_is_header or
            # (text_eq_upper and not line.endswith('.'))
        )
        # header_test = header_test1 or header_test2
        # if is_header and len(utils.rm_punct(line)) < 5 and re.search(r'\d', line):
        #     prev_entry_type = self._get_line_label(line.prev_sibling, False)
        #     next_entry_type = self._get_line_label(line.next_sibling, False)
        #     is_header = prev_entry_type == 'header' or next_entry_type == 'header'
        return is_header


    def _is_subheader(self, line: str, flatworld_tags: Optional[List[str]] = None) -> bool:
        """checks if line fits condition for a "subheader" label. Returns True if
        so, False otherwise.
        """
        text_eq_upper = line == line.upper()
        tag_is_subheader = False
        if flatworld_tags is not None:
            tag_is_subheader = any([bool(re.search(r'header', tag)) for tag in flatworld_tags]) 
        is_subheader = bool(
            line.strip().lower() not in self.headers
            and (
                tag_is_subheader or
                (text_eq_upper and not line.endswith('.')) or
                bool(re.search(r'^first reading|^second reading|^question no|^no[\.,]\s*\d{1,4}$', line, re.IGNORECASE))
            )
        )
        return is_subheader
    

    def _is_subsubheader(self, line: str, flatworld_tags: Optional[List[str]] = None) -> bool:
        """checks if line fits conditions for a "subsubheader" label. Returns True if
        so, False otherwise."""
        tag_is_subsubheader = False
        if flatworld_tags is not None:
            tag_is_subsubheader = any([bool(re.search(r'subsubheader', tag)) for tag in flatworld_tags]) 
        is_subsubheader = bool(
            tag_is_subsubheader or (
                len(line) < 200 and
                bool(re.search(r'^clause|^\(the house resumed\)|^(first|'
                      r'second|third|fourth|fifth|sixth) schedule$', line, re.IGNORECASE))
                
            )
        )
        return is_subsubheader


    def _is_speech(self, line: str, flatworld_tags: Optional[List[str]] = None) -> bool:
        """checks if line fits conditions for a "speech" label. Returns True if
        so, False otherwise."""
        text_neq_upper = line != line.upper()
        tag_is_speech_new = False
        if flatworld_tags is not None:
            tag_is_speech_new = any([bool(re.search(r'speech', tag)) for tag in flatworld_tags]) 
        is_speech_new = bool(
            tag_is_speech_new or
            text_neq_upper
        )
        return is_speech_new


    def _is_scene(self, line: str, flatworld_tags: Optional[List[str]] = None) -> bool:
        """checks if line fits conditions for a "scene" label. Returns True if
        so, False otherwise."""
        text_neq_upper = line != line.upper()
        tag_is_scene = False
        if flatworld_tags is not None:
            tag_is_scene = any([bool(re.search(r'scene', tag)) for tag in flatworld_tags]) 
        scene_test = bool(
            tag_is_scene or
            (text_neq_upper and bool(re.search(r'^[\(\[].+[\)\]]$', line, re.DOTALL)))  # starts with and ends with parentheses
        )
        return scene_test

