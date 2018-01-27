"""Defines the LineLabelExtractor class.
"""

import re
import warnings

from hansardparser.plenaryparser import xml_to_pdf
from hansardparser.plenaryparser import utils

class LineLabelExtractor(object):
    """Extracts line labels.

    Attributes:

        verbose : bool
            False by default. Set to True if detailed output to console is
            desired.

        parliament_dates : dict like {int -> (datetime, datetime)}
            dictionary of parliaments-date pairs. Gives range of dates for
            each parliament.

        italic_phrases : list of strings
            List containing strings that appear as italic phrases in speeches,
            but which should not be treated as a scene entry_type.
    
    Usage:

        >>> from hansardparser.plenaryparser.build_training_set.extract_line_labels import LineLabelExtractor
        >>> file_path = "tests/test_input/1st December, 1999P.pdf"
        >>> extractor = LineLabelExtractor(verbose=3)
        >>> soup = extractor.convert_pdf(file_path, save_soup=False)
        >>> labels, lines = extractor.extract_labels(soup)
        >>> print(labels[78:81])
        ['speech_new', 'garbage', 'speech_ctd']
        >>> print(lines[78:81])
        [
            <text font="2" height="13" left="171" top="632" width="434"><b>Mr. Wamunyinyi </b>asked the Minister for Education, Science and Technology:-</text>,
            '\n',
            <text font="0" height="13" left="171" top="649" width="556">(a) what the Ministry is doing to streamline the financial and administrative structures of secondary</text>
        ]
        
    """

    italic_phrases = [
        'vis-a-vis',
        'pangas',
        'jua kali',
        'Jua Kali',
        'askaris',
        'boda boda',
        'wananchi',
        'mwananchi',
        'dukawallas',
        'persona non grata',
        'kazi kwa vijana',
        'vijana,',
        'vijana',
        'kwa vijana',
        'matatu',
        'matatus',
        'mungiki',
        'mungiki.',
        'animal farm.',
        'taliban',
        'wazungu',
        'et cetera',
        'vipande',
        'whatsapp',
        'facebook',
        'gazette',
        'et cetera',
        'simba',
        'locus standi',
        'Mukurwe wa Nyagathanga',
        'Njuri Ncheke',
        'Mau Mau',
        'Mzungu',
        'et cetera',
        'moranism',
        'morans',
        'El Nino',
        'El-Nino',
        'bona fide',
        'Harambee',
        'Harambee,',
        'ad hoc',
        'Kayas',
        'Kaya',
        'Kaya Tiwi',
        'vice versa.',
        'vice versa',
        'posteriori',
        'Kenya Times',
        'mundu khu mundu',
        'Medusa',
        'vide',
    ]

    def __init__(self, italic_phrases=None, verbose=0):
        if italic_phrases is not None:
            self.italic_phrases += italic_phrases
        self.verbose = verbose

    def convert_pdf(self, file_path, save_soup=False, path=None):
        """converts PDF to XML soup."""
        return xml_to_pdf.convert(file_path, save_soup, path, self.verbose)

    def extract_labels(self, soup):
        """Traverses an xml soup object and extracts the label for each line.

        Arguments:
            
            soup : bs4 soup object. Must have pdf2xml tag.

        Returns:

            labels, lines: 2-tuple.

                labels: list of str. List containing the label of each xml line.
            
                lines: list of bs4 tag objects. List containing each xml line.
        """
        contents = soup.pdf2xml.contents
        labels = []
        lines = []
        i = 0
        # for each page...
        while len(contents):
            i += 1
            tag = contents.pop(0)
            if tag.name is None or tag.text is None or tag.text.strip() is u'':
                labels.append('garbage')
                lines.append(tag)
                continue
            tag_contents = tag.contents
            j = 0
            # for each tag in the page...
            while len(tag_contents):
                j += 1
                line = tag_contents.pop(0)
                if line.name and len(line.findChildren()) > 1 and self.verbose > 1:
                    msg = 'this line in page has more than 1 child:\n{0}'.format(line)
                    warnings.warn(msg, RuntimeWarning)
                # retrieves line label
                label = self._get_line_label(line, check_if_page_header=j < 10)
                if label is not None:
                    labels.append(label)
                    lines.append(line)
                if self.verbose > 2:
                    print('extracted {0} label from line: {1}'.format(label, line))
        assert len(labels) == len(lines), "Should have same number of labels as lines."
        return labels, lines

    def _get_line_label(self, line, check_if_page_header):
        """Returns the label of a line (either header, subheader, subsubheader,
        speech, scene, garbage, or punct).

        Arguments:

            line: bs4 tag object. A single element from body.contents.

            check_if_page_header: bool. If True, checks if line is a page number
                or page header (and returns "garbage" label if so).
        
        Returns:

            label: str. Label of line (either header, subheader, subsubheader,
                speech_new, speech_ctd, scene, garbage, or punct).
        """
        if self.__is_garbage(line, check_if_page_header):
            return 'garbage'
        if utils.is_punct(line.text.strip(), True):
            return 'punct'
        test_results = {
            'special_header': self.__is_special_header(line),
            'header': self.__is_header(line),
            'subheader': self.__is_subheader(line),
            'subsubheader': self.__is_subsubheader(line),
            'speech_new': self.__is_speech_new(line),
            'speech_ctd': self.__is_speech_ctd(line),
            'scene': self.__is_scene(line)
        }
        if sum(test_results.values()) > 1:
            warnings.warn('Multiple labels found for line: {0};\nLabels found: {1}'.format(line, ', '.join([k for k, v in test_results.items() if v])))
            return None
        # returns label string.
        for k, v in test_results.items():
            if v:
                return k
        warnings.warn('Did not find label for line: {0}'.format(line), RuntimeWarning)
        return None

    def __is_garbage(self, line, check_if_page_header):
        """checks if line fits conditions for a "garbage" label. Returns True if
        so, False otherwise.

        Arguments:

            line: bs4 tag object. A single element from body.contents.

            check_if_page_header: bool. If True, checks if line is a page number
                or page header (and returns "garbage" label if so).
        """
        # checks for page number, heading, date.
        if line is None or line.name is None or line.text is None or line.text.strip() is u'':
            return True
        line_text = line.text.strip()
        if check_if_page_header:  # if less than 10 lines from start of page...
            if utils.is_page_number(line_text):
                return True
            if utils.is_page_heading(line_text) or utils.is_page_date(line_text):
                return True
        if utils.is_page_footer(line_text):
            return True
        return False
    
    def __is_header(self, line):
        """checks if line fits conditions for a "header" label. Returns True if
        so, False otherwise."""
        line_text = line.text.strip()
        text_eq_upper = line_text == line_text.upper()
        b_tags = line.find_all('b')
        is_header = bool(
            text_eq_upper and
            len(b_tags) and
            not line_text.endswith('.')
        )
        # header_test = header_test1 or header_test2
        if is_header and len(utils.rm_punct(line_text)) < 5 and re.search(r'\d', line_text):
            prev_entry_type = self._get_line_label(line.prev_sibling, False)
            next_entry_type = self._get_line_label(line.next_sibling, False)
            is_header = prev_entry_type == 'header' or next_entry_type == 'header'
        return is_header

    def __is_subheader(self, line):
        """checks if line fits conditions for a "subheader" label. Returns True if
        so, False otherwise."""
        b_tags = line.find_all('b')
        line_text = line.text.strip()
        text_eq_upper = line_text == line_text.upper()
        is_subheader = bool(
            text_eq_upper and
            not len(b_tags) and
            not line_text.endswith('.')
        )
        if is_subheader and len(utils.rm_punct(line_text)) < 5 and re.search(r'\d', line_text):
            prev_entry_type = self._get_line_label(line.prev_sibling, False)
            next_entry_type = self._get_line_label(line.next_sibling, False)
            is_subheader = prev_entry_type == 'subheader' or next_entry_type == 'subheader'
        return is_subheader
    
    def __is_subsubheader(self, line):
        """checks if line fits conditions for a "subsubheader" label. Returns True if
        so, False otherwise."""
        line_text = line.text.strip()
        i_tags = line.find_all('i')
        is_subsubheader = bool(
            re.search(r'^clause|^question no|^\(the house resumed\)|^(first|second|third|fourth|fifth|sixth) schedule$', line_text, re.IGNORECASE) and
            len(i_tags)
        )
        return is_subsubheader

    def __is_special_header(self, line):
        """checks if line fits conditions for a "special_header" label. Returns True if
        so, False otherwise.

        "special_headers" are kind of an awkward category. Currently,
        special_headers include the "first reading [...]" and "second reading [...]"
        headers. The reason these are classified as a special header is because
        they need to be treated differently in post-processsing the extracted text
        (i.e. when merging together consecutive headers). See XmlParser for how
        this post-processing works. 
        """
        line_text = line.text.strip()
        i_tags = line.find_all('i')
        is_special_header = bool(
            bool(re.search(r'^first reading|^second reading', line_text, re.IGNORECASE)) and
            len(i_tags)
        )
        return is_special_header
    
    def __is_speech_new(self, line):
        """checks if line fits conditions for a "speech_new" label. Returns True if
        so, False otherwise."""
        line_text = line.text.strip()
        text_eq_upper = line_text == line_text.upper()
        i_tags = line.find_all('i')
        italic_text = line.find('i').text.strip() if len(i_tags) else ''
        is_speech_new = bool(
            bool(line.find('b')) and
            not re.search(r'\d+', line.find('b').text) and
            not utils.is_punct(line.find('b').text, True) and
            not text_eq_upper and
            (
                not len(i_tags) or
                (
                    len(i_tags) and
                    italic_text != line_text
                )
            )
            # all([day not in line_text.lower() for day in days])
        )
        return is_speech_new
    
    def __is_speech_ctd(self, line):
        """checks if line fits conditions for a "speech_ctd" label. Returns True if
        so, False otherwise."""
        line_text = line.text.strip()
        text_eq_upper = line_text == line_text.upper()
        b_tags = line.find_all('b')
        i_tags = line.find_all('i')
        italic_text = line.find('i').text.strip() if len(i_tags) else ''
        is_speech_ctd = bool(
            not self.__is_subsubheader(line) and
            (
                not text_eq_upper or
                (
                    text_eq_upper and
                    line_text.endswith('.')
                ) or
                (
                    len(line_text) < 5 and not self.__is_subheader(line)
                )
            ) and
            (
                not len(b_tags) or
                (
                    len(b_tags) and
                    (
                        ''.join([tag.text for tag in b_tags]).strip() is u'' or
                        utils.is_punct(line.find('b').text, True)
                    )
                )
            ) and
            (
                not len(i_tags) or
                (
                    len(i_tags) and
                    (
                        italic_text.lower() in self.italic_phrases or
                        len(italic_text) < 10 or
                        utils.rm_punct(italic_text) != utils.rm_punct(line_text)
                    )
                )
            )
        )
        return is_speech_ctd
    
    def __is_scene(self, line):
        """checks if line fits conditions for a "scene" label. Returns True if
        so, False otherwise."""
        line_text = line.text.strip()
        i_tags = line.find_all('i')
        italic_text = line.find('i').text.strip() if len(i_tags) else ''
        scene_test = bool(
            len(i_tags) and
            not self.__is_subsubheader(line) and
            not self.__is_special_header(line) and
            not italic_text.lower() in self.italic_phrases and
            (
                italic_text == line_text or
                (
                    len(italic_text) > 10 and
                    utils.rm_punct(italic_text) == utils.rm_punct(line_text)
                )
            )
        )
        return scene_test