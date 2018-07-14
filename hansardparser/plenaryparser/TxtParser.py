"""Defines the TxtParser subclass.

Parses a txt Kenya Hansard transcript into a list of Entry objects,
which can than be converted into a dictonary or Pandas DataFrame
using the hansard_convert.py module. Module was initially built
based on April 11th, 2006 transcript.

See super-class (hansard_parser.py) for notes on implementation.
"""

import os
import re
import time
import warnings
from typing import List, Union, Optional, Tuple
import pandas as pd
# from bs4 import BeautifulSoup, Tag, NavigableString
from unicodedata import normalize

from hansardparser.plenaryparser import utils
from hansardparser.plenaryparser.models.Entry import Entry
from hansardparser.plenaryparser.models.Sitting import Sitting
from hansardparser.plenaryparser.HansardParser import HansardParser
from hansardparser import settings

HEADERS_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'plenaryparser', 'headers.txt')


class TxtParser(HansardParser):
    """The TxtParser class parses Hansard txt files into a structured array of
    speeches.

    Attributes:

        see parent class (`Hansardparser`).

    Usage::

        >>> text = # ...load an unparsed Hansard text file.
        >>> parser = TxtParser(verbose=1)
        >>> results = parser.parse_hansards(text, to_format=None)
    """
    # headers that will be used to assign "header" vs. "subheader" label.
    with open(HEADERS_PATH, 'r') as f:
        headers = set([l.strip().lower() for l in f.readlines() if len(l.strip()) > 0])

    def __init__(self, italic_phrases: List[str] = None, verbose: int = 0):
        HansardParser.__init__(self, italic_phrases=italic_phrases, verbose=verbose)


    def parse_hansards(self,
                       text: str,
                       merge: bool = True,
                       to_format: str = 'df-long') -> List[Tuple[Sitting, Union[list, pd.DataFrame]]]:
        """parses one or more Hansard transcripts contained in `text`.

        A single Hansard transcript contains all speeches and parliamentary business
            from a single sitting (e.g. morning sitting, November 12th 2015).

        Arguments:

            text: str. Text of Hansards. May contain multiple Hansard transcripts.
                These will be split using `self._split_sittings` and parsed separately.

            merge: bool = True. Merge entries after `self._parse_entries()` has
                been called. Setting this to False is useful for debugging purposes,
                so that you can see each Entry before merging.

            to_format: str. Converts parsed transcripts to a pd.DataFrame, if
                desired. See `self._convert_contents`. If None, contents are
                not converted at all (i.e. they are left as a list of Entry
                objects).

        Returns:

            results: List[Tuple[Sitting, Union[list, pd.DataFrame]]]. List of tuples,
                where each tuple contains:

                metadata: Sitting. contains metadata on the parsed Hansard
                    transcript.

                entries: Union[List[Entry], pd.DataFrame]. List of entries
                    representing a single Sitting. If `to_format == None`, then
                    this is a list of Entry objects. If `to_format != None`,
                    then this is in the format returned by `self._convert_contents`.
        """
        results = []
        time0 = time.time()
        sitting_texts = self._split_sittings(text)
        for text in sitting_texts:
            self.lines = self._preprocess_text(text)
            metadata = self._extract_metadata()
            entries = self._parse_entries()
            if merge:
                entries = self._merge_entries(entries)
            entries = self._clean_entries(entries)
            if to_format is not None:
                entries = self._convert_contents(entries, to_format=to_format)
            results.append((metadata, entries))
        time1 = time.time()
        if self.verbose:
            print(f'Processed {len(results)} files in {time1 - time0:.2f} seconds.')
        return results


    def _split_sittings(self, text: str) -> List[str]:
        """splits a string of text containing multiple parliamentary sittings
        into a list of strings where each element represents one sitting.

        Arguments:

            text: str.

        Returns:

            sitting_text: List[str].
        """
        # out = text.decode('utf-8')  # windows-1252
        # TODO: this method should not return List of soups. Just return a single
        # soup. Make a decision about how this parser should be used.
        # TODO: remove hard-coded compendium meta at top and bottom of file.
        text = normalize('NFKD', text)  # .encode('ASCII', 'ignore')
        if '|' in text:
            text = text.replace('|', '/')
            if self.verbose > 0:
                warnings.warn('Found "|" in this document. Replaced with "/".', RuntimeWarning)
        # split single text file by sitting.
        sitting_end_regex = re.compile(r'([\n\r]the house rose at .{1,50}\n)', re.IGNORECASE|re.DOTALL)
        split_text = sitting_end_regex.split(text)
        new_split_text = []
        for i in range(0, len(split_text), 2):
            try:
                assert len(split_text[i+1]) < 100
                new_split_text.append(split_text[i] + '\n' + split_text[i+1])
            except IndexError:
                # for split_text[-1]
                new_split_text.append(split_text[i])
        assert new_split_text[-1] == split_text[-1]
        sitting_texts = []
        for i, text in enumerate(new_split_text):
            # soup = BeautifulSoup(text, 'html.parser')
            sitting_texts.append(text)
        return sitting_texts


    def _preprocess_text(self, text: str) -> List[str]:
        """splits text on line breaks and skips over lines with no length.
        """
        # extracts whitespace tags.
        # num_descendants = sum([1 for d in self.soup.descendants])
        text = re.split(r'[\n\r]+', text)
        num_lines = len(text)
        new_text = []
        for line in text:
            line = line.strip()
            if len(line) > 0:
                new_text.append(line)
        # splits tags with lengthy whitespace in between text.
        # for tag in self.soup.descendants:
        #     tag_text = tag.text.strip() if isinstance(tag, Tag) else tag.string.strip()
        #     if re.search(r'\s{4,}', tag_text):
        #         if self.verbose > 2:
        #             print(f'splitting tag: {tag}')
        #         split_text = re.split(r'\s{4,}', tag_text)
        #         current_tag = tag
        #         for i, text in enumerate(split_text):
        #             if isinstance(tag, Tag):
        #                 new_tag = self.soup.new_tag(current_tag.name, **current_tag.attrs)
        #                 new_tag.string = text
        #             else:
        #                 new_tag = NavigableString(text)
        #             # if first in split, replace existing tag. Else insert after.
        #             if i == 0:
        #                 current_tag.replace_with(new_tag)
        #             else:
        #                 current_tag.insert_after(new_tag)
        #             current_tag = new_tag
        # num_descendants_after = sum([1 for d in self.soup.descendants])
        num_lines_after = len(new_text)
        if self.verbose > 1:
            print(f'Number of lines before preprocessing: {num_lines}')
            print(f'Number of lines after preprocessing: {num_lines_after}')
        return new_text



    def _extract_metadata(self, metadata: Sitting = None, max_check: int = 50) -> Sitting:
        """extracts metadata from the initial lines in contents.

        Todos:

            TODO: revise extract metadata.
                lines like "Wednesday,      26th September  1335-1380" should not be a date.
                revise `is_page_date`.
        """
        if metadata is None:
            metadata = Sitting()
        i = 0
        lines_to_rm = []
        max_check = min(max_check, len(self.lines))
        while i < max_check and metadata.is_incomplete():
            line = self.lines[i]
            added = self._add_to_meta(line, metadata)
            i += 1
            if added:
                if self.verbose > 1:
                    print(f'Added line to metadata: {line}')
                    print(f'Updated metadata: {metadata}')
                lines_to_rm.append(i)
            elif utils.is_page_date(line) or utils.is_page_heading(line) or utils.is_page_number(line):
                lines_to_rm.append(i)
        for i in sorted(lines_to_rm)[::-1]:
            self.lines.pop(i)
        # combines date and time.
        if metadata.date is not None:
            if metadata.time is not None:
                regex = re.search(r'(?P<hour>\d{1,2})\.(?P<min>\d{1,2})', metadata.time)
                metadata.date = metadata.date.replace(hour=int(regex.group('hour')), minute=int(regex.group('min')))
            del metadata.time
        if self.verbose > 0 and metadata.date is None:
            warnings.warn('No date found in sitting.', RuntimeWarning)
        return metadata


    def _add_to_meta(self, line: str, metadata: Sitting) -> bool:
        """Attempts to add the contents of a line to the transcript metadata.

        Arguments:

            line: str. a string containing text to be added to the metadata.

            metadata: Sitting. A Sitting object as defined in sitting_class.py.

        Returns:

            bool. returns 0 if line is None or if no update is made, otherwise
                returns 1 once the metadata Sitting object has been updated based
                on line.
        """
        if line is None or len(line) == 0:
            return False
        page_number_test = line.isdigit() and len(line) < 5
        if page_number_test:
            if metadata.start_page is None:
                metadata.start_page = line
                return True
            else:
                return False
        if utils.is_transcript_heading(line):
            if metadata.heading is None:
                metadata.heading = utils.get_transcript_heading(line)
            else:
                metadata.heading += ' ' + utils.get_transcript_heading(line)
            return True
        if utils.is_str_date(line):
            metadata.date = utils.convert_str_to_date(line)
            return True
        time_reg = re.compile(r'The House (?P<action>met|rose) at ?(?P<time>\d+\.\d+.* [ap].m.*)')
        if time_reg.search(line):
            result = time_reg.search(line)
            metadata.time = result.group('time').strip()
            return True
        return False


    def _parse_entries(self):
        """Parses a Hansard Sitting transcript.

        Assigns a "label" to each line (one of: header, speech, scene, or garbage),
            extracts the speaker name from the beginning of each line (where
            applicable), and parses each speaker name into the speaker's title,
            cleaned name, and appointment.

        NOTE: each step in this method could be replaced with a trained classifier.

        Returns:

            contents_merged : list of Entry objects. a processed and cleaned
                list of Entry objects representing each entry in the transcript.
        """
        labels = self._extract_labels()
        speaker_names, texts = self._extract_speaker_names(labels)
        parsed_speaker_names = self._parse_speaker_names(speaker_names)
        entries = self._create_entries(texts, labels, speaker_names, parsed_speaker_names)
        return entries


    def _extract_labels(self) -> List[str]:
        labels = []
        for line in self.lines:
            label = self._get_line_label(line)
            labels.append(label)
            if label is None and self.verbose > 1:
                warnings.warn(f'Did not find entry_type for line: "{line}"', RuntimeWarning)
        return labels


    def _extract_speaker_names(self, labels: List[str]) -> Tuple[List[str], List[str]]:
        """extracts the speaker name from the beginning of each line.

        Extracts the speaker name from the beginning of each line. Only extracts
            speaker names where `label[i] == 'speech'`.

        Returns:

            speaker_names, texts: Tuple[List[str], List[str]].

                speaker_names: List[str]. List of speaker names, of same length
                    as `labels`. If `label[i] != 'speech'` or no speaker name
                    is found, then `speaker_name[i] = None`.

                texts: List[str]. Lines of text after speaker name has been
                    extracted.
        """
        speaker_names = []
        texts = []
        for i, line in enumerate(self.lines):
            speaker_name = None
            text, _, _ = utils.extract_outer_tag(line) # KLUDGE: ...
            if labels[i] == 'speech':
                speaker_name, text = utils.extract_speaker_name(text)
                # speaker_cleaned, title, appointment = utils.parse_speaker_name(speaker_name)
            speaker_names.append(speaker_name)
            texts.append(text)
            # prev_speaker = speaker_name
        return speaker_names, texts


    def _parse_speaker_names(self, speaker_names: List[Optional[str]]) -> List[Tuple[str, str, str]]:
        """parses each speaker name in list of speaker names.

        Each speaker name is parsed by splitting a name into the speaker's title,
        cleaned name, and appointment.

        Returns:

            parsed_names: List[Tuple[str, str, str]]. List of parsed names.
                If an input speaker name is None, the parsed name will be
                `(None, None, None)`.
        """
        parsed_names = []
        for i, name in enumerate(speaker_names):
            parsed_name = utils.parse_speaker_nameV2(name)
            parsed_names.append(parsed_name)
        return parsed_names


    def _create_entries(self,
                        texts: List[str],
                        labels: List[str],
                        speaker_names: List[str],
                        parsed_speaker_names: List[tuple]) -> List[Entry]:
        entries = []
        for i in range(len(texts)):
            # if entry_type in ['speech_new', 'speech_ctd']:
            #     entry_type = 'speech'
            title, speaker_cleaned, appointment = parsed_speaker_names[i]
            entry = Entry(entry_type=labels[i], text=texts[i], speaker=speaker_names[i],
                page_number=None, speaker_cleaned=speaker_cleaned, title=title,
                appointment=appointment)
            entries.append(entry)
            if self.verbose > 1:
                print('------------------------')
                print(f'Line number: {i}')
                print(f'Speaker name: {speaker_names[i]}')
                print(f'Entry type: {labels[i]}')
                print(f'Text: {texts[i]}')
        return entries


    def _get_line_label(self,
                        line: str,
                        check_if_page_header: bool = False) -> Optional[str]:
        """Returns the label of a line.

        Possible labels: [header, speech, scene, garbage, punct].

        Arguments:

            line: str. A single element from body.contents.

            check_if_page_header: bool. If True, checks if line is a page number
                or page header (and returns "garbage" label if so).

        Returns:

            label: Optional[str]. Label of line. One of: [header, speech, scene,
                garbage, punct]. If no label is found, returns None.
        
        Todos:

            TODO: lines with a speaker name in all caps get labeled as a header.
                Examples::

                    `MR. OMYAHCHA (CTD.):`
                    `MR. BIDU (CTD):`

                One way to address this would be to try to extract a speaker name
                from the line. If a speaker name is extracted, then it is a speech.
        """
        if self._is_garbage(line, check_if_page_header):
            return 'garbage'
        if utils.is_punct(line, True):
            return 'punct'
        line_text, open_tag, close_tag = utils.extract_outer_tag(line)
        open_tag = open_tag.strip().lower() if open_tag else open_tag
        close_tag = close_tag.strip().lower() if close_tag else close_tag
        test_results = {
            'header': self._is_header(line_text, open_tag, close_tag),
            'subheader': self._is_subheader(line_text, open_tag, close_tag),
            'subsubheader': self._is_subsubheader(line_text, open_tag, close_tag),
            'speech': self._is_speech(line_text, open_tag, close_tag),
            'scene': self._is_scene(line_text, open_tag, close_tag)
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
        if self.verbose > 0 and sum(test_results.values()) > 1:
            warnings.warn(f'Multiple labels found for line: {line};\nLabels found: {", ".join([k for k, v in test_results.items() if v])}')
        # returns label string.
        for k, v in test_results.items():
            if v:
                return k
        if self.verbose > 0:
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
            if utils.is_page_number(line) or utils.is_page_heading(line) or utils.is_page_date(line):
                return True
        if utils.is_page_footer(line):
            return True
        return False


    def _is_header(self, line: str, open_tag: Optional[str], close_tag: Optional[str]) -> bool:
        """checks if line fits conditions for a "header" label. Returns True if
        so, False otherwise."""
        # text_eq_upper = line == line.upper()
        # tag_is_header = (open_tag and 'header' in open_tag) or (close_tag and 'header' in close_tag)
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

    def _is_subheader(self, line: str, open_tag: Optional[str], close_tag: Optional[str]) -> bool:
        """checks if line fits condition for a "subheader" label. Returns True if
        so, False otherwise.
        """
        text_eq_upper = line == line.upper()
        tag_is_subheader = (open_tag and 'header' in open_tag) or (close_tag and 'header' in close_tag)
        is_subheader = bool(
            line.strip().lower() not in self.headers
            and (
                tag_is_subheader or
                (text_eq_upper and not line.endswith('.')) or
                self.__is_special_header(line)  # KLUDGE: ...
            )
        )
        return is_subheader


    def _is_subsubheader(self, line: str, open_tag: Optional[str], close_tag: Optional[str]) -> bool:
        """checks if line fits conditions for a "subsubheader" label. Returns True if
        so, False otherwise."""
        tag_is_subsubheader = (open_tag and 'subsubheader' in open_tag) or (close_tag and 'subsubheader' in close_tag)
        is_subsubheader = bool(
            tag_is_subsubheader or (
                len(line) < 200 and
                bool(re.search(r'^clause|^question no|^no[\.,] \d{1,4}$|^\(the house resumed\)|^(first|'
                      r'second|third|fourth|fifth|sixth) schedule$', line, re.IGNORECASE))
                
            )
        )
        return is_subsubheader


    def __is_special_header(self, line: str) -> bool:
        """checks if line fits conditions for a "special_header" label. Returns True if
        so, False otherwise.

        "special_headers" are kind of an awkward category. Currently,
        special_headers include the "first reading [...]" and "second reading [...]"
        headers. The reason these are classified as a special header is because
        they need to be treated differently in post-processsing the extracted text
        (i.e. when merging together consecutive headers). See XmlParser for how
        this post-processing works.
        """
        is_special_header = bool(
            re.search(r'^first reading|^second reading', line, re.IGNORECASE)
        )
        return is_special_header


    def _is_speech(self, line: str, open_tag: Optional[str], close_tag: Optional[str]) -> bool:
        """checks if line fits conditions for a "speech" label. Returns True if
        so, False otherwise."""
        text_neq_upper = line != line.upper()
        tag_is_speech_new = open_tag == 'newspeech' or close_tag == 'newspeech'
        is_speech_new = bool(
            tag_is_speech_new or
            text_neq_upper
        )
        return is_speech_new


    def _is_scene(self, line: str, open_tag: Optional[str], close_tag: Optional[str]) -> bool:
        """checks if line fits conditions for a "scene" label. Returns True if
        so, False otherwise."""
        text_neq_upper = line != line.upper()
        tag_is_scene = open_tag == 'scene' or close_tag == 'scene'
        scene_test = bool(
            tag_is_scene or
            (text_neq_upper and bool(re.search(r'^[\(\[].+[\)\]]$', line, re.DOTALL)))  # starts with and ends with parentheses
        )
        return scene_test


    def _merge_entries(self, entries: List[Entry]) -> List[Entry]:
        """merges entries as appropriate."""
        entries_merged = []
        prev_entry = Entry()
        while len(entries) > 0:
            entry = entries.pop(0)
            if prev_entry.can_merge(entry) and len(entries_merged):
                prev_entry.merge_entries(entry, self.verbose)
                entries_merged[-1] = prev_entry
            else:
                entries_merged.append(entry)
                prev_entry = entry
        return entries_merged


    def _clean_entries(self, entries: List[Entry]) -> List[Entry]:
        """cleans each parsed entry.

        This method is called after the list of entries has been constructed and
        merged.
        """
        # cleans text.
        entries_merged = []
        while len(entries):
            entry = entries.pop(0)
            entry.text = utils.clean_text(entry.text)
            entries_merged.append(entry)
        return entries_merged


    # def _assign_speaker(self, entries: List[Entry]) -> List[Entry]:
    #     # assigns speaker name to 'speech_ctd'.
    #     # current speaker is carried forward until (a) header appears; (b) speech
    #     # has a speaker name.
    #     current_speaker = None
    #     for entry in entries:
    #         if entry.entry_type == 'header':
    #             current_speaker = None
    #         if entry.entry_type == 'speech':
    #             if current_speaker is None and entry.speaker is not None:
    #                 # if there entry has a speaker and there is no current speaker,
    #                 # then update current speaker.
    #                 # KLUDGE: this is kinda awkward...
    #                 current_speaker = {'speaker': entry.speaker, 'speaker_cleaned': entry.speaker_cleaned,
    #                     'title': entry.title, 'appointment': entry.appointment}
    #             elif entry.speaker is not None and current_speaker is not None and entry.speaker != current_speaker['speaker']:
    #                 # if entry has a speaker and is not equal to current speaker,
    #                 # then the current speaker has changed.
    #                 # KLUDGE: this is kinda awkward...
    #                 current_speaker = {'speaker': entry.speaker, 'speaker_cleaned': entry.speaker_cleaned,
    #                     'title': entry.title, 'appointment': entry.appointment}
    #             elif current_speaker is not None and entry.speaker is None:
    #                 # else if there is a current speaker and the entry does not
    #                 # have a speaker, then it must be the same speaker continuing.
    #                 # KLUDGE: this is kinda awkward...
    #                 entry.speaker = current_speaker['speaker']
    #                 entry.speaker_cleaned = current_speaker['speaker_cleaned']
    #                 entry.title = current_speaker['title']
    #                 entry.appointment = current_speaker['appointment']
    #     return entries