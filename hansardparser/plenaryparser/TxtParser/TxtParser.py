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
from bs4 import Tag
from unicodedata import normalize

from hansardparser.plenaryparser import utils
from hansardparser.plenaryparser.models.Entry import Entry
from hansardparser.plenaryparser.models.Sitting import Sitting
from hansardparser.plenaryparser.HansardParser import HansardParser
from hansardparser.plenaryparser.TxtParser.LineLabeler import RuleLineLabeler, SupervisedLineLabeler
from hansardparser.plenaryparser.TxtParser.SpeakerParser import RuleSpeakerParser
from hansardparser import settings


ACCEPTABLE_LINE_LABELERS = ['supervised', 'rule']
ACCEPTABLE_SPEAKER_PARSERS = ['rule']


class TxtParser(HansardParser):
    """The TxtParser class parses Hansard txt files into a structured array of
    speeches.

    Attributes:

        line_labeler: str = 'rule'. One of: ['supervised', 'rule']. If 'supervised', uses a
            trained classifier to predict the line label. If 'rule', uses rules
            (regexes and boolean tests) to determine the line label.
        
        line_predict_kws: dict = None. Dict of keyword arguments to pass to
            `predict_from_strings`. Only used if line_labeler == 'supervised'.
            Example::

                {'builder_path': 'PATH_TO_BUILDER', 'clf_path': 'PATH_TO_CLF'}
        
        line_label_codes: dict = None. Dict containing the mapping from a label
            code to the string label. Only used if line_labeler == 'supervised'. 
            Example::

                {0: 'header', 1: 'speech', 2: 'scene'}

        speaker_parser: str = 'rule'. One of: ['rule']. If 'rule', uses rules
            (regexes and boolean tests) to determine extract the speaker name
            from a line of text.
        
        see parent class for additional attributes (`Hansardparser`).

    Usage::

        >>> text = # ...load an unparsed Hansard text file.
        >>> parser = TxtParser(verbosity=1)
        >>> results = parser.parse_hansards(text, to_format=None)
    
    Todos:

        TODO: have a clearer separation of the use of `text` vs. `lines` to refer
            to either a string or list of strings.
    """

    def __init__(self,
                 line_labeler: str = 'rule',
                 line_predict_kws: dict = None,
                 line_label_codes: dict = None,
                 speaker_parser: str = 'rule',
                 verbosity: int = 0,
                 *args, **kwargs):
        assert line_labeler in ACCEPTABLE_LINE_LABELERS, \
            f'`line_labeler` must be one of {ACCEPTABLE_LINE_LABELERS}.'
        if line_labeler == 'rule':
            LineLabeler = RuleLineLabeler(verbosity=verbosity)
            if verbosity > 0:
                if line_predict_kws is not None:
                    warnings.warn('You provided `line_predict_kws`, but `line_predict_kws` '
                        'are only used when `line_labeler="supervised"`. `line_predict_kws` '
                        'will not be used.')
                if line_label_codes is not None:
                    warnings.warn('You provided `line_label_codes`, but `line_label_codes ` '
                        'are only used when `line_labeler="supervised"`. '
                        '`line_label_codes` will not be used.')
        elif line_labeler == 'supervised':
            LineLabeler = SupervisedLineLabeler(predict_kws=line_predict_kws,
                label_codes=line_label_codes, verbosity=verbosity)
        else:
            raise NotImplementedError
        assert speaker_parser in ACCEPTABLE_SPEAKER_PARSERS, \
            f'`speaker_parser` must be one of {ACCEPTABLE_SPEAKER_PARSERS}.'
        if speaker_parser == 'rule':
            SpeakerParser = RuleSpeakerParser(verbosity=verbosity)
        else:
            raise NotImplementedError
        kwargs['LineLabeler'] = LineLabeler
        kwargs['SpeakerParser'] = SpeakerParser
        kwargs['verbosity'] = verbosity
        HansardParser.__init__(self, *args, **kwargs)


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
        if self.verbosity > 0:
            print(f'Found {len(sitting_texts)} sittings in file.')
        for sitting_text in sitting_texts:
            lines = self._preprocess_text(sitting_text)
            metadata = self._extract_metadata(lines)
            entries = self._parse_entries(lines)
            if merge:
                entries = self._merge_entries(entries)
            entries = self._clean_entries(entries)
            if to_format is not None:
                entries = self._convert_contents(entries, to_format=to_format)
            results.append((metadata, entries))
        time1 = time.time()
        if self.verbosity > 0:
            print(f'Processed {len(results)} files in {time1 - time0:.2f} seconds.')
        return results


    def _split_sittings(self, text: str) -> List[str]:
        """splits a string of text containing multiple parliamentary sittings
        into a list of strings where each element represents one sitting.

        Arguments:

            text: str.

        Returns:

            sitting_text: List[str].
        
        Todos:

            TODO: write tests for this method. The whole thing is very kludgy.
        """
        # out = text.decode('utf-8')  # windows-1252
        # TODO: this method should not return List of soups. Just return a single
        # soup. Make a decision about how this parser should be used.
        # TODO: remove hard-coded compendium meta at top and bottom of file.
        text = normalize('NFKD', text)  # .encode('ASCII', 'ignore')
        if '|' in text:
            text = text.replace('|', '/')
            if self.verbosity > 0:
                warnings.warn('Found "|" in this document. Replaced with "/".', RuntimeWarning)
        # split single text file by sitting.
        sitting_end_regex = re.compile(r'[\n\r](<i>)?the house rose at .{1,50}[\n\r]', re.IGNORECASE|re.DOTALL)
        split_text = sitting_end_regex.split(text)
        new_split_text = []
        for i in range(0, len(split_text), 2):
            try:
                assert len(split_text[i+1]) < 100, 'this text is supposed to be less than 100 characters.'
                new_split_text.append(split_text[i] + '\n' + split_text[i+1])
            except (IndexError, TypeError):
                # for split_text[-1]
                new_split_text.append(split_text[i])
        assert new_split_text[-1] == split_text[-1]
        sitting_texts = []
        for i, text in enumerate(new_split_text):
            # soup = BeautifulSoup(text, 'html.parser')
            if len(text) > 0:  # NOTE: might not be necessary?
                sitting_texts.append(text)
        return sitting_texts


    def _preprocess_text(self, text: str) -> List[str]:
        """splits text on line breaks and skips over lines with no length.
        """
        # NOTE: the splitting regex in the line below leads to
        # differences from the line numbers in the original text file. This
        # is only a problem if you want to make direct comparisons to the
        # original text file.
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
        #         if self.verbosity > 2:
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
        if self.verbosity > 1:
            print(f'Number of lines before preprocessing: {num_lines}')
            print(f'Number of lines after preprocessing: {num_lines_after}')
        return new_text



    def _extract_metadata(self, lines: List[str], metadata: Sitting = None, max_check: int = 50) -> Sitting:
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
        max_check = min(max_check, len(lines))
        while i < max_check and metadata.is_incomplete():
            line = lines[i]
            added = self._add_to_meta(line, metadata)
            i += 1
            if added:
                if self.verbosity > 1:
                    print(f'Added line to metadata: {line}')
                    print(f'Updated metadata: {metadata}')
                lines_to_rm.append(i)
            elif utils.is_page_date(line) or utils.is_page_heading(line) or utils.is_page_number(line):
                lines_to_rm.append(i)
        for i in sorted(lines_to_rm)[::-1]:
            lines.pop(i)
        # combines date and time.
        if metadata.date is not None:
            if metadata.time is not None:
                regex = re.search(r'(?P<hour>\d{1,2})\.(?P<min>\d{1,2})', metadata.time)
                metadata.date = metadata.date.replace(hour=int(regex.group('hour')), minute=int(regex.group('min')))
            del metadata.time
        if self.verbosity > 0 and metadata.date is None:
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


    def _parse_entries(self, lines: List[str]) -> List[Entry]:
        """Parses a Hansard Sitting transcript into a list of 'entries'.

        Assigns a "label" to each line, extracts the speaker name from the
        beginning of each line (where applicable), and parses each speaker
        name into the speaker's title, cleaned name, and appointment.

        Returns:

            contents_merged: List[Entry]. a processed and cleaned
                list of Entry objects representing each entry in the transcript.
        """
        labels = self.LineLabeler.label_lines(lines)
        if self.verbosity > 1:
            for i, label in enumerate(labels):
                if label is None:
                    warnings.warn(f'Did not find label for line: "{lines[i]}"', RuntimeWarning)
        speaker_names, parsed_speaker_names, texts = self.SpeakerParser.extract_speaker_names(lines, labels)
        entries = self._create_entries(texts, labels, speaker_names, parsed_speaker_names)
        return entries


    def _create_entries(self,
                        texts: List[str],
                        labels: List[str],
                        speaker_names: List[str],
                        parsed_speaker_names: List[tuple]) -> List[Entry]:
        """converts data into a list of Entry objects.
        """
        entries = []
        for i in range(len(texts)):
            # if entry_type in ['speech_new', 'speech_ctd']:
            #     entry_type = 'speech'
            title, speaker_cleaned, appointment = parsed_speaker_names[i]
            entry = Entry(entry_type=labels[i], text=texts[i], speaker=speaker_names[i],
                page_number=None, speaker_cleaned=speaker_cleaned, title=title,
                appointment=appointment)
            entries.append(entry)
            if self.verbosity > 1:
                print('------------------------')
                print(f'Line number: {i}')
                print(f'Speaker name: {speaker_names[i]}')
                print(f'Entry type: {labels[i]}')
                print(f'Text: {texts[i]}')
        return entries


    def _merge_entries(self, entries: List[Entry]) -> List[Entry]:
        """merges entries as appropriate."""
        entries_merged = []
        prev_entry = Entry()
        while len(entries) > 0:
            entry = entries.pop(0)
            if prev_entry.can_merge(entry) and len(entries_merged):
                prev_entry.merge_entries(entry, self.verbosity)
                entries_merged[-1] = prev_entry
            else:
                entries_merged.append(entry)
                prev_entry = entry
        return entries_merged


    def _clean_entries(self, entries: List[Entry]) -> List[Entry]:
        """cleans each parsed entry.

        This method is called after the list of entries has been constructed and
        merged.

        Todos:

            TODO: remove html tags from text.
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