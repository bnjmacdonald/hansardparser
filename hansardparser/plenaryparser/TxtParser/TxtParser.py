"""Defines the TxtParser subclass.

Parses a txt Kenya Hansard transcript into a list of Entry objects,
which can than be converted into a dictonary or Pandas DataFrame using
the hansard_convert.py module.

Notes:

    * This implementation was inspired in part by: https://github.com/mysociety/pombola/blob/master/pombola/hansard/kenya_parser.py
"""

import os
import re
import time
import warnings
import datetime
from copy import deepcopy
from typing import List, Union, Optional, Tuple
import pandas as pd
from unicodedata import normalize

import utils
from models.Entry import Entry
from TxtParser import LineTypeLabeler, LineSpeakerSpanLabeler, LineHasSpeakerLabeler

ALLOWED_EXTENSIONS = ['txt', 'pdf']
ALLOWED_LINE_TYPE_LABELERS = ['supervised', 'rule']
# ALLOWED_LINE_HAS_SPEAKER_LABELERS = ['supervised', 'rule']
ALLOWED_LINE_SPEAKER_SPAN_LABELERS = ['supervised', 'hybrid', 'rule']

DEBUG = False

class TxtParser(object):
    f"""The TxtParser class parses Hansard txt or pdf files into a structured
    array of speeches.

    Attributes:

        line_labeler: str = 'rule'. One of: {ALLOWED_LINE_TYPE_LABELERS}. If 'supervised', uses a
            trained classifier to predict the line label. If 'rule', uses rules
            (regexes and boolean tests) to determine the line label.

        speaker_parser: str = 'rule'. One of: {ALLOWED_LINE_SPEAKER_SPAN_LABELERS}. If 'supervised',
            uses a trained classifier to extract the speaker name. If 'rule', uses
            rules (regexes and boolean tests) to extract the speaker name from a
            line of text.

    Usage::

        >>> fname = # path to a txt or pdf file containing a hansard transcript.
        >>> parser = TxtParser(verbosity=1)
        >>> results = parser.parse_hansard(fname)
    """

    def __init__(self,
                 line_type_labeler: str = 'rule',
                 line_speaker_span_labeler: str = 'rule',
                 verbosity: int = 0):
        assert line_type_labeler in ALLOWED_LINE_TYPE_LABELERS, \
            f'`line_type_labeler` must be one of {ALLOWED_LINE_TYPE_LABELERS}.'
        # initializes line type labeler.
        if line_type_labeler == 'rule':
            line_type_labeler = LineTypeLabeler.Rule(verbosity=verbosity)
        elif line_type_labeler == 'supervised':
            line_type_labeler = LineTypeLabeler.Supervised(verbosity=verbosity)
        else:
            raise NotImplementedError
        # initializes line speaker span labeler
        assert line_speaker_span_labeler in ALLOWED_LINE_SPEAKER_SPAN_LABELERS, \
            f'`line_labeler` must be one of {ALLOWED_LINE_SPEAKER_SPAN_LABELERS}.'
        if line_speaker_span_labeler == 'rule':
            line_speaker_span_labeler = LineSpeakerSpanLabeler.Rule(verbosity=verbosity)
        elif line_speaker_span_labeler == 'supervised':
            line_speaker_span_labeler = LineSpeakerSpanLabeler.Supervised(verbosity=verbosity)
        elif line_speaker_span_labeler == 'hybrid':
            line_speaker_span_labeler = LineHasSpeakerLabeler.Supervised(verbosity=verbosity)
        else:
            raise NotImplementedError
        self.line_type_labeler = line_type_labeler
        self.line_speaker_span_labeler = line_speaker_span_labeler
        self.verbosity = verbosity
        self._sitting_text = None
        self._unmerged_parsed_transcript = None
        self._line_type4_preds = None
        self._line_speaker_span_preds = None


    def parse_hansard(self,
                       filepath_or_buffer: str,
                       start_line: int = 0,
                       end_line: int = None,
                       filetype: str = None) -> List[Tuple[dict, Union[list, pd.DataFrame]]]:
        """parses one or more Hansard transcripts contained in `text`.

        A single Hansard transcript contains all speeches and parliamentary business
            from a single sitting (e.g. morning sitting, November 12th 2015).

        Arguments:

            text: str. Text of Hansards. May contain multiple Hansard transcripts.
                These will be split using `self._split_sittings` and parsed separately.

            merge: bool = True. Merge entries after `self._parse_entries()` has
                been called. Setting this to False is useful for debugging purposes,
                so that you can see each Entry before merging.

        Returns:

            results: dict. Dict containing metadata and list of entries.
                Keys:

                meta: dict. contains metadata on the parsed Hansard
                    transcript.

                entries: List[dict]. List of entries representing a single
                    sitting.
        """
        time0 = time.time()
        try:
            if isinstance(filepath_or_buffer, str):
                if self.verbosity > 0:
                    print(f'Parsing sitting(s) in {filepath_or_buffer}...')
                if filetype is None:
                    filetype = utils.get_filetype(filepath_or_buffer)
                f = open(filepath_or_buffer, 'rb')
            else:
                f = filepath_or_buffer
            assert filetype in ALLOWED_EXTENSIONS
            if filetype == 'txt':
                text = f.read()
            elif filetype == 'pdf':
                text = utils.pdf2str(f)
            if start_line > 0:
                text = text.split('\n')
                if end_line is None:
                    text = '\n'.join(text[start_line:])
                else:
                    text = '\n'.join(text[start_line:end_line])
        finally:
            f.close()
        text = utils.normalize_text(text)
        if DEBUG:
            text = text[:1500]
        self._sitting_text = text
        lines = self._preprocess_text(text)
        metadata = None
        entries = None
        if len(lines) > 0:
            metadata = self._extract_metadata(text)
            entries = self._parse_entries(lines)
            # constructs dict of unmerged parsed transcript.
            self._unmerged_parsed_transcript = {"meta": deepcopy(metadata), "entries": [deepcopy(entry.__dict__) for entry in entries]}
            entries = self._merge_entries(entries)
            entries = self._postprocess_entries(entries)
            entries = [{k:v for k, v in sorted(entry.__dict__.items())} for entry in entries]
        result = {"meta": metadata, "entries": entries}
        time1 = time.time()
        if self.verbosity > 0:
            print(f'Processed transcript in {time1 - time0:.2f} seconds.')
        return result


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
            line, _ = utils.extract_flatworld_tags(line)
            if len(line) > 0:
                new_text.append(line)
        num_lines_after = len(new_text)
        if self.verbosity > 1:
            print(f'Number of lines before preprocessing: {num_lines}')
            print(f'Number of lines after preprocessing: {num_lines_after}')
        return new_text


    def _extract_metadata(self, text: str) -> dict:
        """extracts metadata from the initial lines in contents.
        """
        date = self._extract_sitting_date(text)
        times = self._extract_sitting_time(text)
        if self.verbosity > 0:
            if date is None:
                warnings.warn('Date not found in transcript.')
            if 'start' not in times:
                warnings.warn('Start time not found in transcript.')
            if 'end' not in times:
                warnings.warn('Start time not found in transcript.')
        start_date = date
        if date is not None and times['start'] is not None:
            start_date = start_date.replace(hour=times['start'][0], minute=times['start'][1])
        end_date = None
        if date is not None and times['end'] is not None:
            end_date = date.replace(hour=times['end'][0], minute=times['end'][1])
        return {'start_date': start_date, 'end_date': end_date}


    def _extract_sitting_time(self, text: str) -> dict:
        """extracts the sitting start and end times from a text transcript.
        
        Time formats:
            The House met at 2.30 p.m.
            The House rose at 8.30 p.m.
            ...
        """
        start_regex = re.compile(r'the\s*house\s*(?P<action>met)\s*at\s*(?P<hour>\d{1,2})[\.:;](?P<minute>\d{1,2})\s*(?P<ampm>[ap]\.?m\.?)', flags=re.IGNORECASE)
        end_regex = re.compile(r'the\s*house\s*(?P<action>rose)\s*at\s*(?P<hour>\d{1,2})[\.:;](?P<minute>\d{1,2})\s*(?P<ampm>[ap]\.?m\.?)', flags=re.IGNORECASE)
        # searches for start time in first n characters of the sitting text.
        n_chars = 100
        start_time = None
        while start_time is None and n_chars < 1500:
            this_text = re.sub(r'[\n\r]+', '', text[:n_chars])
            this_text = re.sub(r'\s+', ' ', this_text.strip())
            regex_res = start_regex.search(this_text)
            if regex_res is not None:
                start_time = regex_res.groupdict()
                assert start_time['action'].lower() == 'met'
                start_time['hour'] = int(start_time['hour'])
                start_time['minute'] = int(start_time['minute'])
                if start_time['ampm'].startswith('p') and start_time['hour'] != 12:
                    start_time['hour'] += 12
                elif start_time['ampm'].startswith('a') and start_time['ampm'] == 12:
                    # e.g. 12:30am -> 0:30
                    start_time['hour'] = 0
                start_time = (start_time['hour'], start_time['minute'])
                assert start_time[0] < 24, f'Start time hour is greater than 23 (start time hour: {start_time[0]}).'
                assert start_time[1] < 60, f'Start time minute is greater than 59 (start time minutee: {start_time[1]}).'
            n_chars += 100
        # searches for end time in last n characters of the sitting text.
        n_chars = 100
        end_time = None
        while end_time is None and n_chars < 1500:
            this_text = re.sub(r'[\n\r]+', '', text[-n_chars:])
            this_text = re.sub(r'\s+', ' ', this_text.strip())
            regex_res = end_regex.search(this_text)
            if regex_res is not None:
                end_time = regex_res.groupdict()
                assert end_time['action'].lower() == 'rose'
                end_time['hour'] = int(end_time['hour'])
                end_time['minute'] = int(end_time['minute'])
                if end_time['ampm'].startswith('p') and end_time['hour'] != 12:
                    end_time['hour'] += 12
                elif end_time['ampm'].startswith('a') and end_time['ampm'] == 12:
                    # e.g. 12:30am -> 0:30
                    end_time['hour'] = 0
                end_time = (end_time['hour'], end_time['minute'])
                assert end_time[0] < 24, f'End time hour is greater than 23 (end time hour: {end_time[0]}).'
                assert end_time[1] < 60, f'End time minute is greater than 59 (end time minutee: {end_time[1]}).'
            n_chars += 100
        return {'start': start_time, 'end': end_time}


    def _extract_sitting_date(self, text: str) -> datetime.datetime:
        """extracts the sitting date from a text transcript.

        Date formats:
            WWWW, DDth MMMM, YYYY ("Thursday, 29th November, 2001")
            WWWW, MMMM DDth, YYYY ("Thursday, November 29th, 2001")
            MMMM DDth, YYYY ("October 10th, 2005")
            DDth MMMM, YYYY ("10th October, 2005")
        """
        # date regexes.
        month2num = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
            'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        month_abbr2num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sept': 9, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month2num.update(month_abbr2num)
        days_of_week = 'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday'
        months = '|'.join(month2num.keys())
        days = '|'.join([str(x) for x in range(1, 32)])
        day_suffixes = 'th|st|nd|rd'
        years = '|'.join([str(x) for x in range(1900, datetime.datetime.today().year+1)])
        flags = re.DOTALL|re.IGNORECASE
        tests = [
            re.compile(rf'(?P<day_of_week>{days_of_week})[\s,]*(?P<day>{days})[\s\.,]*({day_suffixes})?[\s\.,]*(?P<month>{months})[\s,]*(?P<year>{years})', flags=flags),
            re.compile(rf'(?P<day_of_week>{days_of_week})[\s,]*(?P<month>{months})[\s,]*(?P<day>{days})[\s\.,]*({day_suffixes})?[\s\.,]*(?P<year>{years})', flags=flags),
            re.compile(rf'(?P<month>{months})[\s,]*(?P<day>{days})[\s\.,]*({day_suffixes})?[\s\.,]*(?P<year>{years})', flags=flags),
            re.compile(rf'(?P<day>{days})[\s\.,]*({day_suffixes})?[\s\.,]*(?P<month>{months})[\s,]*(?P<year>{years})', flags=flags),
        ]
        # searches for date in first n characters of the sitting text.
        n_chars = 100
        date = None
        while date is None and n_chars < 1500:
            tests_temp = tests.copy()
            this_text = re.sub(r'[\n\r]+', '', text[:n_chars])
            this_text = re.sub(r'\s+', ' ', this_text.strip())
            while date is None and len(tests_temp) > 0:
                regex = tests_temp.pop(0)
                regex_res = regex.search(this_text)
                if regex_res is not None:
                    year = int(regex_res.groupdict()['year'])
                    month_int = month2num[regex_res.groupdict()['month']]
                    day = int(regex_res.groupdict()['day'])
                    date = datetime.datetime(year, month_int, day)
            n_chars += 100
        return date


    def _parse_entries(self, lines: List[str]) -> List[Entry]:
        """Parses a Hansard sitting transcript into a list of 'entries'.

        Assigns a "label" to each line, extracts the speaker name from the
        beginning of each line (where applicable), and parses each speaker
        name into the speaker's title, cleaned name, and appointment.

        Returns:

            contents_merged: List[Entry]. a processed and cleaned
                list of Entry objects representing each entry in the transcript.
        """
        line_type_labels = self.line_type_labeler.label_lines(lines)
        assert len(line_type_labels) == len(lines)
        self._line_type4_preds = list(zip(lines, line_type_labels))
        if self.verbosity > 1:
            for i, label in enumerate(line_type_labels):
                if label is None:
                    warnings.warn(f'Did not find label for line: "{lines[i]}"', RuntimeWarning)
        # speaker_names, parsed_speaker_names, texts, speaker_span_labels = self.line_speaker_span_labeler.extract_speaker_names(lines, types=line_type_labels)
        speaker_span_labels = self.line_speaker_span_labeler.label_speaker_spans(lines, types=line_type_labels)
        assert len(speaker_span_labels) == len(lines)
        speaker_names, texts = self.line_speaker_span_labeler.extract_speaker_names(lines, speaker_span_labels)
        assert len(speaker_names) == len(lines) and len(texts) == len(lines)
        RuleLineSpeakerSpanLabeler = LineSpeakerSpanLabeler.Rule(verbosity=self.verbosity)
        parsed_speaker_names = RuleLineSpeakerSpanLabeler.parse_speaker_names(speaker_names)
        assert len(parsed_speaker_names) == len(lines)
        # overrides speaker name extraction for lines that are not speeches.
        for i, line in enumerate(lines):
            if line_type_labels[i] != 'speech':
                if self.verbosity > 1 and re.search(r'[BI]', speaker_span_labels[i]):
                    warnings.warn(f'I found a speaker name in a "{line_type_labels[i]}" line. '
                                  f'I am overriding this by setting the speaker_name '
                                  f'to None. Line: "{lines[i]}"', RuntimeWarning)
                speaker_names[i] = None
                parsed_speaker_names[i] = (None, None, None)
                texts[i] = line
                speaker_span_labels[i] = 'O' * len(texts[i])
        self._line_speaker_span_preds = list(zip(lines, speaker_span_labels))
        # KLUDGE: for lines that contain only part of a speaker name that is
        # continued on the next line, append this speaker name to the beginning
        # of the next line.
        for i, bio_labels in enumerate(speaker_span_labels):
            # if line ends with a speaker name and next line starts with a speaker name...
            if i+1 < len(lines) and bio_labels.endswith('I') and re.search(r'^[BI]', speaker_span_labels[i+1]):
                assert len(texts[i].strip()) == 0, f'Expected text to be empty, but text = {texts[i]}'
                speaker_names[i+1] = speaker_names[i] + ' ' + speaker_names[i+1]
                parsed_speaker_names[i+1] = RuleLineSpeakerSpanLabeler._parse_speaker_name(speaker_names[i+1])
        entries = self._create_entries(texts, line_type_labels, speaker_names, parsed_speaker_names)
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
        return entries


    def _merge_entries(self, entries: List[Entry]) -> List[Entry]:
        """merges entries as appropriate."""
        # speech_divider: highest level line type between current line and last
        # known speech line (header > scene > garbage > None). 
        entries_merged = []
        prev_entry = None
        while len(entries) > 0:
            entry = entries.pop(0)
            if entry.entry_type == 'garbage':
                continue
            if len(entries_merged) > 0 and prev_entry.can_merge(entry):
                prev_entry.merge_entries(entry, self.verbosity)
                entries_merged[-1] = prev_entry
                continue
            elif len(entries_merged) > 1 and entry.entry_type == 'speech' and prev_entry.entry_type in ['scene', 'garbage']:
                # if current entry is a speech and previous entry is a
                # scene or is garbage, then there is a chance that the
                # prev entry is dividing a speech from the same person.
                j = len(entries_merged) - 1
                temp_prev_entry = entries_merged[j]
                while temp_prev_entry.entry_type in ['scene', 'garbage'] and j >= 0:
                    j -= 1
                    temp_prev_entry = entries_merged[j]
                if temp_prev_entry.can_merge(entry):
                    temp_prev_entry.merge_entries(entry, self.verbosity)
                    entries_merged[j] = temp_prev_entry
                    continue
            entries_merged.append(entry)
            prev_entry = entries_merged[-1]
        return entries_merged


    def _postprocess_entries(self, entries: List[Entry]) -> List[Entry]:
        """cleans and prunes entries after they have been parsed and merged.

        This method is called after the list of entries has been
        constructed and merged. This method is useful for cleaning up
        small details (e.g. remove extra spacing in header words, drop
        speech entries with no text, etc).

        Todos:

            TODO: remove html tags from text.
        """
        # cleans text.
        for entry in entries:
            entry.text = utils.clean_text(entry.text)
            if entry.entry_type in ['header', 'subheader']:
                entry.text = utils.fix_header_words(entry.text.lower())
                entry.text = re.sub(r'^PRAYERS?\s+(.+)$', r'\1', entry.text, flags=re.IGNORECASE|re.DOTALL)
            # entries2.append(entry)
        self._distribute_headers(entries)
        # prunes list of entries. Only keeps speeches and scenes.
        entries_pruned = []
        while len(entries) > 0:
            entry = entries.pop(0)
            if entry.entry_type is not None and entry.entry_type not in ['garbage']:
                if 'header' not in entry.entry_type and (entry.text is None or len(entry.text) == 0):
                    continue
                entries_pruned.append(entry)
        return entries_pruned


    def _distribute_headers(self, entries: List[Entry]) -> List[Entry]:
        """Carries forward headers, subheaders, and subsubheadeers throughout the
        entries."""
        current_header = None
        current_subheader = None
        current_subsubheader = None
        consec_headers = False
        for entry in entries:
            if entry.entry_type is not None and 'header' in entry.entry_type:
                # if len(data) > 0 and data[-1][:3] != [current_header, current_subheader, current_subsubheader]:
                #     data.append([current_header, current_subheader, current_subsubheader] + [None]*len(attributes))
                if entry.entry_type == 'header':
                    current_header = entry.text
                    current_subheader = None
                    current_subsubheader = None
                elif entry.entry_type == 'subheader':
                    current_subheader = entry.text
                    current_subsubheader = None
                elif entry.entry_type == 'subsubheader':
                    current_subsubheader = entry.text
                else:
                    raise RuntimeError(f'{entry.entry_type} entry type not recognized.')
                entry.text = None
                assert entry.speaker is None, f'Header entry has a speaker: {entry.__dict}'
            entry.header = current_header
            entry.subheader = current_subheader
            entry.subsubheader = current_subsubheader
        return None

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