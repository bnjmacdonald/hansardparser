"""Defines the XmlParser class.

Parses a pdf Kenya Hansard transcript into a list of Entry objects,
which can than be converted into a dictonary or Pandas DataFrame
using the hansard_convert.py module. Module was initially built
based on April 11th, 2006 transcript.
"""

import os
import re
import warnings
import copy
from bs4 import BeautifulSoup, Tag, NavigableString

import settings
from hansardparser.plenaryparser import xml_to_pdf
from hansardparser.plenaryparser.Entry import Entry
from hansardparser.plenaryparser.Sitting import Sitting
from hansardparser.plenaryparser.HansardParser import HansardParser
from hansardparser.plenaryparser import utils

class XmlParser(HansardParser):
    """The Hansard_parser contains methods for parsing an Hansard PDF into
    txt or html.

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
    """

    def __init__(self, italic_phrases=None, verbose=0):
        HansardParser.__init__(self, italic_phrases=italic_phrases, verbose=verbose)

    def _convert_pdf(self, file_path, save_soup=False, path=None):
        return xml_to_pdf.convert(file_path, save_soup, path, self.verbose)

    def _preprocess_soup(self):
        """removes empty strings and does other basic preprocessing of soup.

        Note: this could be marginally faster by combining the functionality
            into a single loop, but because there are tags getting extracted
            and inserted, it seems safer to keep the logics separate.
        """
        # extracts whitespace tags.
        num_descendants = sum([1 for d in self.soup.pdf2xml.descendants])
        tags_to_extract = []
        for tag in self.soup.pdf2xml.descendants:
            tag_text = tag.text.strip() if isinstance(tag, Tag) else tag.string.strip()
            if tag.name != 'fontspec' and tag_text is u'':
                tags_to_extract.append(tag)
        for tag in tags_to_extract:
            if self.verbose > 2:
                print('extracting tag: {0}'.format(tag))
            tag.extract()
        # splits tags with lengthy whitespace in between text.
        for tag in self.soup.pdf2xml.descendants:
            if tag.name == 'text':
                tag_text = tag.text.strip()
                if re.search(r'\s{10,}', tag_text):
                    if self.verbose > 2:
                        print('splitting tag: {0}'.format(tag))
                    split_text = re.split(r'\s{10,}', tag_text)
                    tag.string = split_text[0]
                    current_tag = tag
                    for text in split_text[1:]:
                        new_tag = self.soup.new_tag(current_tag.name, **current_tag.attrs)
                        new_tag.string = text
                        current_tag.insert_after(new_tag)
                        current_tag = new_tag
        num_descendants_after = sum([1 for d in self.soup.pdf2xml.descendants])
        if self.verbose > 1:
            print('Number of descendants before: {0}'.format(num_descendants))
            print('Number of descendants after: {0}'.format(num_descendants_after))
        return 0

    def _extract_metadata(self, metadata, max_check):
        """extracts metadata from the initial lines in contents."""

        first_page_contents = self.soup.find('page', {'number': 1}).contents
        max_check = min(max_check, len(first_page_contents))
        first_page_contents = first_page_contents[:max_check]
        # checked = 0
        # lines_to_extract = []
        while len(first_page_contents) and metadata.is_incomplete():  # line_text != 'PRAYERS' and entry_type != 'scene'
            line = first_page_contents.pop(0)
            added = self._add_to_meta(line, metadata)
            if added:
                if self.verbose > 1:
                    print('Added line to metadata: {0}'.format(line))
                    print('Metadata: {0}'.format(metadata))
                _ = line.extract()
            else:
                line_text = line.text.strip() if isinstance(line, Tag) else line.string.strip()
                if utils.is_page_date(line_text) or utils.is_page_heading(line_text) or utils.is_page_number(line_text):
                    _ = line.extract()
            # print(checked, metadata.is_incomplete())
            # else:
                # checked += 1
        # combines date and time.
        if metadata.date is not None:
            if metadata.time is not None:
                regex = re.search(r'(?P<hour>\d{1,2})\.(?P<min>\d{1,2})', metadata.time)
                metadata.date = metadata.date.replace(hour=int(regex.group('hour')), minute=int(regex.group('min')))
            del metadata.time
        return metadata

    def _add_to_meta(self, line, metadata):
        """Attempts to add the contents of line_text to the transcript metadata.

        Arguments:
            line_text : str
                a string containing text to be added to the metadata.
            metadata : Sitting object
                a Sitting object as defined in sitting_class.py.

        Returns:
            returns 0 if line_text is None or if no update is made, otherwise
            returns 1 once the metadata Sitting object has been updated based
            on line_text.
        """
        line_text = line.text.strip() if isinstance(line, Tag) else line.string.strip()
        if line_text in [None, '']:
            return False
        page_number_test = line_text.isdigit() and len(line_text) < 5
        if page_number_test:
            if metadata.start_page is None:
                metadata.start_page = line_text
                return True
            else:
                return False
        if utils.is_transcript_heading(line_text):
            if metadata.heading is None:
                metadata.heading = utils.get_transcript_heading(line_text)
            else:
                metadata.heading += ' ' + utils.get_transcript_heading(line_text)
            return True
        if utils.is_str_date(line_text):
            metadata.date = utils.convert_str_to_date(line_text)
            return True
        if re.match(r'^(Monday|Tuesday|Wednesday|Thursday|Friday)', line_text) and len(line_text) < 70:
            # line.next_sibling.text.strip() in ('th', 'st', 'nd', 'rd') and
            if utils.is_str_date(line_text):
                metadata.date = utils.convert_str_to_date(line_text)
                return True
            else:
                text = line_text + ' ' + line.next_sibling.text.strip()
                if utils.is_str_date(text):
                    metadata.date = utils.convert_str_to_date(text)
                    line.next_sibling.extract()
                    return True
                else:
                    text += ' ' + line.next_sibling.next_sibling.text.strip()
                    if utils.is_str_date(text):
                        metadata.date = utils.convert_str_to_date(text)
                        line.next_sibling.extract()
                        return True
        time_reg = re.compile(r'The House (?P<action>met|rose) at ?(?P<time>\d+\.\d+.* [ap].m.*)')
        if time_reg.search(line_text):
            result = time_reg.search(line_text)
            metadata.time = result.group('time').strip()
            return True
        return False



    def _process_contents(self, current_page):
        """Processes the contents of a transcript in xml format and returns a
        list of cleaned Entries.

        NOTE: prev_prev_entry is used for concatenating speeches that are
        split by one scene entry. This causes scene entries to end up at the
        END of each speech. So it is still possible to count the number of applauses, et cetera for a given speech, but this approach makes it much harder to determine exactly what was being applauded.

        Arguments:
            current_page : int
                int representing current page number of transcript

        Returns:
            contents_merged : list of Entry objects
                a processed and cleaned list of Entry objects representing each entry in the transcript.
        """
        contents = self.soup.pdf2xml.contents
        contents_entries = []
        prev_entry = Entry()
        page_number = current_page
        # for each page in contents...
        i = 0
        while len(contents):
            i += 1
            tag = contents.pop(0)
            if tag.name is None or tag.text is None or tag.text.strip() is u'':
                continue
            tag_contents = tag.contents
            # for each tag on the page...
            j = 0
            while len(tag_contents):
                j += 1
                line = tag_contents.pop(0)
                line_text = line.text.strip()
                if line.name is None or line.text is None or line.text.strip() is u'':
                    continue
                if len(line.findChildren()) > 1 and self.verbose > 1:
                    print('WARNING: line in page has more than 1 child:\n{0}'.format(line))
                # check for page number, heading, date.
                if j < 10:
                    if utils.is_page_number(line_text):
                        page_number = line.text.strip()
                        if self.verbose > 1:
                            print('Passing over page number: {0}'.format(line))
                        continue
                    if utils.is_page_heading(line_text) or utils.is_page_date(line_text):
                        if self.verbose > 1:
                            print('Passing over page header or date: {0}'.format(line))
                        continue
                if utils.is_page_footer(line_text):
                    if self.verbose > 1:
                        print('Passing over page header or date: {0}'.format(line))
                    continue
                # convert line into Entry object.
                entry_type = self._get_entry_type(line)
                if entry_type is None:
                    if self.verbose > 1:
                        print('Did not find entry_type for line: {0}'.format(line))
                    continue
                speaker_name = self._get_speaker_name(line, entry_type, prev_entry)
                speaker_cleaned, title, appointment = utils.parse_speaker_name(speaker_name)
                text = self._get_text(line)
                if text.strip() is u'':
                    if self.verbose > 1:
                        print('Did not find text for line: {0}'.format(line))
                    continue
                if self.verbose > 1:
                    print('------------------------')
                    print('Line number: %d-%d' % (i, j))
                    print('Speaker name: %s' % speaker_name)
                    print('Entry type: %s' % entry_type)
                    print('Text: %s' % text)
                    print('Page number: %s' % page_number)
                # if entry_type in ['speech_new', 'speech_ctd']:
                #     entry_type = 'speech'
                entry = Entry(entry_type=entry_type, text=text, speaker=speaker_name, page_number=page_number, speaker_cleaned=speaker_cleaned, title=title, appointment=appointment)
                # add Entry to contents_merged.
                contents_entries.append(entry)
        # assigns speaker name to 'speech_ctd'.
        current_speaker = None
        for entry in contents_entries:
            if entry.entry_type == 'speech_new':
                current_speaker = entry.speaker
            elif entry.entry_type in ['header', 'subheader', 'subsubheader']:
                current_speaker = None
            elif entry.entry_type == 'speech_ctd' and current_speaker is not None:
                if entry.speaker is None:
                    entry.speaker = current_speaker
                else:
                    if self.verbose:
                        print('WARNING: continued speech already has a speaker.')
                        print('continued speech: ', entry)
                    if entry.speaker != current_speaker:
                        if self.verbose:
                            print('WARNING: new speech speaker different than continued speech speaker.')
                            print('new speech speaker: ', current_speaker)
                            print('continued speech: ', entry)
        # converts "special_header" to header or subheader depending on position
        # in text.
        contents_entries2 = []
        prev_entry = Entry()
        while len(contents_entries):
            entry = contents_entries.pop(0)
            if entry.entry_type == 'special_header':
                new_entry_type = 'scene'
                # prev_header.text += ' (' + entry.text + ')'
                # if immediately following a header (e.g. "BILLS") or next
                # entry is a subheader, then classify it as a header.
                if prev_entry.entry_type == 'header' or contents_entries[0].entry_type == 'subheader':
                    new_entry_type = 'header'
                # else if prev_entry is a subheader and next_entry is not,
                # then classify as a subheader.
                elif prev_entry.entry_type == 'subheader' and contents_entries[0].entry_type != 'subheader':
                    new_entry_type = 'subheader'
                entry.entry_type = new_entry_type
                # entry.text = '(' + entry.text + ')'
            else:
                prev_entry = entry
            contents_entries2.append(entry)

        # merges entries as appropriate.
        contents_merged = []
        prev_entry = Entry()
        while len(contents_entries2):
            entry = contents_entries2.pop(0)
            if prev_entry.can_merge(entry) and len(contents_merged):
                prev_entry.merge_entries(entry, self.verbose)
                contents_merged[-1] = prev_entry
            else:
                contents_merged.append(entry)
                prev_entry = entry

        # iterates through contents_merged backwards, merging speeches.
        # contents_merged_final = []
        # entry = contents_merged.pop()
        # prev_entry = contents_merged.pop()
        # while len(contents_merged):
        #     prev_prev_entry = contents_merged.pop()
        #     if entry.entry_type in ['speech_new', 'speech_ctd'] and prev_entry.entry_type in ['speech_new', 'speech_ctd'] and entry.speaker is None:
        #         prev_entry.text = prev_entry.text + '\n' + entry.text
        #         entry = copy.deepcopy(prev_entry)
        #         prev_entry = copy.deepcopy(prev_prev_entry)
        #     elif entry.entry_type in ['speech_new', 'speech_ctd'] and prev_entry.entry_type == 'scene' and prev_prev_entry.entry_type in ['speech_new', 'speech_ctd'] and entry.speaker is None:
        #         prev_prev_entry.text = prev_prev_entry.text + '\n' + entry.text
        #         entry = copy.deepcopy(prev_entry)
        #         prev_entry = copy.deepcopy(prev_prev_entry)
        #     else:
        #         contents_merged_final.append(copy.deepcopy(entry))
        #         entry = copy.deepcopy(prev_entry)
        #         prev_entry = copy.deepcopy(prev_prev_entry)
        # contents_merged_final.reverse()

        # fixes issue in which subsubheaders need to be reverted to None
        contents_merged2 = []
        # subsubheader_entry = None
        while len(contents_merged):
            entry = contents_merged.pop(0)
            if entry.entry_type == 'scene' and re.search(r'(clause|schedule|part|title).+(as amended agreed to|agreed to)', entry.text, re.IGNORECASE):
                new_entry = Entry(entry_type='subsubheader', text='no_subsubheading', speaker=None, page_number=entry.page_number)
                contents_merged2.append(new_entry)
                # entry.text = None
            contents_merged2.append(entry)

        # converts "QUORUM" headers to scenes.
        # Note: kludgy that this is not done in self._get_entry_type, but it
        # is easier to check for once the "Q" and "UORUM" text has been
        # concatenated.
        for entry in contents_merged2:
            if entry.entry_type == 'header' and entry.text.lower() in ['quorum', 'quorom']:
                entry.entry_type = 'scene'
                entry.text = '(' + entry.text.lower() + ')'

        # fixes issue in which header text is spaced out
        # (e.g. "EXTENSION OF S ITING H OURS".
        for entry in contents_merged2:
            if entry.entry_type in ('header', 'subheader'):
                entry.text = utils.fix_header_words(entry.text)

        # fixes issue of numbers/punctuation being
        # classified as a header or subheader.
        contents_merged3 = []
        prev_entry = Entry()
        while len(contents_merged2):
            entry = contents_merged2.pop(0)
            if len(contents_merged3) and entry.entry_type in ('header', 'subheader'):
                temp_text = re.sub(r'[\d\s]+', '', utils.rm_punct(entry.text), re.IGNORECASE) if entry.text is not None else None
                if temp_text is not None and not len(temp_text):
                    entry.entry_type = 'speech_ctd'
                    # print('Fixed a numeric header: {0}'.format(entry.text))
                    if prev_entry.can_merge(entry):
                        prev_entry.merge_entries(entry, self.verbose)
                        contents_merged3[-1] = prev_entry
                        continue
            contents_merged3.append(entry)
            prev_entry = entry

        # assigns position to each entry.
        for i, entry in enumerate(contents_merged3):
            entry.position = i

        return contents_merged3


    def _get_entry_type(self, line):
        """Returns the entry type of line (either header, subheader, speech,
        or scene).

        Arguments:
            line : tag object from bs4
                A single element from body.contents.
        """
        # days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if line is None:
            return None
        line_text = line.text.strip()
        if not line_text:
            raise ValueError('line.text.strip() must have length > 0 when passed to self._get_entry_type.')
        punct_test = utils.is_punct(line_text, True)
        if punct_test and line_text is not u'':
            return 'punct'
        text_eq_upper = line_text == line_text.upper()
        b_tags = line.find_all('b')
        i_tags = line.find_all('i')
        italic_text = line.find('i').text.strip() if len(i_tags) else ''
        # "SPECIAL" HEADER TESTS
        special_header_test = bool(
            bool(re.search(r'^first reading|^second reading', line_text, re.IGNORECASE)) and
            len(i_tags)
        )
        # HEADER TESTS
        header_test = bool(
            text_eq_upper and
            len(b_tags) and
            not line_text.endswith('.')
        )

        # header_test = header_test1 or header_test2
        if header_test and len(utils.rm_punct(line_text)) < 5 and re.search(r'\d', line_text):
            prev_entry_type = self._get_entry_type(line.prev_sibling)
            next_entry_type = self._get_entry_type(line.next_sibling)
            header_test = prev_entry_type == 'header' or next_entry_type == 'header'
        # SUBHEADER TESTS
        subheader_test = bool(
            text_eq_upper and
            not len(b_tags) and
            not line_text.endswith('.')
        )
        # subheader_test = subheader_test1 or subheader_test2
        if subheader_test and len(utils.rm_punct(line_text)) < 5 and re.search(r'\d', line_text):
            prev_entry_type = self._get_entry_type(line.prev_sibling)
            next_entry_type = self._get_entry_type(line.next_sibling)
            subheader_test = prev_entry_type == 'subheader' or next_entry_type == 'subheader'

        # SUBSUBHEADER TESTS
        subsubheader_test = bool(
            re.search(r'^clause|^question no|^\(the house resumed\)|^(first|second|third|fourth|fifth|sixth) schedule$', line_text, re.IGNORECASE) and
            len(i_tags)
        )
        # NEW SPEECH TESTS
        new_speech_test = bool(
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
        # CONTINUED SPEECH TESTS
        ctd_speech_test = bool(
            not subsubheader_test and
            (
                not text_eq_upper or
                (
                    text_eq_upper and
                    line_text.endswith('.')
                ) or
                (
                    len(line_text) < 5 and not subheader_test
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
        # ctd_speech_test = ctd_speech_test1 or ctd_speech_test2
        # SCENE TESTS
        scene_test = bool(
            len(i_tags) and
            not subsubheader_test and
            not special_header_test and
            not italic_text.lower() in self.italic_phrases and
            (
                italic_text == line_text or
                (
                    len(italic_text) > 10 and
                    utils.rm_punct(italic_text) == utils.rm_punct(line_text)
                )
            )
        )
        test_results = {
            'special_header': special_header_test,
            'header': header_test,
            'subheader': subheader_test,
            'subsubheader': subsubheader_test,
            'speech_new': new_speech_test,
            'speech_ctd': ctd_speech_test,
            'scene': scene_test
        }
        if sum(test_results.values()) > 1:
            return 'Multiple entry types: {0}'.format(', '.join([k for k, v in test_results.items() if v]))
        # if sum(test_results.values()) == 0:
        #     return None
        for k, v in test_results.items():
            if v:
                return k
        return None

    def _get_speaker_name(self, line, entry_type, prev_entry):
        """Returns a string representing the name of a speaker in a new
        speech.

        Removes the speaker name from the text of line.
        """
        if entry_type == 'speech_new':
            if len(line.find_all('b')) > 1 and self.verbose > 1:
                print('WARNING: more than one <b> tag in line: {0}'.format(line))
            speaker_name = line.find('b').extract().text.strip()
            # speaker_name = b_tags[0].text.strip() if len(b_tags) else line.text.strip()
            # condition 1: b tag text is simply '[name]:'.
            if re.search(':$', speaker_name):
                speaker_name = speaker_name[:-1].strip()
            # condition 2: b tag text is appointment, followed by name in parentheses outside of b tag (e.g. "<b>assistant minister for finance </b>(mr. arap-kirui): I beg to ...")
            line_text = line.text.strip()
            if re.search(r'^\(.{2,}\):', line_text):
                parenth_name, speech = utils.extract_parenth_name(line_text, name_at_begin=True)
                line.string = speech.strip()
                speaker_name = speaker_name + ' ' + parenth_name
        elif entry_type == 'speech_ctd':
            line_text = line.text.strip()
            if re.search(r'^\(.{2,}\):', line_text):
                parenth_name, speech = utils.extract_parenth_name(line_text, name_at_begin=True)
                line.string = speech.strip()
                speaker_name = prev_entry.speaker + ' ' + parenth_name if prev_entry.speaker else parenth_name
                if re.search(':$', speaker_name):
                    speaker_name = speaker_name[:-1].strip()
            else:
                speaker_name = prev_entry.speaker
        else:
            speaker_name = None
        speaker_name = utils.clean_speaker_name(speaker_name)
        return speaker_name

    def _get_text(self, line):
        """Gets text from line, not including speaker name if speaker name
        exists.

        Arguments:
            line : bs4 tag
                a bs4 tag or string.
            entry_type : str
                a string representing the entry type.
            prev_entry : Entry object
                an Entry object, as defined in entry_class.py. Should be the entry previous to line.

        Returns:
            text : string
                the text of the entry.
        """
        # if re.search('^' + re.escape(speaker_name), text):
        #     text = text.replace(speaker_name, '', 1)
        # NOTE TO SELF: this if condition is a kludge in order to prevent speaker name from ending up at beginning of speech when doing layout analysis and when get_merged_contents is called recursively b/c of multiple_speeches. If this works well, may be able to delete "if entry_type == 'speech_new' and i == 0:" line above.
        if line.name:
            text = line.text.strip()
        else:
            text = line.string.strip()
        if re.search('^:', text):
            text = text[1:].strip()
        # text = text.replace('\n', ' ')
        # text = text.encode('utf-8')
        return text

