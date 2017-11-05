"""Defines the HansardParser class, which is a meta-class
containing generic methods for parsing an Hansard transcript
into a list of Entry objects, which can than be converted
into a dictonary or Pandas DataFrame using the
hansard_convert.py module.

Notes:

    This implementation was inspired in part by: https://github.com/mysociety/pombola/blob/master/pombola/hansard/kenya_parser.py

    The basic pipeline of the HansardParser class is as follows:
        (a) convert PDF to html/txt (convert_pdf_to_format).
        (b) extract Hansard metadata (process_html_meta).
        (c) for each line/tag, create an entry object (process_html_contents).
        (d) clean/process list of entries (process_html_contents).

Todos:

    TODO: convert sitting start_page to int.
"""

from datetime import datetime
import os
import sys
import time
import traceback

import settings
from hansardparser.plenaryparser.Sitting import Sitting
from hansardparser.plenaryparser import utils
from hansardparser.plenaryparser.convert_hansard import convert_contents

class HansardParser(object):
    """ The HansardParser class contains methods for parsing
    an Hansard transcript.

    Attributs:
        fmt : str. Options: 'txt', 'html'. Represents whether pdf should be
            converted to .txt or .html.
        verbose : bool
            False by default. Set to 1 or 2 if detailed output to console is
            desired.
        parliament_dates : dict like {int -> (datetime, datetime)}
            dictionary of parliaments-date pairs. Gives range of dates for
            each parliament.
        italic_phrases : list of strings
            List containing strings that appear as italic phrases in speeches,
            but which should not be treated as a scene entry_type.
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
        if italic_phrases is None:
            italic_phrases = self.italic_phrases
        self.italic_phrases = italic_phrases
        self.verbose = verbose
        self.soup = None

    def process_transcript(self, file_path, save_soup=False, path=None, rm_whitespace=True, append_meta=True, to_format='df-long'):
        """wrapper for processing a list of transcript bs4 contents.

        Arguments:
            file_path: str. location of transcript to parse.
            save_soup: bool. Default: False. If True, saves soup to disk.
            path: str. Path to save soup to disk.
            rm_whitespace : bool
                True by default. Removes excess whitespace and line breaks.
            append_meta : bool
                True by default. Adds metadata to each entry in contents.
            to_format: str. See self._convert_contents. If None, contents are
                not converted at all (i.e. they are left as a list of Entry
                objects).

        Returns:
            metadata: Sitting object. contains metadata on the processed
                transcript.
            contents: data from processed transcript. If to_format=None, then
                this is a list of processed Entries. If to_format!=None, then
                this is in the format returned by self._convert_contents.
        """
        # try:
        time0 = time.time()
        self.soup = self._convert_pdf(file_path, save_soup, path)
        self._preprocess_soup()
        metadata = self._process_meta()
        contents = self._process_contents(metadata.start_page)
        if append_meta:
            for entry in contents:
                entry.metadata = metadata
        # cleans text (removes whitespace).
        if rm_whitespace:
            for entry in contents:
                entry.text = utils.clean_text(entry.text)
        if to_format is not None:
            contents = self._convert_contents(contents, metadata, to_format=to_format)
        time1 = time.time()
        if self.verbose:
            print('Processed "{0}" in {1:.2f} seconds.'.format(file_path.split('/')[-1], time1 - time0))
        return (metadata, contents)
        # except KeyboardInterrupt:
        #     print('KeyboardInterrupt')
        #     sys.exit(1)
        # except ValueError as e:
        #     print('ERROR: Value Error in file: %s' % (file_path))
        #     print(str(e))
        # except IOError as e:
        #     print('ERROR: IO Error in file: %s' % (file_path))
        #     print(str(e))
        # except IndexError as e:
        #     print('ERROR: Index Error in file: %s' % (file_path))
        #     print(str(e))
        # except RuntimeError as e:
        #     print('ERROR: Runtime Error in file: %s' % (file_path))
        #     print(str(e))
        # except TypeError as e:
        #     print('ERROR: Type Error in file: %s' % (file_path))
        #     print(str(e))
        #     print(traceback.format_exc())
        # except AttributeError as e:
        #     print('ERROR: Attribute Error in file: %s' % (file_path))
        #     print(str(e))
        #     print(traceback.format_exc())


    def _convert_contents(self, contents, metadata, to_format='df-long'):
        """converts list of hansard contents to df-long, df-raw, or dict.

        Arguments:
            contents: list of Entry objects, like returned from
                self._process_contents.
            metadata: Sitting object, like returned from self._process_meta.
            to_format : str
                Desired output format. Either 'dict', 'df-raw', or 'df-long'.
                'df-raw' is a pandas dataframe where each row is an entry.
                This format is more useful for comparing the parsed transcript
                to the original pdf for errors.
                'df-long' is a multi-index pandas dataframe (where header and
                subheader are the indices). This format is more useful for
                analysis since speeches are organized under headers and
                subheaders.
                'dict' is a nested dictionary with the following structure:
                {header: {subheader: [entries as tuples]}}. Not sure what this
                structure will actually be useful for, though it is used to create
                the df-long format.
            save_contents: bool. Whether to save output to disk.
        """
        # NOTE TO SELF: two .remove lines below are kind of a kludge. Should make this cleaner.
        attributes = list(contents[0].__dict__.keys())
        attributes.remove('metadata')
        attributes.remove('speaker_uid')
        contents_conv = convert_contents(contents, metadata, attributes, to_format=to_format, verbose=self.verbose > 0)
        # exports to csv
        # filename = file_path.replace(input_dir,'').replace('/', '_').replace('.pdf', '')[1:]
        # TODO: check if file already exists.
        # if save_contents and len(contents_conv):
        #     hansard_convert.export_contents(filename, contents_conv, output_dir, input_format=_DATA_FORMAT, output_format='hdf')
        return contents_conv

    def _convert_pdf(self, file_path):
        raise NotImplementedError

    def _preprocess_soup(self):
        raise NotImplementedError

    def _process_meta(self, metadata=None, max_check=50):
        """Extracts meta-data from the transcript.

        Arguments:
            contents : list
                a list containing the body.contents of a bs4 object.
            max_check : int representing number of lines to check.

        Returns:
            metadata : Sitting object
                Sitting object as defined in Sitting.py

        TODO:
            * don't need to check all max_check lines. Just check until
                metadata is complete.

        """
        if metadata is None:
            metadata = Sitting()
        # if not self.metadata_exists(contents, max_check):
        #     return metadata
        metadata = self._extract_metadata(metadata, max_check)
        if metadata.date is None:
            print('WARNING: No date found in transcript.')
        # print(metadata)
        # print(contents[0:10])
        return metadata


    def _metadata_exists(self, contents=None):
        """boolean function that returns True if metadata exists in this
        document. """
        raise NotImplementedError

    def _add_to_meta(self, line, metadata):
        """Attempts to add the contents of line_text to the transcript
        metadata.

        Arguments:
            line : bs4.Tag or bs4.NavigableString
                a bs4 object containing text to be added to the metadata.
            metadata : Sitting object
                a Sitting object as defined in sitting_class.py.

        Returns:
            returns 0 if line_text is None or if no update is made, otherwise
            returns 1 once the metadata Sitting object has been updated based
            on line_text.
        """
        raise NotImplementedError



    def _process_contents(self, current_page):
        """Takes in a list of contents and returns a processed list of merged
        contents.

        Arguments:
            current_page : int
                int representing current page number of transcript

        Returns:
            contents_merged : list of Entry objects
                a list of Entry objects representing each entry in the
                transcript. This list is not quite fully processed -- the
                processing steps are finished in process_html_contents().
        """
        raise NotImplementedError

    def _get_entry_type(self, line):
        """Returns the entry type of line (either header, subheader, speech,
        or scene).

        Arguments:
            line : tag object from bs4. A single element from body.contents.
        """
        raise NotImplementedError

    def _get_speaker_name(self, line, entry_type, prev_entry):
        """Returns a string representing the name of a speaker in a new
        speech."""
        raise NotImplementedError

    def _get_text(self, line, speaker, entry_type, prev_entry):
        """Gets text from line, not including speaker name if speaker name
        exists.

        Arguments:
            line : bs4 tag
                a bs4 tag or string.
            speaker : str
                a string representing the speaker name.
            entry_type : str
                a string representing the entry type.
            prev_entry : Entry object
                an Entry object, as defined in entry_class.py. Should be the entry previous to line.

        Returns:
            text: string
                the text of the entry.
        """
        raise NotImplementedError

