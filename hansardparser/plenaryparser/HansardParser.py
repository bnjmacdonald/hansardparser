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

import warnings
import time
from typing import Union, Optional, List
from bs4 import Tag

from hansardparser.plenaryparser.models.Sitting import Sitting
from hansardparser.plenaryparser.models.Entry import Entry
from hansardparser.plenaryparser import utils
from hansardparser.plenaryparser.convert_hansard import convert_contents


class HansardParser(object):
    """The HansardParser class contains methods for parsing an Hansard transcript.

    This is an astract class, that is not meant to be instantiated directly. Use
    `TxtParser` or `HtmlParser` instead.

    Attributes:
        
        italic_phrases: Optional[List[str]]. List containing strings that appear
            as italic phrases in speeches, but which should not be treated as a
            'scene' label.

        LineLabeler: object. Instance of a class with a `label_lines` method
            that receives a list of lines as a single argument and returns the
            label of each line.
        
        SpeakerParser: object. Instance of a class with a `extract_speaker_names`
            method that receives a list of lines as a single argument and returns
            a list of tuples where each tuple contains the speaker name, parsed
            speaker name, and remaining text of the line.

        verbosity: int = 0. Set to 1 or 2 if detailed output to console is
            desired.
    """

    italic_phrases = [
        'ad hoc',
        'animal farm.',
        'askaris',
        'boda boda',
        'bona fide',
        'dukawallas',
        'El Nino',
        'El-Nino',
        'et cetera',
        'et cetera',
        'et cetera',
        'facebook',
        'gazette',
        'Harambee,',
        'Harambee',
        'Jua Kali',
        'jua kali',
        'Kaya Tiwi',
        'Kaya',
        'Kayas',
        'kazi kwa vijana',
        'Kenya Times',
        'kwa vijana',
        'locus standi',
        'matatu',
        'matatus',
        'Mau Mau',
        'Medusa',
        'moranism',
        'morans',
        'Mukurwe wa Nyagathanga',
        'mundu khu mundu',
        'mungiki.',
        'mungiki',
        'mwananchi',
        'Mzungu',
        'Njuri Ncheke',
        'pangas',
        'persona non grata',
        'posteriori',
        'simba',
        'taliban',
        'vice versa.',
        'vice versa',
        'vide',
        'vijana,',
        'vijana',
        'vipande',
        'vis-a-vis',
        'wananchi',
        'wazungu',
        'whatsapp',
    ]


    def __init__(self,
                 LineLabeler: object,
                 SpeakerParser: object,
                 italic_phrases: Optional[List[str]] = None,
                 verbosity: int = 0):
        if italic_phrases is not None:
            self.italic_phrases = italic_phrases
        self.LineLabeler = LineLabeler
        self.SpeakerParser = SpeakerParser
        self.verbosity = verbosity
        self.soup = None


    def process_transcript(self, file_path, save_soup=False, path=None, to_format='df-long'):
        """wrapper for processing a list of transcript bs4 contents.

        Arguments:
            file_path: str. location of transcript to parse.
            save_soup: bool. Default: False. If True, saves soup to disk.
            path: str. Path to save soup to disk.
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
        if to_format is not None:
            contents = self._convert_contents(contents, to_format=to_format)
        time1 = time.time()
        if self.verbosity > 0:
            print(f'Processed "{file_path.split("/")[-1]}" in {time1 - time0:.2f} seconds.')
        return (metadata, contents)


    def _preprocess_soup(self):
        raise NotImplementedError


    def _convert_pdf(self, file_path, *args, **kwargs):
        raise NotImplementedError


    def _process_meta(self, metadata: Sitting = None, max_check: int = 50):
        """Extracts meta-data from the transcript.

        Arguments:

            contents: list. a list containing the body.contents of a bs4 object.
            
            max_check: int. Integer representing number of lines to check.

        Returns:
            metadata : Sitting object
                Sitting object as defined in Sitting.py

        Todos:

            TODO: don't need to check all max_check lines. Just check until
                metadata is complete.

        """
        if metadata is None:
            metadata = Sitting()
        # if not self.metadata_exists(contents, max_check):
        #     return metadata
        metadata = self._extract_metadata(metadata, max_check)
        if metadata.date is None:
            warnings.warn('No date found in transcript.', RuntimeWarning)
        # print(metadata)
        # print(contents[0:10])
        return metadata


    def _extract_metadata(self, metadata: Sitting = None, max_check: int = 50) -> Sitting:
        """extracts metadata from the initial lines in contents.
        
        Must be implemented by child class.
        """
        raise NotImplementedError('Must be implemented by child class.')


    def _process_contents(self, current_page: int) -> List[Entry]:
        """Takes in a list of contents and returns a processed list of merged
        contents.

        Must be implemented by child class.

        Arguments:

            current_page : int. Integer representing current page number of transcript.

        Returns:

            contents_merged : list of Entry objects. a list of Entry objects
                representing each entry in the transcript.
        """
        raise NotImplementedError('Must be implemented by child class.')


    def _convert_contents(self, contents, to_format='df-long'):
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
        contents_conv = convert_contents(contents, attributes, to_format=to_format, verbosity=self.verbosity > 0)
        # exports to csv
        # filename = file_path.replace(input_dir,'').replace('/', '_').replace('.pdf', '')[1:]
        # TODO: check if file already exists.
        # if save_contents and len(contents_conv):
        #     hansard_convert.export_contents(filename, contents_conv, output_dir, input_format=_DATA_FORMAT, output_format='hdf')
        return contents_conv
