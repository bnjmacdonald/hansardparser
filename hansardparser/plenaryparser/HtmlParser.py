""" Defines the KenyaHansardParser subclass.

Parses a pdf Kenya Hansard transcript into a list of Entry objects,
which can than be converted into a dictonary or Pandas DataFrame
using the hansard_convert.py module. Module was initially built
based on April 11th, 2006 transcript.

See super-class (hansard_parser.py) for notes on implementation.
"""

import copy
import subprocess
import re
from bs4 import BeautifulSoup, Tag
import warnings

from plenaryparser.Entry import Entry
from plenaryparser.Sitting import Sitting
from plenaryparser.hansard_parser import HansardParser
from plenaryparser import utils

class HtmlParser(HansardParser):
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

    text_style = 'Times-Roman|TimesNewRomanPSMT'
    italic_style = 'Times-Italic|TimesNewRomanPS-ItalicMT'
    bold_style = 'Times-Bold|TimesNewRomanPS-BoldMT'

    def __init__(self, italic_phrases=None, verbose=0):
        HansardParser.__init__(self, italic_phrases=italic_phrases, verbose=verbose)
        raise NotImplementedError
