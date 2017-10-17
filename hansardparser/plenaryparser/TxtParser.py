"""Defines the KenyaHansardParser subclass. 

Parses a pdf Kenya Hansard transcript into a list of Entry objects, 
which can than be converted into a dictonary or Pandas DataFrame 
using the hansard_convert.py module. Module was initially built
based on April 11th, 2006 transcript. 

See super-class (hansard_parser.py) for notes on implementation.
"""

from hansardparser.Entry import Entry
from hansardparser.Sitting import Sitting
from hansardparser.HansardParser import HansardParser
from hansardparser import utils
from rawdatamgr.utils import get_parliament_num
import re
import warnings
import copy
import subprocess

class TxtParser(HansardParser):
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
        """converts pdf to txt. Wrapper to pdftotext library. 

        Arguments:
            file_path: str. location of transcript to parse.
            save_soup: bool. Default: False. If True, saves soup to disk.
            path: str. Path to save soup to disk.
        """
        
        # command = "python ~/myapps/bin/pdf2txt.py -n -t %(output_type)s '%(file_path)s'" % locals()  # -c %(codec)s 
        command = 'pdftotext "{0}" -'.format(file_path)  # 
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out = p.stdout.read()
        error = p.stderr.read()
        if error:
            raise Exception(error)
        # print out
        # result = ''.join(out)
        out = out.decode('utf-8')  # windows-1252
        out = out.encode('ascii', 'ignore')
        if '|' in out:
            print('WARNING: Found a "|" in this document. Replaced with "/".')
            out = out.replace('|', '/')

        if save_soup:
            if path is None:
                path = os.path.join(settings.BASE_DIR, 'hansardparser', 'output', 'parsed_txt')
            if not os.path.exists(path):
                os.makedirs(path)
            fname = file_path.split('/')[-1].replace('.pdf', '.txt')
            if os.path.exists(os.path.join(path, fname)):
                print('WARNING: overwriting file {0}'.format(os.path.join(path, fname)))
            with open(os.path.join(path, fname), 'w') as f:
                f.write(out)
        return out
