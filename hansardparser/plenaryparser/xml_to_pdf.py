
import os
from unicodedata import normalize
import subprocess
from bs4 import BeautifulSoup

from hansardparser import settings

def convert(file_path, save_soup=False, path=None, verbose=0):
    """converts pdf to xml. Wrapper to pdftohtml library. 

    Arguments:
        file_path: str. location of transcript to parse.
        save_soup: bool. Default: False. If True, saves soup to disk.
        path: str. Path to save soup to disk.
    """
    
    # command = "python ~/myapps/bin/pdf2txt.py -n -t %(output_type)s '%(file_path)s'" % locals()  # -c %(codec)s 
    command = 'pdftohtml -xml -stdout "{0}"'.format(file_path)  # 
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = p.stdout.read()
    error = p.stderr.read()
    if error:
        # NOTE: kludge for optional content group list error. This seems to be something that can be ignored.
        if not error.startswith(b'Syntax Error: Expected the optional content group list'):
            print('Command: {0}'.format(command))
            raise Exception(error)
    # print out
    # result = ''.join(out)
    # out = 'úäô§abc<>'
    out = out.decode('utf-8')  # windows-1252
    out = normalize('NFKD', out).encode('ASCII', 'ignore')
    # out = out.encode('ascii', 'ignore')
    if b'|' in out:
        out = out.replace(b'|', b'/')
        if verbose:
            print('WARNING: Found a "|" in this document. Replaced with "/".')
    soup = BeautifulSoup(out, 'xml')
    if save_soup:
        if path is None:
            path = os.path.join(settings.DATA_ROOT, 'temp', 'parsed_xml')
        if not os.path.exists(path):
            os.makedirs(path)
        fname = file_path.split('/')[-1].replace('.pdf', '.xml')
        if verbose > 1:
            print('saving soup to {0}'.format(path))
        if os.path.exists(os.path.join(path, fname)) and verbose:
            print('WARNING: overwriting file {0}'.format(os.path.join(path, fname)))
        with open(os.path.join(path, fname), 'wb') as f:
            f.write(out)
    return soup