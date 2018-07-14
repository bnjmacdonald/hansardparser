
import os
import re
import warnings
import shutil
from typing import List, Tuple
import numpy as np
import pandas as pd
from unidecode import unidecode


def insert_xml_tag_whitespace(s: str) -> str:
    """inserts whitespace between an xml tag and text.

    Examples::

        >>> s = '<i>hello</i>'
        >>> insert_xml_tag_whitespace(s)
        ' <i> hello </i> '
        >>> s = 'my name is <b>bob</b>'
        >>> insert_xml_tag_whitespace(s)
        'my name is  <b> bob </b> '
    """
    s2 = re.sub(r'<', ' <', s)
    s2 = re.sub(r'>', '> ', s2)
    return s2


def str2ascii_safe(s):
    """converts string to ascii.
    """
    if pd.isnull(s):
        return None
    return unidecode(s)

