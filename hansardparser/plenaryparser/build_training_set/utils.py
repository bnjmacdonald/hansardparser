
import re

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
