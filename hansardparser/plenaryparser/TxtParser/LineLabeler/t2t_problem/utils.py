import re
from typing import Tuple, List

def extract_flatworld_tags(s: str) -> Tuple[str, List[str]]:
    """extracts open and close html/xml Flatworld tags from a string.

    note: the reason we do not use BeautifulSoup for this is that the Flatworld
    tags often have syntax errors (e.g. "header>") that would not be read properly
    by BeautifulSoup. So, instead, we use custom regular expressions to extract
    the tags.

    Flatworld tags include `["<header>", "<newspeech>", "<scene>"]` and minor
    variants (e.g. "<speech>").

    Returns:

        Tuple[str, List[str]]. First element of the tuple is the string with the
            Flatworld tags removed. Second element of the tuple is a list of the
            unique Flatworld strings that were found and removed. If no tags are
            found, an empty list is returned. Tags are listed in alphabetical
            order.

    Example::

        >>> extract_flatworld_tags("<header><b>MOTIONS</b></header>")
        ('<b>MOTIONS</b>', ['header'])
    """
    # extracts the unique Flatworld tags found.
    inner_regex = r'[/\s]{0,3}(new[\-\s]?speech|speech|sub[\-\s]?header|header|scene|district)[/\s]{0,3}'
    regex = re.compile(rf'<({inner_regex})>|<({inner_regex})|({inner_regex})>', flags=re.IGNORECASE)
    result = re.findall(regex, s)
    tags = set({})
    if len(result):
        for taglist in result:
            for tag in taglist:
                if tag is not None:
                    tag = re.sub(r'[</>\-\s]', '', tag).lower().strip()
                    if len(tag):
                        tags.add(tag)
    tags = sorted(tags)
    # removes the Flatworld tags from the string.
    s = re.sub(regex, '', s)
    # if verbosity > 1 and re.search(r'(<[/ \w]{3,})|([/ \w]{3,}>)', s):
    #     warnings.warn(f'angle bracket exists in line: {s}')
    return s, tags
