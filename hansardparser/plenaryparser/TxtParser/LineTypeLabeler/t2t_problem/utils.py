import re
import numpy as np
import pandas as pd
from typing import Tuple, List
import chardet
from unicodedata import normalize

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


def get_line_context(lines: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        lines = lines.copy()
        lines.sort_values(by=['year', 'file', 'line'], inplace=True)
        lines.reset_index(inplace=True, drop=True)
        contexts = []
        for nm, gp in lines.groupby(['year', 'file']):
            for i, (_, line) in enumerate(gp.iterrows()):
                prev_text = '\n'.join(gp.iloc[i-n:i,]['text'].values)
                next_text = '\n'.join(gp.iloc[i+1:i+1+n,]['text'].values)
                # line['prev_context'] = prev_line_text
                # line['next_context'] = next_line_text
                contexts.append(nm + (line['line'], prev_text, next_text))
        contexts = pd.DataFrame(contexts, columns=['year', 'file', 'line', 'prev_context', 'next_context'])
        return contexts


def normalize_text(text):
    if isinstance(text, bytes):
        encoding = chardet.detect(text)['encoding']
        text = text.decode(encoding)
    # convert to ascii.
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text


def split_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """randomly splits lines.

    Used for constructing more training data.
    """
    lines.sort_values(by=['year', 'file', 'line'], inplace=True)
    lines_split = []
    for nm, gp in lines.groupby(['year', 'file']):
        j = 0
        for _, row in gp.iterrows():
            text = row['text']
            label = row['label']
            split = np.random.choice([False, True])
            if split and len(text) > 1:
                split_idx = np.random.randint(low=1, high=len(text))
                new_line0 = {'text': text[:split_idx], 'label': label, 'line': j, 'year': nm[0], 'file': nm[1]}
                j += 1
                new_line1 = {'text': text[split_idx:], 'label': label, 'line': j, 'year': nm[0], 'file': nm[1]}
                j += 1
                lines_split.append(new_line0)
                lines_split.append(new_line1)
            else:
                new_line0 = {'text': text, 'label': label, 'line': j, 'year': nm[0], 'file': nm[1]}
                lines_split.append(new_line0)
                j += 1
    lines_split = pd.DataFrame(lines_split)
    lines_split['line'] += 1
    return lines_split
