import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import chardet
from unicodedata import normalize

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


def span2bio(s: str, start: int, end: int) -> str:
    """constructs a BIO label sequence from string s.

    Example::

        >>> span2bio("example 1", start=0, end=7)
        'BIIIIIIOO'
    """
    targets = ['O'] * len(s)
    if pd.notnull(start) and pd.notnull(end):
        targets[int(start)] = 'B'
        for i in range(int(start) + 1, int(end)):
            targets[i] = 'I'
    targets = ''.join(targets)
    assert len(targets) == len(s)
    return targets


def bio2span(s: str) -> Tuple[Optional[int], Optional[int]]:
    """constructs a span from a BIO label sequence.

    Example::

        >>> bio2span('BIIIIIIOO')
        (0, 7)
    """
    span = (None, None)
    result = re.search(r'[BI]+', s)
    if result:
        span = result.span()
    return span


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
            bio = row['bio']
            split = np.random.choice([False, True])
            if split and len(text) > 1:
                split_idx = np.random.randint(low=1, high=len(text))
                new_line0 = {'text': text[:split_idx], 'bio': bio[:split_idx], 'line': j, 'year': nm[0], 'file': nm[1]}
                j += 1
                new_line1 = {'text': text[split_idx:], 'bio': bio[split_idx:], 'line': j, 'year': nm[0], 'file': nm[1]}
                j += 1
                lines_split.append(new_line0)
                lines_split.append(new_line1)
            else:
                new_line0 = {'text': text, 'bio': bio, 'line': j, 'year': nm[0], 'file': nm[1]}
                lines_split.append(new_line0)
                j += 1
    lines_split = pd.DataFrame(lines_split)
    lines_split['line'] += 1
    return lines_split
