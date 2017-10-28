"""utility methods for processing text using nltk.
"""

import string
from nltk.corpus import words

try:
    _ENGLISH_WORDS = set(words.words())
    _ENGLISH_WORDS.update([
        'hon',
        'hon.',
        'mr',
        'mr.',
        'prof',
        'prof.',
        'dr',
        'dr.',
    ])
except:
    _ENGLISH_WORDS = set([])
    print('WARNING: corpora/words not found. Need to run nltk.download()')

def is_english_word(word):
    """returns True if word is an english word."""
    return word in _ENGLISH_WORDS
