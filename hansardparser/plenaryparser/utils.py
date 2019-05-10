
import os
import json
import numpy as np
import warnings
from typing import Tuple, Optional, List, Iterable, Generator, Any
import collections
import datetime
import string
import re
import calendar
from PyPDF2 import PdfFileReader
import string
from nltk.corpus import words
import chardet
from unicodedata import normalize


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
    warnings.warn('Corpora/words not found. Need to run nltk.download().', RuntimeWarning)


def get_file_paths(input_dirs, verbose=0):
    file_paths = []
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            raise RuntimeError('Input must be valid directory. Please enter choice again.')
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                # print(f)
                if f.startswith('.'):
                    if verbose > 1:
                        print('Passing over hidden file: %s' % f)
                    continue
                if '.pdf' not in f:
                    if verbose > 1:
                        print('Passing over %s' % f)
                    continue
                file_paths.append(os.path.join(subdir, f))
        # else:
        #     for f in os.listdir(input_dir):
        #         if len(f) > 3 and f[:4] in years:
        #             file_paths.append('/'.join([input_dir, f]))
    return file_paths


def date_search(s):
    """searches for date in string (s). Returns re.search object if found.
    Returns None otherwise.
    """
    months = '|'.join(calendar.month_name)[1:]
    endings = '|'.join(['th', 'st', 'nd', 'rd'])
    date_regex = re.compile(r'^(?P<weekday>[A-z]{1,12})[\s,]{1,4}(?P<day>\d{1,2})\s{0,2}(%s)[\s,]{1,4}(?P<month>%s)[\s,]{1,4}(?P<year>\d{4})' % (endings, months))
    return re.search(date_regex, s)

def clean_text(s):
    """Removes extra whitespace from a string."""
    # text = text.replace('\n', ' ')
    if s is None:
        return None
    s = re.sub(r'[ \t]+', ' ', s.strip())
    while len(s) and s[0] in [')', ':']:
        s = s[1:]
    if s == '(':
        s = ''
    return s

def is_punct(s, strip=False):
    """Tests whether a string (s) is all punctuation characters, as defined in string.punctuation. If strip is set to True, strip() is first called on the string. strip=False by default."""
    if strip:
        s = s.strip()
    return all([char in string.punctuation for char in s])


def is_page_heading(text):
    """checks if line merely contains page heading information.
    Returns True if so.
    """
    return bool(re.search(r"^PARLI(A)?(M)?E(A)?NTARY( )?(L)?DEBATE(S)?$", text))

def is_page_footer(text):
    """returns True if text is page footer."""
    if text is None:
        return False
    regexes = [
        r'^Disclaimer:\s+The\s+electronic\s+version\s+of\s+the',
        r'Official\s+Hansard\s+Report\s+is\s+for\s+information\s+purposes',
        r'A\s+certified\s+version\s+of\s+this\s+Report\s+can\s+be\s+obtained\s+from\s+the\s+Hansard\s+Editor\.$'
    ]
    footer_test = re.compile('|'.join(regexes))
    test_text = text.strip()
    return len(test_text) > 0 and bool(footer_test.search(test_text))


def is_page_date(text):
    """checks if string (text) merely contains page date information.
    Returns True if so.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    if len(text) > 70:
        return False
    days = '|'.join(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    months = 'January|February|March|April|May|June|July|August|September|October|November|December'
    months_abbr = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec'
    endings = '|'.join(['th', 'st', 'nd', 'rd'])
    tests = [
        re.compile(r'^(%s)[\s,]{1,5}\d+$' % (days), re.DOTALL),
        re.compile(r'^(%s|%s)[\s,]{1,5}\d+$' % (months, months_abbr), re.DOTALL),
        re.compile(r'^(%s|%s)\s*(\d{1,2}),*\s*(\d{4})' % (months, months_abbr), re.DOTALL),
        re.compile(r'(%s|%s)\s*(\d{1,2}),*\s*(\d{4})$' % (months, months_abbr), re.DOTALL),
        re.compile(r'\d{,4}\s*(%s),\s*\d{1,2}[A-z]{1,2}\s*(%s|%s),\s*\d{4}' % (days, months, months_abbr)),
        re.compile(r'^(%s)$' % endings, re.DOTALL)
    ]
    if date_search(text):
        return True
    while len(tests):
        test = tests.pop(0)
        if bool(test.search(text)):
            return True
    return False


def is_page_number(text):
    """checks if line merely contains page number information.
    Returns True if so.

    TODO:
        * find out if this if-else is really necessary. Motivation is that
            sometimes a number will appear in a heading, so it would be
            nice to check if the span is regular text. But it's also
            possible that the line is a page number with no span, so the
            if-else allows flexibility here.
    """
    # line_text = line.text.strip()
    # spans = line.find_all('span')
    # if len(spans):
        # page_number_test = line_text.isdigit() and len(line_text) < 5
         # and bool(re.search(self.text_style, spans[0].attrs['style']))
    # else:
    page_number_test = text.isdigit() and len(text) < 5
    return page_number_test


def is_transcript_heading(text):
    """returns True if text is transcript heading.
    """
    if text is None:
        return False
    heading_test = re.compile(r'^\s*(NATIONAL ASSEMBLY)*\s*(OFFICIAL REPORT)*\s*$')
    test_text = text.strip()
    return len(test_text) > 0 and bool(heading_test.search(test_text))


def extract_parenth_name(text, name_at_begin=True):
    """Extracts a name in parentheses from the beginning of a speech.

    Returns:
        
        the name and the following text.
    
    Example::

        >>> extract_parenth_name('Hon. Mwakileo (The Minister for Agriculture)', False)
        ('The Minister for Agriculture', 'Hon. Mwakileo')
    """
    if name_at_begin:
        parenth_reg = re.compile(r'^\s*(?P<in_parenth>\(.+\))\s*:\s*(?P<out_parenth>.*)', re.DOTALL)
    else:
        parenth_reg = re.compile(r'^\s*(?P<out_parenth>.*)\s*\((?P<in_parenth>.+)\)', re.DOTALL)
    result = parenth_reg.match(text)
    if result is None:
        return text, text
        # NOTE TO SELF: kludge here. Not sure what cases result in None. Most likely due to clean_speaker, but not sure how best to deal with None cases here. Come back to this later.
    in_parenth = result.group('in_parenth').lstrip()
    out_parenth = result.group('out_parenth').strip()
    return (in_parenth, out_parenth)


def clean_speaker_name(name):
    """Cleans speaker name so that it can be matched in name dict.
    """
    if name is None:
        return None
    speaker_name = re.sub(r'\s+', ' ', name)
    speaker_name = rm_wrapping_punct(speaker_name)
    return speaker_name


def parse_speaker_name(name):
    """decomposes speaker name into "speaker_cleaned", "title", and
    "appointment".

    Todos:

        TODO: this has been deprecated for TxtParser. Check if I can also deprecate
            it for XML parser.
    """
    if name is None:
        return None, None, None
    name = clean_speaker_name(name).lower()
    title = None
    appt = None
    # extracts name from parenthesis (if applicable)
    if '(' in name:
        name, appt = extract_parenth_name(name, name_at_begin=False)
    # removes punctuation.
    # NOTE TO SELF: may want to remove additional punctuation
    name = rm_punct(name)
    appt = rm_punct(appt)
    # removes titles.
    reg_title = re.compile(r'^\s*(?P<title>mr |bw |ms |bi |hon |capt |mrs |dr '
        r'|prof |gen |maj-gen |maj |major |an hon|a hon|eng |engineer |col |rtd '
        r'|rev |sen |mheshimiwa)(?P<name>.+)', re.IGNORECASE | re.DOTALL)
    matches = reg_title.search(name)
    if matches is not None:
        name = matches.group('name').strip()
        title = matches.group('title').strip()
    # NOTE TO SELF: "if name is not None" is a kludge.
    if name is not None:
        if 'speaker' in name:
            appt = name
            name = None
    if name is not None:
        if 'minister' in name:
            appt = name
            name = None
    if name is not None:
        if 'members' in name:
            appt = name
            name = None
    name = clean_text(name)
    appt = clean_text(appt)
    title = clean_text(title)
    # entry.speaker_cleaned = name
    # entry.title = title
    # entry.appointment = appt
    return name, title, appt


def fix_header_words(text):
    assert text == text.lower()
    if text is None:
        return text
    open_punct = re.escape('([{')
    close_punct = re.escape('!),.:;?]}')
    # other_punct = '"#$%\'*+-/<=>@\\^_`|~'
    text = re.sub(r"(.) ' ([A-z])([A-z]) ([A-z]+)", r"\1'\2 \3\4", text)
    words = re.split(r'\s+', text.lower())
    new_words = []
    while len(words):
        word = words.pop(0)
        word_alphanum = re.sub(r'\W', '', word)
        # for debugging purposes:
        # print('----')
        # print(word, word_alphanum)
        # print(new_words)
        # print(words)
        # if word is one of these pieces of punctuation, then add it to the previous word.
        if len(new_words) and word in ['\'', '-']:
            new_words[-1] += word
            continue
        # if it's a one-letter word and does not end in closed punctuation...
        elif len(word_alphanum) == 1 and not re.search(r'.+[{0}]$'.format(re.escape(close_punct)), word):
            # if previous word ends with an apostrophe and current word is an 's', then add 's' to the previous word.
            if len(new_words) and re.search(r'\'$', new_words[-1]) and word_alphanum == 's':
                new_words[-1] += word
                continue
            # if it's a one-letter word not equal to 'a', then add it to next word.
            if len(words) and not word_alphanum == 'a':  #  and is_english_word(words[0])
                next_word = words.pop(0)
                word += next_word
            # else if word is 'a' and next word is in X or is not an english word, then add to next word.
            elif word_alphanum == 'a' and len(words) and (words[0] in ['mend', 'id', 'rid', 'ward', 'broad'] or not is_english_word(words[0])):
                next_word = words.pop(0)
                word += next_word
            # else if word is 'a' and previous word is not an english word, add 'a' to the previous word.
            elif len(new_words) and word_alphanum == 'a' and not is_english_word(new_words[-1]):
                new_words[-1] += word
                continue
        new_words.append(word)
    text = re.sub(r'\s+', ' ', ' '.join(new_words)).strip()
    text = text.replace(' -', '-').replace('- ', '-')
    text = re.sub('([{0}]) '.format(open_punct), '\g<1>', text)
    text = re.sub(' ([{0}])'.format(close_punct), '\g<1>', text)
    text = re.sub('([{0}])(\w+)'.format(close_punct), '\g<1> \g<2>', text)
    return text


def is_english_word(word):
    """returns True if word is an english word."""
    return word in _ENGLISH_WORDS


def rm_wrapping_punct(s):
    """removes punctuation "wrapping" a string. String must start and end
    with a single punctuation.

    Examples::
        "(Mr. Kariuki)" => "Mr. Kariuki"
        "[Some text." => "Some text"
        "()" => ""
    """
    if s is None:
        return None
    return re.sub(r'^[{0}](.*)[{1}]$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '\g<1>', s)


def rm_punct(s):
    """removes all punctuation from a string."""
    if s is None:
        return None
    return re.sub(r'[{0}]'.format(re.escape(string.punctuation)), '', s)


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
    inner_regex = r'[/\s]{0,3}(new[\-\s]?spe{1,2}ch|spe{1,2}ch|sub[\-\s]?heade?r|heade?r|scene?|district)[/\s]{0,3}'
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


def pdf2str(filepath_or_buffer, page_sep: str = '\n\n') -> str:
    """converts a pdf file to string.
    """
    assert isinstance(page_sep, str) or isinstance(page_sep, bytes)
    if isinstance(filepath_or_buffer, str):
        f = open(filepath_or_buffer, 'rb')
    else:
        f = filepath_or_buffer
    try:
        pdf = PdfFileReader(f)
        text = []
        n_pages = pdf.getNumPages()
        for i in range(n_pages):
            page = pdf.getPage(i)
            page_text = page.extractText()
            # page_text = re.sub(r'(\S)\n(\S)', r'\1\2', page_text)
            # page_text = re.sub(r'(\S) ?\n ?(\S)', r'\1 \2', page_text)
            # page_text = re.sub(r'(\s*)?\n(\s*)?', '\n', page_text)
            text.append(page_text.strip())
        assert len(text) == n_pages
        text = page_sep.join(text)
    finally:
        if isinstance(filepath_or_buffer, str):
            f.close()
    return text


def batchify(iterable: Iterable[Any], batch_size: int = 1000) -> Generator[Iterable[Any], None, None]:
    """yields batches of an iterable in batches of size n.

    Divides an iterable into batches of size n, yielding one batch at a time.

    Arguments:

        iterable: Iterable[Any]. Iterable (list, np.array, ...) to be split into
            batches. Must be an iterable that can be sliced (e.g. `iterable[0:5]`)
            and must have a length.

        batch_size: int = 1000. Size of each batch.

    Yields:

        Iterable[Any]. Batch from iterable.

    Example::

        >>> data = list(range(0,11))
        >>> for batch in batchify(data, 3):
        >>>     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9, 10]
    """
    assert isinstance(batch_size, int), f'batch_size must be int, but received {type(batch_size)}'
    assert batch_size > 0, f'batch_size must be greater than 0, but received {batch_size}'
    l = len(iterable)
    for i in range(0, l, batch_size):
        yield iterable[i:min(i + batch_size, l)]


def normalize_text(text):
    if isinstance(text, bytes):
        encoding = chardet.detect(text)['encoding']
        text = text.decode(encoding)
    # convert to ascii.
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text


def get_filetype(fname: str) -> str:
    return fname.split('.')[-1].lower()
