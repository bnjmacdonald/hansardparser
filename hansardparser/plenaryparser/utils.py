
import os
import numpy as np
from typing import Tuple, Optional
import collections
import datetime
import pytz
import string
import re
import calendar
from hansardparser.plenaryparser.text_utils import is_english_word

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

def is_str_date(s):
    """Tests whether a string (s) can be converted to a datetime object based
    on a format like '%A, %dth %B, %Y'."""
    if date_search(s.strip()):
        return True
    return False

def str_from_date(date):
    if date is None:
        return None
    return date.strftime('%Y-%m-%d')

def convert_str_to_date(s):
    """ Converts a string (s) to a datetime object.
    Returns None if unsuccessful. """
    result = None
    regex_result = date_search(s.strip())
    if regex_result:
        date_fmt = '%Y-%B-%d'
        year = regex_result.groupdict()['year']
        month = regex_result.groupdict()['month']
        day = regex_result.groupdict()['day']
        date_str = '{0}-{1}-{2}'.format(year, month, day)
        result = pytz.utc.localize(datetime.datetime.strptime(date_str, date_fmt))
    return result

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

def left_extend(l, l2):
    """left extends a list. While this would be easy to do
    via l2 + l, this method allows the leftextend to be done
    by reference rather than by value."""
    for el in reversed(l2):
        l.insert(0, el)

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

def get_transcript_heading(text):
    """returns the transcript heading. Assumes that is_transcript_heading
    returns True.
    """
    heading_test = re.compile(r'^(NATIONAL ASSEMBLY)*\s*(OFFICIAL REPORT)*\s*$')
    test_text = text.strip()
    return re.sub(r'\s+', ' ', heading_test.search(test_text).group())

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


def parse_speaker_nameV2(s: str) -> Tuple[str, str, str]:
    """Parses speaker title, name, and appointment.
    
    NOTE: Very similar to `parse_speaker_name`.

    Todos:

        TODO: This implementation fails to extract speaker name when it spills
            over a single line. Example::

            `<newspeech>The Assistant Minister for Finance and Planning\n
            (Mr. Odupoy): Bw. Spika, -naomba kujibu...`

            This could be fixed by merging lines prior to extracting speakers.
            But then we run into the problem of how to know when to merge
            two speeches if you haven't extracted speakers yet. One way around this
            would be to compile a list of all appointments, and then check if a
            line matches an appointment.

        TODO: combine with `parse_speaker_name` (will require some refactoring of
            `XmlParser` class).
    """
    if s is None:
        return None, None, None
    title_regex = (r'(?P<title>mr|bw|ms|bi|hon|capt|mrs|dr|prof|gen|maj-gen|maj'
        r'|major|an hon|a hon|eng|engineer|col|rtd|rev|sen|mheshimiwa)')
    # title followed by space and name, then colon.
    # e.g. "Mr. Bett: Thank, you, Mr. Speaker."
    name_regex1 = re.compile(rf'^({title_regex}[\. ]{{1,2}})?'
        r'(?P<name>[\dA-z-\', ]{3,100})$', re.IGNORECASE|re.DOTALL)
    # for cases in which appointment is first, followed by name in parentheses.
    # e.g. "The Minister for Agriculture (Hon. Bett): Thank you, Mr. speaker."
    name_regex2 = re.compile(rf'^(?P<appt>[\dA-z-\',\. ]{{5,}})\(\s{{0,2}}{title_regex}[\. ]{{1,2}}'
        r'(?P<name>[\dA-z-\', ]{3,100})\)\s{0,2}$', re.IGNORECASE|re.DOTALL)
    # name is first, followed by appointment in parentheses.
    # e.g. "Hon. Bett (The Minister for Agriculture): Thank you, Mr. speaker."
    name_regex3 = re.compile(rf'^{title_regex}[\. ]{{1,2}}'
        r'(?P<name>[\dA-z-\', ]{3,100})\((?P<appt>[\dA-z-\',\. ]{5,})\)\s{0,2}$', re.IGNORECASE|re.DOTALL)
    # appointment only.
    name_regex4 = re.compile(rf'^(?P<appt>[\dA-z-\',\. ]{{5,}})\s{{0,2}}$', re.IGNORECASE|re.DOTALL)
    name, title, appt = None, None, None
    result = None
    for regex in [name_regex1, name_regex2, name_regex3, name_regex4]:
        result = regex.search(s)
        # print(result)
        if result is not None:
            name = result.group('name') if 'name' in result.groupdict() else None
            title = result.group('title') if 'title' in result.groupdict() else None
            appt = result.group('appt') if 'appt' in result.groupdict() else None                
            break
    if name is not None:
        if re.search(r'speaker|minister|members', name, re.IGNORECASE):
            if appt is not None:
                appt += f' ({name})'  # KLUDGE: when does this actually happen?
            else:
                appt = name
            name = None
    name = clean_speaker_name(name)
    return title, name, appt


def extract_speaker_name(s: str) -> Tuple[str, str]:
    """Extracts speaker name from text.

    Todos:

        TODO: this method still runs into problems when a line starts with a few
            words followed by a colon. Example:: `Wizara: hiyo.`
                
    """
    if s is None:
        return None, None
    title_regex = (r'(?P<title>mr|bw|ms|bi|hon|capt|mrs|dr|prof|gen|maj-gen|maj'
        r'|major|an hon|a hon|eng|engineer|col|rtd|rev|sen|mheshimiwa)')
    regex1 = re.compile(rf'^(?P<name>[\dA-z-\'\(\)\., ]{{3,100}})(?P<segue>:)(?P<text>.+)$', re.IGNORECASE|re.DOTALL)
    # regex2: allows for more segue possibilities, but restricts pre-segue to start
    # with a speaker title.
    regex2 = re.compile(rf'^(?P<name>{title_regex}[\. ]{{1,2}}[A-z-\'\(\)\., ]{{3,40}})'
        r'(?P<segue>:| said| asked| ali)(?P<text>.+)$', re.IGNORECASE|re.DOTALL)
    name = None
    text = s
    for i, regex in enumerate([regex1, regex2]):
        result = regex.search(s)
        # print(result)
        if result is not None:
            name = result.group('name')
            # KLUDGE: strings such as "Mr. Speaker, I said that..." should not have a speaker extracted.
            if i == 1 and ':' not in result.group('segue') and re.search(r'speaker', name, re.IGNORECASE):
                name = None
            if ':' in result.group('segue'):
                text = result.group('text')
            break
    return name, text


def fix_header_words(text):
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
        # if it's a one-letter word not equal to 'a', then add it to next word.
        if len(word_alphanum) == 1 and not re.search(r'.+[{0}]$'.format(re.escape(close_punct)), word):
            if len(words) and not word_alphanum == 'a':  #  and is_english_word(words[0])
                next_word = words.pop(0)
                word += next_word
            # else if word is 'a' and next word is in X or is not an english word, then add to next word.
            elif word_alphanum == 'a' and len(words) and (words[0] in ['mend', 'id', 'rid', 'ward', 'broad'] or not is_english_word(words[0])):
                next_word = words.pop(0)
                word += next_word
            elif len(new_words) and word_alphanum == 'a' and not is_english_word(new_words[-1]):
                new_words[-1] += word
                continue
        new_words.append(word)
    text = ' '.join(new_words)
    text = text.replace(' -', '-').replace('- ', '-')
    text = re.sub('([{0}]) '.format(open_punct), '\g<1>', text)
    text = re.sub(' ([{0}])'.format(close_punct), '\g<1>', text)
    text = re.sub('([{0}])(\w+)'.format(close_punct), '\g<1> \g<2>', text)
    return text

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
