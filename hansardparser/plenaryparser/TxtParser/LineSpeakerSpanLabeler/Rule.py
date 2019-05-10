
import re
import warnings
from typing import List, Tuple, Optional

from utils import extract_flatworld_tags, clean_speaker_name


class Rule(object):

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity


    def label_speaker_spans(self,
                              lines: List[str],
                              types: List[str] = None) -> Tuple[List[str],
                                                         List[Tuple[str, str, str]],
                                                         List[str]]:
        """Assigns BIO labels to each line.
        """
        
        pred_labels = []
        if self.verbosity > 0:
            print(f'predicting speaker name spans for {len(lines)} lines...')
        for i, line in enumerate(lines):
            speaker_name = None
            text, _ = extract_flatworld_tags(line) # KLUDGE: ...
            orig_text = text
            # KLUDGE: if there is a next line and next line starts with a segue
            # (:| said| asked| ali| to ask), then 
            # remove ':' from next line and append it to end of current line.
            # NOTE: since `lines` only includes "speech" lines (i.e. headers have
            # been removed), this kludge will introduce a bug when a header would
            # have fallen inbetween line i and i+1 and line i+1 started with ':'.
            pat = r'^\s*(:|said|asked|ali|to ask)'
            if (i + 1) < len(lines) and re.search(pat, lines[i+1]):
                regex = re.search(rf'({pat})', lines[i+1], flags=re.IGNORECASE)
                text += ' ' + regex.group()
                lines[i+1] = re.sub(pat, '', lines[i+1], flags=re.IGNORECASE)
            pred = self._get_prediction(text)
            pred_labels.append(pred)
        return pred_labels

    def extract_speaker_names(self, lines: List[str], labels: List[str]):
        speaker_names = []
        texts = []
        for i, line in enumerate(lines):
            text, _ = extract_flatworld_tags(line) # KLUDGE: ...
            orig_text = text
            pat = r'^\s*(:|said|asked|ali|to ask)'
            if (i + 1) < len(lines) and re.search(pat, lines[i+1]):
                regex = re.search(rf'({pat})', lines[i+1], flags=re.IGNORECASE)
                text += ' ' + regex.group()
                lines[i+1] = re.sub(pat, '', lines[i+1], flags=re.IGNORECASE)
            speaker_name, text = self._extract_speaker_name(text, labels[i])
            # KLUDGE: if speaker_name is not None and current line starts with '(',
            # then check if previous line is an appointment string (e.g. "The
            # Assistant Minister for..."). If so, then append previous line to 
            # current speaker_name and set prev text and speaker_name to ''.
            # And, then, if the next line ...
            if len(texts) > 0 and speaker_name is not None and re.search(r'^\s*\(', speaker_name):
                if self.verbosity > 1 and speaker_names[-1] is not None:
                    warnings.warn(f'expected previous speaker name to be None, but received {speaker_names[-1]}. Current line: {text}.')
                pat = r'^the.+minister|^assistant.+minister|^minister|^the.+speaker|^temporary.+speaker'
                if re.search(pat, texts[-1], flags=re.IGNORECASE):
                    speaker_name = ' '.join([texts[-1], speaker_name])
                    texts[-1] = ''
                    speaker_names[-1] = None
            # KLUDGE: if current speaker name ends with a parenthesis but is missing
            # an opening parenthesis and prev text contains an opening parenthesis
            # but is missing a closing parenthesis, then 
            if len(texts) > 0 and speaker_name is not None and re.search(r'^[^\(]+\)\s*$', speaker_name):
                if re.search(r'^\s*\([^\)]+$', texts[-1]):
                    speaker_name = ' '.join([texts[-1], speaker_name])
                    texts[-1] = ''
                    speaker_names[-1] = None
            # KLUDGE: if current line's original text (without FlatWorld
            # tags) ends with ':' and next line begins with '-', then
            # current line does not contain a speaker name. 
            if (i + 1) < len(lines) and re.search(r':\s*$', orig_text) and re.search(r'^\s*-', lines[i+1]):
                speaker_name = None
                text = orig_text
            speaker_names.append(speaker_name)
            texts.append(text)
            # prev_speaker = speaker_name
        return speaker_names, texts


    def _get_prediction(self, s: str) -> str:
        """retrieves BIO prediction for each character in string.

        Uses regex to retrieve prediction.
        """
        if s is None:
            return None
        # s = s.strip()
        title_regex = (r'(?P<title>mr|bw|ms|bi|hon|capt|mrs|dr|prof|gen|maj-gen|maj'
            r'|major|an hon|a hon|eng|engineer|col|rtd|rev|sen|mheshimiwa)')
        regex1 = re.compile(rf'^\s*(?P<name>[A-z-\'\(\)\., ]{{3,100}})(?P<segue>:)(?P<text>[^\-]?.*)\s*$', re.IGNORECASE|re.DOTALL)
        # regex2: allows for more segue possibilities, but restricts pre-segue to start
        # with a speaker title.
        regex2 = re.compile(rf'^\s*(?P<name>{title_regex}[\. ]{{1,2}}[A-z-\'\(\)\., ]{{3,40}})'
            r'(?P<segue>:| said| asked| ali| to ask)(?P<text>[^\-]?.*)\s*$', re.IGNORECASE|re.DOTALL)
        pred = None
        for i, regex in enumerate([regex1, regex2]):
            result = regex.search(s)
            if result is not None:
                start = result.span('name')[0]
                end = result.span('name')[1]
                pred = []
                pred = ['O'] * len(s)
                pred[start] = 'B'
                for i in range(start + 1, end):
                    pred[i] = 'I'
                pred = ''.join(pred)
                # old way:
                # name = result.group('name')
                # # KLUDGE: strings such as "Mr. Speaker, I said that..." should not have a speaker extracted.
                # if i == 1 and ':' not in result.group('segue') and re.search(r'speaker', name, re.IGNORECASE):
                #     name = None
                # if ':' in result.group('segue'):
                #     text = result.group('text')
                # break
        if pred is None:
            pred = 'O' * len(s)
        assert len(s) == len(pred)
        return pred


    def _extract_speaker_name(self, s: str, pred: str) -> Tuple[str, str]:
        """Extracts speaker name from a line of text given a string of BIO predictions

        Todos:

            TODO: this method still runs into problems when a line starts with a few
                words followed by a colon. Example:: `Wizara: hiyo.`
                    
        """
        if s is None or pred is None:
            return None, s
        if len(s) != len(pred):
            if len(s) > len(pred):
                pred += 'O' * (len(s) - len(pred))
            if self.verbosity > 1:
                warnings.warn(f'Length of string should equal length of '
                              f'prediction, but {len(s)} != {len(pred)}. '
                              f'String: "{s}". Prediction: "{pred}".')
        speaker_name = ''
        text = ''
        # for each character, add it to `speaker_name` or `text` depending on
        # predicted label.
        for i, c in enumerate(s):
            pred_char = pred[i]
            if pred_char in ['B', 'I']:
                speaker_name += c
            elif pred_char == 'O':
                text += c
            else:
                raise RuntimeError(f'pred_char must be in ["B", "I", "O"], but pred_char={pred_char}.')
        if len(speaker_name) > 0:
            text = re.sub(r'\s*:\s*', '', text)
        if len(speaker_name) == 0:
            speaker_name = None
        return speaker_name, text


    def parse_speaker_names(self, speaker_names: List[Optional[str]]) -> List[Tuple[str, str, str]]:
        """parses each speaker name in list of speaker names.

        Each speaker name is parsed by splitting a name into the speaker's title,
        cleaned name, and appointment.

        Returns:

            parsed_names: List[Tuple[str, str, str]]. List of parsed names.
                If an input speaker name is None, the parsed name will be
                `(None, None, None)`.
        """
        parsed_names = []
        for name in speaker_names:
            parsed_name = self._parse_speaker_name(name)
            parsed_names.append(parsed_name)
        return parsed_names
    

    def _parse_speaker_name(self, s: str) -> Tuple[str, str, str]:
        """Parses speaker title, name, and appointment from a single speaker name.

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

            TODO: move `clean_speaker_name` into this method, which will require
                some refactoring with XmlParser.

            TODO: combine with `parse_speaker_name` (will require some refactoring of
                `XmlParser` class).
        """
        if s is None:
            return None, None, None
        s = s.strip()
        title_regex = (r'(?P<title>mr|bw|ms|bi|hon|capt|mrs|dr|prof|gen|maj-gen|maj'
            r'|major|an hon|a hon|eng|engineer|col|rtd|rev|sen|mheshimiwa)')
        # title followed by space and name, then colon. (may or may not be in
        # parentheses)
        # e.g. "Mr. Bett: Thank, you, Mr. Speaker."
        name_regex1 = re.compile(rf'^([\(\[])?({title_regex}[\. ]{{1,2}})?'
            r'(?P<name>[\dA-z-\', ]{3,100})([\)\]])?$', re.IGNORECASE|re.DOTALL)
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
