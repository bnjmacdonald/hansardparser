
import re
from typing import List, Tuple, Optional

from utils import extract_flatworld_tags, clean_speaker_name


class RuleSpeakerParser(object):

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity


    def extract_speaker_names(self,
                              lines: List[str],
                              labels: List[str]
                              ) -> Tuple[List[str],
                                         List[Tuple[str, str, str]],
                                         List[str]]:
        """Extracts the speaker name from the beginning of each line.

        Extracts the speaker name from the beginning of each line. Only
        extracts speaker names where `label[i] == 'speech'`.

        Returns:

            speaker_names, texts: Tuple[List[str], List[str]].

                speaker_names: List[str]. List of speaker names, of same length
                    as `labels`. If `label[i] != 'speech'` or no speaker name
                    is found, then `speaker_name[i] = None`.

                parsed_speaker_names: List[Tuple[str, str, str]]. List of parsed
                    names. If an input speaker name is None, the parsed name will
                    be `(None, None, None)`.

                texts: List[str]. Lines of text after speaker name has been
                    extracted.
        """
        speaker_names = []
        texts = []
        for i, line in enumerate(lines):
            speaker_name = None
            text, _ = extract_flatworld_tags(line) # KLUDGE: ...
            if labels[i] == 'speech':
                speaker_name, text = self._extract_speaker_name(text)
                # speaker_cleaned, title, appointment = utils.parse_speaker_name(speaker_name)
            speaker_names.append(speaker_name)
            texts.append(text)
            # prev_speaker = speaker_name
        parsed_speaker_names = self._parse_speaker_names(speaker_names)
        return speaker_names, parsed_speaker_names, texts


    def _extract_speaker_name(self, s: str) -> Tuple[str, str]:
        """Extracts speaker name from a line of text.

        Todos:

            TODO: this method still runs into problems when a line starts with a few
                words followed by a colon. Example:: `Wizara: hiyo.`
                    
        """
        if s is None:
            return None, None
        s = s.strip()
        title_regex = (r'(?P<title>mr|bw|ms|bi|hon|capt|mrs|dr|prof|gen|maj-gen|maj'
            r'|major|an hon|a hon|eng|engineer|col|rtd|rev|sen|mheshimiwa)')
        regex1 = re.compile(rf'^(?P<name>[\dA-z-\'\(\)\., ]{{3,100}})(?P<segue>:)(?P<text>.+)$', re.IGNORECASE|re.DOTALL)
        # regex2: allows for more segue possibilities, but restricts pre-segue to start
        # with a speaker title.
        regex2 = re.compile(rf'^(?P<name>{title_regex}[\. ]{{1,2}}[A-z-\'\(\)\., ]{{3,40}})'
            r'(?P<segue>:| said| asked| ali| to ask)(?P<text>.+)$', re.IGNORECASE|re.DOTALL)
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


    def _parse_speaker_names(self, speaker_names: List[Optional[str]]) -> List[Tuple[str, str, str]]:
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
