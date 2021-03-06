"""Defines the Entry class used in hansard_parser.py.

Each Entry object represents a single entry from a transcript, which can either be a heading, subheading, scene, or speech.

"""

# from abc import ABCMeta, abstractmethod
import re
import numpy as np

class Entry(object):
    """An entry in a Hansard transcript.

    Attributes:
        entry_type: str
            One of X entry types: 'page_number', 'page_heading', 'page_date',
            'header', 'subheader', 'subsubheader', speech_new', 'speech_ctd', 'scene'
        text: str
            text of the entry.
        speaker: str
            name of the speaker. None is no speaker.
        page_number: int
            page number of the entry.
    """

    # __metaclass__ = ABCMeta

    def __init__(self,
                 entry_type=None,
                 text=None,
                 speaker=None,
                 speaker_cleaned=None,
                 page_number=None,
                 title=None,
                 appointment=None):
        # NOTE TO SELF: attributes to add:
        #   speaker ID
        self.entry_type = entry_type
        self.text = text
        self.speaker = speaker
        self.speaker_cleaned = speaker_cleaned
        self.page_number = page_number
        self.title = title
        self.appointment = appointment

    def __str__(self):
        return str(self.__dict__)
        # 'Position: ' +  str(self.position) + \
        #     '\nPage number: ' +  str(self.page_number) + \
        #     '\nEntry type: ' + str(self.entry_type) + \
        #     '\nSpeaker: ' + str(self.speaker) + \
        #     '\nText: ' +  str(self.text) + \

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def can_merge(self, next_entry):
        """Returns True if two entries can be merged together.

        next_entry must come after self.

        NOTE TO SELF: may also want to check if next_entry comes immediately after self.

        NOTE TO SELF: will need to deal with problems where same speaker is spelled somewhat differently.

        NOTE TO SELF: add flexibility to this function for fuzzy matching.

        Todos:

            TODO: deal with problem where speech separated by a scene should still
                have the speaker carry forward.
        """
        if next_entry.entry_type == 'punct':
            return True
        both_speeches = (self.entry_type and self.entry_type.startswith('speech_') and \
                         next_entry.entry_type and next_entry.entry_type.startswith('speech_'))
        if (self.entry_type == next_entry.entry_type) or both_speeches:
            if self.entry_type in ['speech', 'speech_new', 'speech_ctd']:
                if self.speaker == next_entry.speaker:
                    return True
                if next_entry.speaker is None:
                    return True
                # for attr in ['speaker', 'speaker_cleaned', 'appointment']:
                #     if self.__getattribute__(attr) is not None:
                #         if self.__getattribute__(attr) == next_entry.__getattribute__(attr):
                #             return True
                #         else:
                #             return False
                # return False

                # NOTE TO SELF: these elif conditions will set up the fuzzy matching.
                # if exactly the same speaker, then can be merged.
                # if self.speaker_cleaned == next_entry.speaker_cleaned:
                    # return True
                # elif length is long (i.e. greater than 15) and first 15 characters match, then it is a match.
                # NOTE: this captures situations like "Minister of X" which may or may not end with the name in parentheses.
                # if self.speaker is not None and len(self.speaker) > 20 and self.speaker[:14] == next_entry.speaker[:14]:
                    # return True
                    # this is the most basic fuzzy matching condition. A length of 20 is arbitrary.
            elif self.entry_type == 'scene':
                if not self.text.endswith(')'):
                    return True
            elif self.entry_type == 'subheader':
                if not (re.search(r"bill$|b ill$", self.text.strip(), re.IGNORECASE) and re.search(r'^t', next_entry.text, re.IGNORECASE)):
                    return True
            else:
                return True
        return False


    def merge_entries(self, next_entry, verbose=0):
        """Merges two entries.

        The second argument is an entry that comes AFTER the first entry.
        """
        # NOTE TO SELF: anything else to merge besides text?
        # if self.text is None:
        #     self.text = ''
        # if next_entry.text is None:
        #     next_entry.text = ''
        if not self.can_merge(next_entry):
            raise RuntimeError('Entries cannot be merged, perhaps because they do not have the same entry_type or same speaker.')
        # if self.entry_type == 'subheader' and len(self.text.strip()) == 1:
        #     self.text = self.text + next_entry.text
        self.text = self.text + ' ' + next_entry.text
        # NOTE TO SELF: need to make this fuzzy matching more rigorous. Also need
        # to better coordinate it with can_merge. Issue is that sometimes speaker
        # from first entry is preferred, whereas other times speaker from
        # next_entry is preferred.
        if next_entry.speaker is None:
            next_entry.speaker = self.speaker
            next_entry.speaker_cleaned = self.speaker_cleaned
            next_entry.appointment = self.appointment
            next_entry.title = self.title
        both_speeches = (self.entry_type and self.entry_type.startswith('speech_') and \
                         next_entry.entry_type and next_entry.entry_type.startswith('speech_'))
        if both_speeches and self.speaker != next_entry.speaker:
            raise RuntimeError(f'Speakers are not equal: {self.speaker} != {next_entry.speaker}. {self.__dict__}, {next_entry.__dict__}')
            # speaker_len = len(self.speaker) if self.speaker else 0
            # next_speaker_len = len(next_entry.speaker) if next_entry.speaker else 0
            # if speaker_len > next_speaker_len:
            #     pass
            # elif speaker_len < next_speaker_len:
            #     self.speaker = next_entry.speaker
            # elif speaker_len == next_speaker_len:
            #     raise RuntimeError('Entries have same speaker length but are not equal.')
        return None

    # def to_list(self):
    #     """ Returns entry values as a list. """
