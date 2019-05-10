"""tests for `RuleLineLabeler`.

Todos:

    TODO: get rid of `LINES` and replace tests with more targeted/specific tests
        that test specific functionality of each RuleLineLabeler method.
"""

import unittest
from hansardparser.plenaryparser.TxtParser.LineLabeler import RuleLineLabeler

LINES = [
    # headers without tags
    ('THE NATIONAL ASSEMBLY',                                           'header'),
    ('ORAL ANSWERS TO QUESTIONS',                                       'header'),
    ('ADOPTION OF PSC R ECOMMENDATIONS ON C HAIR M EMBERS OF IIEC',     'header'),
    # headers with tags.
    ('<header>CONSTRUCTION OF FUNYULA-BUMALA WATER PROJECT</header>',   'header'),
    ('<header>CONSTRUCTION OF FUNYULA-BUMALA WATER PROJECT',            'header'),
    ('CONSTRUCTION OF FUNYULA-BUMALA WATER PROJECT</header>',           'header'),
    # header without tags.
    ('The Statute Law (Miscellaneous Amendments) Bill',                 'header'),
    # subheaders
    # subheader: all caps, with numbers and punctuation.
    ('436: 1091 SUPPLY, COMMITTEE OF',                                  'subheader'),
    # subheader: mostly caps.
    ('INCREASE or CATTLE RUSTLING IN TAVETA',                           'subheader'),
    ('Vot B.P. PETROL STATION DISMISSAL OF EMPLOYEES',                  'subheader'),
    # subheader: all caps, ending with period.
    ('ISSUE OF CERTIFICATE TO MR. MOSONIK.',                            'subheader'),
    # subsubheaders
    ('<i>Question No. 247</i>',                                         'subsubheader'),
    ('No. 722',                                                         'subsubheader'),
    ('NO. 612',                                                         'subsubheader'),
    ('clause 7',                                                        'subsubheader'),
    ('Clause 6',                                                        'subsubheader'),
    ('Clause. 3',                                                       'subsubheader'),
    # special header
    ('first reading',                                                   'special_header'),
    ('second reading',                                                  'special_header'),
    # new speech with opening tag
    ('<newspeech>Mr. Speaker: Anyone here from the Ministry of',        'speech_new'),
    ('<newspeech>Dr. Kanyarna: Mr. Deputy Speaker, Sir, let me join',   'speech_new'),
    ('<newspeech>The Assistant Minister, Ofifce of the President (Mr ', 'speech_new'),
    ('<Newspeech>MR. MWACHOFI: On a point of order, Mr. Speaker, Sir.', 'speech_new'),
    ('<Newspeech>"MR.. SHIKD30J: On a point of order, Mr. Speaker, Sir.', 'speech_new'),
    ('<Newspeech>THE CHAIRMAN: Hon. Members, we will now come to the part (b) of Order Ho. 8,', 'speech_new'),
    # new speech with closing tag.
    ('Mr. Speaker: Anyone here from the Ministry of</newspeech>',       'speech_new'),
    # new speech without tags
    ('Mr. Speaker: Anyone here from the Ministry of',                   'speech_new'),
    ('Dr. Kanyarna: Mr. Deputy Speaker, Sir, let me join',              'speech_new'),
    ('HON. Members: Which particular areas?',                           'speech_new'),
    # new speech with cut-off speaker
    ('<Newspeech>The Assistant Minister for Agriculture',               'speech_new'),
    ('Maiyani): Mr Acting Speaker, Sir I have been directed \r',        'speech_new'),
    ('Minister for Agriculture:',                                       'speech_new'),
    # new speech in third person
    ('Mr. A.H.O. Momanyi asked the Minister for Agriculture:-',         'speech_new'),
    # continued speech
    ('be communicated to the Agricultural Finance Corporation.',        'speech_ctd'),
    ('(2) The council may place \'on deposit with ',                    'speech_ctd'),
    ('<i>(e) Income Tax</i>',                                           'speech_ctd'),
    ('Clause IIB of the enacted Bill provides that',                    'speech_ctd'),
    # continued speech with end tag.
    (' are complete?</newspeech>\r',                                    'speech_ctd'),
    (' are complete</newspeech>',                                       'speech_ctd'),
    ('second reading of the',                                           'speech_ctd'),
    ('Mr. Speaker, Sir, my last point is: if</newspeech>',              'speech_ctd'),
    # continued speech that begins with what looks like a subsubheader
    ('Clause 2 of the Bill goes further ',                              'speech_ctd'),
    ('clause. I support.',                                              'speech_ctd'),
    ('clause 7, on page 264 of the Bill, you will see that in',         'speech_ctd'),
    ('clause 7B (2).',                                                  'speech_ctd'),
    # continued speech that is a number.
    ('115',                                                             'speech_ctd'),
    ('2947.40',                                                         'speech_ctd'),
    # scene
    ('(Question withdrawn)',                                            'scene'),
    ('(Hon. Okondo laid the papers on the Table)',                      'scene'),
    ('(Laughter)',                                                      'scene'),
    ('(Question of the second part of the amendment, that the words to be inserted in place thereof be inserted, put and\r', 'scene'),
    # scene not fully in parentheses.
    ('A(pplause)',                                                      'scene'),
    ('(Question of the. amendment proposed) (Question. that the word to be inserted he inserted, put and agreed to)', 'scene')
]


class IsHeaderTests(unittest.TestCase):
    
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_header(self):
        for s, expected_label in LINES:
            if expected_label == 'header':
                self.assertTrue(self.labeler._is_header(s))
            else:
                self.assertFalse(self.labeler._is_header(s))


class IsSubheaderTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_subheader(self):
        for s, expected_label in LINES:
            if expected_label == 'subheader':
                self.assertTrue(self.labeler._is_subheader(s))
            else:
                self.assertFalse(self.labeler._is_subheader(s))


class IsSubsubheaderTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_subsubheader(self):
        for s, expected_label in LINES:
            if expected_label == 'subsubheader':
                self.assertTrue(self.labeler._is_subsubheader(s))
            else:
                self.assertFalse(self.labeler._is_subsubheader(s))


class IsSpecialHeaderTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_special_header(self):
        for s, expected_label in LINES:
            if expected_label == 'special_header':
                self.assertTrue(self.labeler._is_special_header(s))
            else:
                self.assertFalse(self.labeler._is_special_header(s))


class IsSpeechNewTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_speech_new(self):
        for s, expected_label in LINES:
            if expected_label == 'speech_new':
                self.assertTrue(self.labeler._is_speech_new(s))
            else:
                self.assertFalse(self.labeler._is_speech_new(s))
    

class IsSpeechCtdTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_speech_ctd(self):
        for s, expected_label in LINES:
            if expected_label == 'speech_ctd':
                self.assertTrue(self.labeler._is_speech_ctd(s))
            else:
                self.assertFalse(self.labeler._is_speech_ctd(s))


class IsSceneTests(unittest.TestCase):
    def setUp(self):
        self.labeler = RuleLineLabeler(verbosity=2)

    def test_is_scene(self):
        for s, expected_label in LINES:
            if expected_label == 'scene':
                self.assertTrue(self.labeler._is_scene(s))
            else:
                self.assertFalse(self.labeler._is_scene(s))


if __name__ == '__main__':
    unittest.main()
