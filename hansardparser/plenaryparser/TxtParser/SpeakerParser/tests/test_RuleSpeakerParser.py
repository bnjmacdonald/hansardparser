"""tests for RuleSpeakerParser.
"""

import unittest
from hansardparser.plenaryparser.TxtParser.SpeakerParser import RuleSpeakerParser

class ExtractSpeakerNameTests(unittest.TestCase):
    """tests the `_extract_speaker_name` method.
    """

    def setUp(self):
        self.speaker_parser = RuleSpeakerParser(verbosity=1)


    def test_name_with_prefix(self):
        """tests that `extract_speaker_name` extracts speaker name when speaker name has a prefix
        and surname(s) followed by a colon.
        """
        lines = [
            # line, (prefix, speaker_name, text)
            ('Mr. Speaker: Anyone here from the Ministry of',           ('Mr. Speaker', ' Anyone here from the Ministry of')),
            ('Dr. Kanyarna: Mr. Deputy Speaker, Sir, let me join',      ('Dr. Kanyarna', ' Mr. Deputy Speaker, Sir, let me join')),
            ('Hon. Kanyarna: Mr. Deputy Speaker, Sir, let me join',     ('Hon. Kanyarna', ' Mr. Deputy Speaker, Sir, let me join')),
            ('Bw. Haji: Anyone here from the Ministry of',              ('Bw. Haji', ' Anyone here from the Ministry of')),
            ('Bi. Haji: Anyone here from the Ministry of',              ('Bi. Haji', ' Anyone here from the Ministry of')),
            ('Ms. Haji: Anyone here from the Ministry of',              ('Ms. Haji', ' Anyone here from the Ministry of')),
            ('Mrs. Haji: Anyone here from the Ministry of',             ('Mrs. Haji', ' Anyone here from the Ministry of')),
            ('Maj. Kanyarna: Mr. Deputy Speaker, Sir, let me join',     ('Maj. Kanyarna', ' Mr. Deputy Speaker, Sir, let me join')),
            ('MR. MWACHOFI: On a point of order, Mr. Speaker, Sir.',    ('MR. MWACHOFI', ' On a point of order, Mr. Speaker, Sir.')),
            ('Maj. JK Kanyarna: Mr. Deputy Speaker, Sir, let me join',  ('Maj. JK Kanyarna', ' Mr. Deputy Speaker, Sir, let me join')),
            ('Maj. Julius Kanyarna: Mr. Deputy Speaker, Sir, let me join', ('Maj. Julius Kanyarna', ' Mr. Deputy Speaker, Sir, let me join')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_without_prefix(self):
        """tests that `extract_speaker_name` extracts speaker name when speaker name
        has surname(s) followed by a colon.
        """
        lines = [
            ('Bett: Thank you, Mr. Speaker. Let me start by',           ('Bett', ' Thank you, Mr. Speaker. Let me start by')),
            ('JK Bett: Thank you, Mr. Speaker. Let me start by',        ('JK Bett', ' Thank you, Mr. Speaker. Let me start by')),
            ('Willy Bett: Thank you, Mr. Speaker. Let me start by',     ('Willy Bett', ' Thank you, Mr. Speaker. Let me start by')),
            ('Willy K Bett: Thank you, Mr. Speaker. Let me start by',   ('Willy K Bett', ' Thank you, Mr. Speaker. Let me start by')),
            ('Willy Kitu Bett: Thank you, Mr. Speaker. Let me start by',('Willy Kitu Bett', ' Thank you, Mr. Speaker. Let me start by'))
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_with_prefix_no_punct(self):
        """tests that `extract_speaker_name` extracts speaker name when speaker name
        has a prefix (without punctuation) followed by surname(s) and a colon.
        """
        lines = [
            # prefix variations.
            ('Mr Mwaura: I rise to oppose this bill.',  ('Mr Mwaura', ' I rise to oppose this bill.')),
            ('Ms Mwaura: I rise to oppose this bill.',  ('Ms Mwaura', ' I rise to oppose this bill.')),
            ('Mrs Mwaura: I rise to oppose this bill.', ('Mrs Mwaura', ' I rise to oppose this bill.')),
            ('Dr Mwaura: I rise to oppose this bill.',  ('Dr Mwaura', ' I rise to oppose this bill.')),
            ('Maj Mwaura: I rise to oppose this bill.', ('Maj Mwaura', ' I rise to oppose this bill.')),
            ('Bw Mwaura: I rise to oppose this bill.',  ('Bw Mwaura', ' I rise to oppose this bill.')),
            ('Bi Mwaura: I rise to oppose this bill.',  ('Bi Mwaura', ' I rise to oppose this bill.')),
            ('Hon Mwaura: I rise to oppose this bill.', ('Hon Mwaura', ' I rise to oppose this bill.')),
            # + multiple names
            ('Hon Isaac Mwaura: I rise to oppose this bill.',      ('Hon Isaac Mwaura', ' I rise to oppose this bill.')),
            ('Mr Isaac Mwaura: I rise to oppose this bill.',       ('Mr Isaac Mwaura', ' I rise to oppose this bill.')),
            ('Mrs Isaac Ali Mwaura: I rise to oppose this bill.',  ('Mrs Isaac Ali Mwaura', ' I rise to oppose this bill.')),
            # lowercase prefix.
            ('hon Mwaura: I rise to oppose this bill.', ('hon Mwaura', ' I rise to oppose this bill.')),
            ('mr Mwaura: I rise to oppose this bill.',  ('mr Mwaura', ' I rise to oppose this bill.')),
            ('ms Mwaura: I rise to oppose this bill.',  ('ms Mwaura', ' I rise to oppose this bill.')),
            ('mrs Mwaura: I rise to oppose this bill.', ('mrs Mwaura', ' I rise to oppose this bill.')),
            # lowercase prefix and lowercase name.
            ('hon mwaura: I rise to oppose this bill.', ('hon mwaura', ' I rise to oppose this bill.')),
            ('mr mwaura: I rise to oppose this bill.',  ('mr mwaura', ' I rise to oppose this bill.')),
            ('ms mwaura: I rise to oppose this bill.',  ('ms mwaura', ' I rise to oppose this bill.')),
            ('mrs mwaura: I rise to oppose this bill.', ('mrs mwaura', ' I rise to oppose this bill.')),
            # + multiple names
            ('hon isaac mwaura: I rise to oppose this bill.',     ('hon isaac mwaura', ' I rise to oppose this bill.')),
            ('mr isaac mwaura: I rise to oppose this bill.',      ('mr isaac mwaura', ' I rise to oppose this bill.')),
            ('ms isaac ali mwaura: I rise to oppose this bill.',  ('ms isaac ali mwaura', ' I rise to oppose this bill.')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_without_colon(self):
        """tests that `extract_speaker_name` extracts speaker name when speaker name
        is not immediately followed by a colon.
        """
        lines = [
            ('Hon. Muli asked the Minister for Justice whether',  ('Hon. Muli', 'Hon. Muli asked the Minister for Justice whether')),
            ('Mr. Muli asked the Minister for Justice whether',   ('Mr. Muli', 'Mr. Muli asked the Minister for Justice whether')),
            ('Mr. Muli said to the Minister for Justice',         ('Mr. Muli', 'Mr. Muli said to the Minister for Justice')),
            ('Mr Muli alisema to the Minister for Justice',       ('Mr Muli', 'Mr Muli alisema to the Minister for Justice')),
            ('Ms. Muli ali to the Minister for Justice',          ('Ms. Muli', 'Ms. Muli ali to the Minister for Justice')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_in_parentheses(self):
        """tests that `extract_speaker_name` extracts speaker name when speaker name
        is in parentheses after the speaker's appointment.
        """
        lines = [
            ('The Minister for Agriculture (Mr. Mwakileo): I rise to second this motion, Mr. Speaker.',     ('The Minister for Agriculture (Mr. Mwakileo)', ' I rise to second this motion, Mr. Speaker.')),
            ('The Minister for Agriculture (mr mwakileo): I rise to second this motion, Mr. Speaker.',     ('The Minister for Agriculture (mr mwakileo)', ' I rise to second this motion, Mr. Speaker.')),
            ('The Minister for Agriculture (mr mark mwakileo): I rise to second this motion, Mr. Speaker.', ('The Minister for Agriculture (mr mark mwakileo)', ' I rise to second this motion, Mr. Speaker.')),
            # ('The Minister for Agriculture (mr mark mwakileo) asked the speaker whether',                   ('The Minister for Agriculture (mr mark mwakileo)', 'The Minister for Agriculture (mr mark mwakileo) asked the speaker whether')),
            # ('The Assistant Minister for Nairobi Metropolitan Development (mr mark mwakileo): I rise to second this motion, Mr. Speaker.', ('The Assistant Minister for Nairobi Metropolitan Development (mr mark mwakileo)', ' I rise to second this motion, Mr. Speaker.')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_appt_in_parentheses(self):
        """tests that `extract_speaker_name` extracts speaker name appointment
        is in parentheses after the speaker's name.
        """
        lines = [
            ('Hon. Oparanya (The Minister for Agriculture): Thank you, Mr. Speaker.', ('Hon. Oparanya (The Minister for Agriculture)', ' Thank you, Mr. Speaker.')),
            ('Hon. Oparanya (The Minister for Agriculture) asked the speaker whether', ('Hon. Oparanya (The Minister for Agriculture)', 'Hon. Oparanya (The Minister for Agriculture) asked the speaker whether')),
            ('ms jm oparanya (the minister for justice and legal affairs): Thank you, Mr. Speaker.', ('ms jm oparanya (the minister for justice and legal affairs)', ' Thank you, Mr. Speaker.')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_none_input(self):
        """tests that `extract_speaker_name` returns None is input string is None.
        """
        self.assertEqual(self.speaker_parser._extract_speaker_name(None), (None, None))


    def test_false_positive_segue(self):
        """tests that `extract_speaker_name` does not extract speaker name when
        it is a false positive.
        """
        lines = [
            ('Mr. Speaker, I said that he was', (None, 'Mr. Speaker, I said that he was')),
            ('I propose that:', (None, 'I propose that:')),
            ('Mr. Speaker, I propose that:', (None, 'Mr. Speaker, I propose that:')),
            ('Mr. Speaker I propose that: observing that', (None, 'Mr. Speaker I propose that: observing that')),
            ('and then he asked whether', (None, 'and then he asked whether')),
            ('Wizara: hiyo.', (None, 'Wizara: hiyo.')),
            ('The following Papers were laid on the Table:', (None, 'The following Papers were laid on the Table:')),
            ('MR. EKIDOR, Kwa niaba ya Bw. Twerith alimwuliza Wazairi wa Nchi, Ofisi ya Rais ni ambaye alikuwa mfanyakazi', (None, 'MR. EKIDOR, Kwa niaba ya Bw. Twerith alimwuliza Wazairi wa Nchi, Ofisi ya Rais ni ambaye alikuwa mfanyakazi'))
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_digits(self):
        """tests that `extract_speaker_name` extracts speaker name when letters
        have been wrongly OCR'd as digits.
        """
        lines = [
            ('THE AS5I5TANT MINISTER FOR HEALTH ( Mr.Ogur) : Mr. Speaker, Sir,', ('THE AS5I5TANT MINISTER FOR HEALTH ( Mr.Ogur) ', ' Mr. Speaker, Sir,')),
            ('Mr. 0Gur: Mr. Speaker, sir,', ('Mr. 0Gur', ' Mr. Speaker, sir,'))
        ]
        for line, expected in lines:
            result = self.speaker_parser._extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


class ParseSpeakerNameV2Tests(unittest.TestCase):
    """tests the `_parse_speaker_name` method.
    """
    
    def setUp(self):
        self.speaker_parser = RuleSpeakerParser(verbosity=1)

    def test_name_with_prefix(self):
        """tests that `parse_speaker_nameV2` extracts speaker name when speaker name has a prefix
        and surname(s) followed by a colon.
        """
        lines = [
            # line, (prefix, speaker_name, text)
            ('Mr. Kanyarna',        ('Mr', 'Kanyarna', None)),
            ('Dr. Kanyarna',        ('Dr', 'Kanyarna', None)),
            ('Hon. Kanyarna',       ('Hon', 'Kanyarna', None)),
            ('Bw. Haji',            ('Bw', 'Haji', None)),
            ('Bi. Haji',            ('Bi', 'Haji', None)),
            ('Ms. Haji',            ('Ms', 'Haji', None)),
            ('Mrs. Haji',           ('Mrs', 'Haji', None)),
            ('Maj. Kanyarna',       ('Maj', 'Kanyarna', None)),
            ('MR. MWACHOFI',        ('MR', 'MWACHOFI', None)),
            ('Maj. JK Kanyarna',    ('Maj', 'JK Kanyarna', None)),
            ('Maj. Julius Kanyarna',('Maj', 'Julius Kanyarna', None)),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_without_prefix(self):
        """tests that `parse_speaker_nameV2` extracts speaker name when speaker name
        has surname(s) followed by a colon.
        """
        lines = [
            ('Bett',           (None, 'Bett', None)),
            ('JK Bett',        (None, 'JK Bett', None)),
            ('Willy Bett',     (None, 'Willy Bett', None)),
            ('Willy K Bett',   (None, 'Willy K Bett', None)),
            ('Willy Kitu Bett',(None, 'Willy Kitu Bett', None)),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_with_prefix_no_punct(self):
        """tests that `parse_speaker_nameV2` extracts speaker name when speaker name
        has a prefix (without punctuation) followed by surname(s) and a colon.
        """
        lines = [
            # prefix variations.
            ('Mr Mwaura',  ('Mr', 'Mwaura', None)),
            ('Ms Mwaura',  ('Ms', 'Mwaura', None)),
            ('Mrs Mwaura', ('Mrs', 'Mwaura', None)),
            ('Dr Mwaura',  ('Dr', 'Mwaura', None)),
            ('Maj Mwaura', ('Maj', 'Mwaura', None)),
            ('Bw Mwaura',  ('Bw', 'Mwaura', None)),
            ('Bi Mwaura',  ('Bi', 'Mwaura', None)),
            ('Hon Mwaura', ('Hon', 'Mwaura', None)),
            # + multiple names
            ('Hon Isaac Mwaura',      ('Hon', 'Isaac Mwaura', None)),
            ('Mr Isaac Mwaura',       ('Mr', 'Isaac Mwaura', None)),
            ('Mrs Isaac Ali Mwaura',  ('Mrs', 'Isaac Ali Mwaura', None)),
            # lowercase prefix.
            ('hon Mwaura', ('hon', 'Mwaura', None)),
            ('mr Mwaura',  ('mr', 'Mwaura', None)),
            ('ms Mwaura',  ('ms', 'Mwaura', None)),
            ('mrs Mwaura', ('mrs', 'Mwaura', None)),
            # lowercase prefix and lowercase name.
            ('hon mwaura', ('hon', 'mwaura', None)),
            ('mr mwaura',  ('mr', 'mwaura', None)),
            ('ms mwaura',  ('ms', 'mwaura', None)),
            ('mrs mwaura', ('mrs', 'mwaura', None)),
            # + multiple names
            ('hon isaac mwaura',     ('hon', 'isaac mwaura', None)),
            ('mr isaac mwaura',      ('mr', 'isaac mwaura', None)),
            ('ms isaac ali mwaura',  ('ms', 'isaac ali mwaura', None)),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_without_colon(self):
        """tests that `parse_speaker_nameV2` extracts speaker name when speaker name
        is not immediately followed by a colon.
        """
        lines = [
            ('Hon. Muli',  ('Hon', 'Muli', None)),
            ('Mr. Muli',   ('Mr', 'Muli', None)),
            ('Mr. Muli',   ('Mr', 'Muli', None)),
            ('Mr Muli',    ('Mr', 'Muli', None)),
            ('Ms. Muli',   ('Ms', 'Muli', None)),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_name_in_parentheses(self):
        """tests that `parse_speaker_nameV2` extracts speaker name when speaker name
        is in parentheses after the speaker's appointment.
        """
        lines = [
            ('The Minister for Agriculture (Mr. Mwakileo)', ('Mr', 'Mwakileo', 'The Minister for Agriculture ')),
            ('The Minister for Agriculture (mr mwakileo)', ('mr', 'mwakileo', 'The Minister for Agriculture ')),
            ('The Minister for Agriculture (mr mark mwakileo)', ('mr', 'mark mwakileo', 'The Minister for Agriculture ')),
            ('The Minister for Agriculture (mr mark mwakileo)', ('mr', 'mark mwakileo', 'The Minister for Agriculture ')),
            ('The Assistant Minister for Nairobi Metropolitan Development (mr mark mwakileo)', ('mr', 'mark mwakileo', 'The Assistant Minister for Nairobi Metropolitan Development ')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_appt_in_parentheses(self):
        """tests that `parse_speaker_nameV2` extracts speaker name appointment
        is in parentheses after the speaker's name.
        """
        lines = [
            ('Hon. Oparanya (The Minister for Agriculture)', ('Hon', 'Oparanya ', 'The Minister for Agriculture')),
            ('Hon. Oparanya (The Minister for Agriculture)', ('Hon', 'Oparanya ', 'The Minister for Agriculture')),
            ('ms jm oparanya (the minister for justice and legal affairs)', ('ms', 'jm oparanya ', 'the minister for justice and legal affairs')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])

    def test_none_input(self):
        """tests that `parse_speaker_nameV2` returns None is input string is None.
        """
        self.assertEqual(self.speaker_parser._parse_speaker_name(None), (None, None, None))
    

    def test_speaker_without_name(self):
        """tests that `parse_speaker_nameV2` returns Speaker as an appointment
        raher than name.
        """
        lines = [
            ('Mr. Speaker', ('Mr', None, 'Speaker')),
            ('Ms. Speaker', ('Ms', None, 'Speaker')),
            ('Mr. Deputy Speaker', ('Mr', None, 'Deputy Speaker')),
            ('Mr. Temporary Deputy Speaker', ('Mr', None, 'Temporary Deputy Speaker')),
            ('Deputy Speaker', (None, None, 'Deputy Speaker')),
            ('Temporary Deputy Speaker', (None, None, 'Temporary Deputy Speaker')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


    def test_appt_without_name(self):
        """tests that `parse_speaker_nameV2` returns appointment only when string
        contains an appointment but no speaker name.
        """
        lines = [
            ('The Minister for Agriculture', (None, None, 'The Minister for Agriculture')),
            ('the assistant minister for nairobi metropolitan development', (None, None, 'the assistant minister for nairobi metropolitan development')),
            ('asst. minister for education', (None, None, 'asst. minister for education')),
            ('mr. minister for education', ('mr', None, 'minister for education')),
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])

    def test_digits(self):
        """tests that `parse_speaker_nameV2` parses speaker name when letters
        have been wrongly OCR'd as digits.
        """
        lines = [
            ('THE AS5I5TANT MINISTER FOR HEALTH ( Mr.Ogur) ', ('Mr', 'Ogur', 'THE AS5I5TANT MINISTER FOR HEALTH ')),
            ('Mr. 0Gur', ('Mr', '0Gur', None))
        ]
        for line, expected in lines:
            result = self.speaker_parser._parse_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


if __name__ == '__main__':
    unittest.main()
