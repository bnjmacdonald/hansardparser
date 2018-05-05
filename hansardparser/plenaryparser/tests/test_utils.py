
import unittest
import datetime
from hansardparser.plenaryparser import utils


class ExtractSpeakerNameTests(unittest.TestCase):
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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])

    def test_none_input(self):
        """tests that `extract_speaker_name` returns None is input string is None.
        """
        self.assertEqual(utils.extract_speaker_name(None), (None, None))


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
            result = utils.extract_speaker_name(line)
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
            result = utils.extract_speaker_name(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


class ParseSpeakerNameV2Tests(unittest.TestCase):
    """tests for `TxtParser.extract_speaker_name` method.
    """

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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])

    def test_none_input(self):
        """tests that `parse_speaker_nameV2` returns None is input string is None.
        """
        self.assertEqual(utils.parse_speaker_nameV2(None), (None, None, None))
    
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
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
            result = utils.parse_speaker_nameV2(line)
            for i, el in enumerate(result):
                self.assertEqual(el, expected[i])


class UtilsTests(unittest.TestCase):
    """tests for other methods in `utils` module.

    Todos:

        TODO: divide these tests into classes, where each method gets its own
            class.
    """

    def setUp(self):
        self.str_dates = [
            'Tuesday, 12th October, 2010',
            'Tuesday, 12th October, 2010 ',
            'Tuesday, 8th February, 2011 ',
            'Wednesday, 22nd February, 2012(A)',
            'Thursday, 14 th May, 2009'
        ]
        self.page_dates = [
            ('Tuesday, 12th October, 2010', True),
            ('Tuesday, 12th October, 2010 ', True),
            ('Tuesday, 8th February, 2011 ', True),
            ('March4, 2015', True),
            ('Wednesday, 22nd February, 2012(A)', True),
            ('Thursday, 14 th May, 2009', True),
            ('Tuesday, 12', True),
            ('th', True),
            (' April, 2011 ', True),
            ('April was the last time I', False),
            ('April 2006 was the last time I', False),
            ('word Tuesday 12', False),
            ('word May 1998', False),
            ('November, 2010.', False),
        ]
        self.page_headings = [
            ("PARLIAMENTARY DEBATES", True),
            ("PARLIAMENTARY DEBATE", True),
            ("PARLIMENTARY DEBATE", True),
            ("PARLIAMENTARY LDEBATES", True),
            ("PARLIAMEANTARY DEBATES", True),
            ("PARLIAENTARY DEBATES", True),
            ("PARLIAMENTARYDEBATES", True),
            ("PARLIAMENTARY", False),
            ("blah blah blah PARLIAMENTARY DEBATES", False),
            ("PARLIAMENTARY DEBATES blah ... blah", False),
            ("PARLIAMENTARY DEBATES.", False),
            ("parliamentary debates", False),
        ]
        self.true_headings = [
            'NATIONAL ASSEMBLY',
            '\n NATIONAL ASSEMBLY',
            '\n NATIONAL ASSEMBLY  \t',
            'NATIONAL ASSEMBLY \n  OFFICIAL REPORT',
            ' OFFICIAL REPORT',
            ' OFFICIAL REPORT '
        ]
        self.false_headings = [
            '',
            ' ',
            'NATIONAL ASSEMBL',
            'national assembly',
            'aee f df fd',
            'NATIONAL ASSEMBLY abc OFFICIAL REPORT',
            'NATIONAL ASSEMBLY words word another word ',
            None,
        ]
        self.true_footers = [
            'Disclaimer:  The electronic version of theOfficial Hansard Report is for information purposes',
            'only. A certified version of this Report can be obtained from the Hansard Editor.',
            'Disclaimer:  The electronic version of the Official Hansard Report is for information purposes',
            'blah blah blah A certified version of this Report can be obtained from the Hansard Editor.',
            'Official Hansard Report is for information purposes',
            'Disclaimer:  The electronic version of the',
        ]
        self.false_footers = [
            'Disclaimer is some other text blah blah',
            'A certified version blah blah',
        ]
        self.headers = [
            ('EXTENSION OF S ITING H OURS', 'extension of siting hours'),
            ('RESUMPTION OF O RAL A NSWERS TO Q UESTIONS', 'resumption of oral answers to questions'),
            ('MOTION OF ADJOURNMENT', 'motion of adjournment'),
            ('MOTION', 'motion'),
            ('A MOTION', 'a motion'),
            ('FORM A', 'form a'),
            ('ADOPTION OF PSC R ECOMMENDATIONS ON C HAIR M EMBERS OF IIEC', 'adoption of psc recommendations on chair members of iiec'),
            ('THE CHILDREN (A MENDMENT )B ILL', 'the children (amendment) bill'),
            ('THE LAND LAWS (A MENDMENT )B ILL', 'the land laws (amendment) bill'),
            ('T HE S UPPLEMENTARY A PPROPRIATION ( NO. 2) B ILL', 'the supplementary appropriation (no. 2) bill'),
            ('THE L EGAL A ID BILL', 'the legal aid bill'),
            ('A B ILL TO A MEND THE KTDA A', 'a bill to amend the ktdaa'),
            ('A BILL TO A MEND THE NCPB A', 'a bill to amend the ncpba'),
            ('BILL ON A RID LANDS', 'bill on arid lands'),
            ('THE PRESIDENT \' SA WARD BILL', 'the president\'s award bill'),
        ]
        # self.parenth_names = [
        #     ('The Temporary Deputy Speaker (Mr. Imanyara)', (''))
        # ]

    def test_is_str_date(self):
        for date_str in self.str_dates:
            date = utils.convert_str_to_date(date_str)
            # print(date_str, date)
            self.assertTrue(isinstance(date, datetime.datetime))

    def test_is_page_heading(self):
        for heading, truth in self.page_headings:
            # print(date_str, truth)
            # assert is_page_date(date_str) == truth
            self.assertEqual(utils.is_page_heading(heading), truth)

    def test_is_page_date(self):
        for date_str, truth in self.page_dates:
            # print(date_str, truth)
            # assert is_page_date(date_str) == truth
            self.assertEqual(utils.is_page_date(date_str), truth)

    def test_is_transcript_heading(self):
        for text in self.true_headings:
            self.assertTrue(utils.is_transcript_heading(text))
        for text in self.false_headings:
            self.assertFalse(utils.is_transcript_heading(text))

    def test_get_transcript_heading(self):
        results = ['NATIONAL ASSEMBLY OFFICIAL REPORT', 'NATIONAL ASSEMBLY', 'OFFICIAL REPORT']
        for text in self.true_headings:
            self.assertTrue(utils.get_transcript_heading(text) in results)

    def test_fix_header_words(self):
        for text, truth in self.headers:
            self.assertEqual(utils.fix_header_words(text),  truth)

    def test_is_page_footer(self):
        for text in self.true_footers:
            self.assertTrue(utils.is_page_footer(text))
        for text in self.false_footers:
            self.assertFalse(utils.is_page_footer(text))

    def test_extract_outer_tag(self):
        strings = [
            ('<header>ORAL ANSWERS TO QUESTIONS </header>', ('ORAL ANSWERS TO QUESTIONS ', 'header', 'header')),
            # capitalized tag
            ('<HEADER>ORAL ANSWERS TO QUESTIONS </HEADER>', ('ORAL ANSWERS TO QUESTIONS ', 'HEADER', 'HEADER')),
            # close tag but no open tag.
            ('<newspeech>Mr. Speaker: Anyone here from the Ministry of', ('Mr. Speaker: Anyone here from the Ministry of', 'newspeech', None)),
            # open tag but no close tag.
            ('Mr. Speaker: Anyone here from the Ministry of</newspeech>', ('Mr. Speaker: Anyone here from the Ministry of', None, 'newspeech')),
            # nested tags.
            ('<newspeech>Mr. Speaker: Anyone <b>here</b> from the Ministry of</newspeech>', ('Mr. Speaker: Anyone <b>here</b> from the Ministry of', 'newspeech', 'newspeech')),
            ('Mr. Speaker: Anyone <b>here</b> from the Ministry of</newspeech>', ('Mr. Speaker: Anyone <b>here</b> from the Ministry of', None, 'newspeech')),
        ]
        for s, expected in strings:
            s2, open_tag, close_tag = utils.extract_outer_tag(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(open_tag, expected[1])
            self.assertEqual(close_tag, expected[2])


if __name__ == '__main__':
    unittest.main()
