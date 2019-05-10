"""tests methods in `utils`.
"""

import unittest
import datetime
import utils

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


class ExtractFlatworldTagsTests(unittest.TestCase):
    """tests for `utils.extract_flatworld_tags`.
    """

    def test_extract_nothing(self):
        """tests that `extract_flatworld_tags` extracts nothing when there is
        no Flatworld tag to extract.
        """
        strings = [
            ('ORAL ANSWERS TO QUESTIONS ', ('ORAL ANSWERS TO QUESTIONS ', [])),
            ('bills', ('bills', [])),
            ('Header 3 states that', ('Header 3 states that', [])),
            ('speech by Mr. Gikaria was ', ('speech by Mr. Gikaria was ', [])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_header_tag(self):
        """tests that `extract_flatworld_tags` extracts <header> tags.
        """
        strings = [
            ('<header>ORAL ANSWERS TO QUESTIONS </header>', ('ORAL ANSWERS TO QUESTIONS ', ['header'])),
            ('<header>bills</header>', ('bills', ['header'])),
            ('<header>MOTIONS</header>', ('MOTIONS', ['header'])),
            ('<subheader>Question no. 259</subheader>', ('Question no. 259', ['subheader'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_speech_tag(self):
        """tests that `extract_flatworld_tags` extracts <speech> tags.
        """
        strings = [
            ('<newspeech>Mr. Speaker: Anyone here from the Ministry of</newspeech>', ('Mr. Speaker: Anyone here from the Ministry of', ['newspeech'])),
            ('<newspeech>Mr. Gikaria:</newspeech>', ('Mr. Gikaria:', ['newspeech'])),
            ('<newspeech>MR. GIKARIA: </newspeech>', ('MR. GIKARIA: ', ['newspeech'])),
            ('<speech>MR. GIKARIA: </speech>', ('MR. GIKARIA: ', ['speech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_scene_tag(self):
        """tests that `extract_flatworld_tags` extracts <scene> tags.
        """
        strings = [
            ('<scene>(Question proposed)</scene>', ('(Question proposed)', ['scene'])),
            ('<scene>(applause) </scene>', ('(applause) ', ['scene'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_misspelled_tag(self):
        """tests that `extract_flatworld_tags` is insensitive to common tag mis-spellings.
        """
        strings = [
            ('<headr>bills</headr>', ('bills', ['headr'])),
            ('<scen>(Question proposed)</scen>', ('(Question proposed)', ['scen'])),
            ('<newspech>Mr. Gikaria:</newspech>', ('Mr. Gikaria:', ['newspech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_hyphen_tag(self):
        """tests that `extract_flatworld_tags` is insensitive to common hyphens
        (e.g. "new-speech").
        """
        strings = [
            ('<sub-header>bills</sub-header>', ('bills', ['subheader'])),
            ('<new-speech>Mr. Gikaria:</new-speech>', ('Mr. Gikaria:', ['newspeech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])

    
    def test_extract_tag_any_case(self):
        """tests that `extract_flatworld_tags` is insensitive to the case of the tag.
        """
        strings = [
            ('<NEWSPEECH>Mr. Speaker: Anyone here from the Ministry of</NEWSPEECH>', ('Mr. Speaker: Anyone here from the Ministry of', ['newspeech'])),
            ('<NEWSPEECH>Mr. Speaker: Anyone here from the Ministry of</newspeech>', ('Mr. Speaker: Anyone here from the Ministry of', ['newspeech'])),
            ('<Newspeech>Mr. Gikaria:</Newspeech>', ('Mr. Gikaria:', ['newspeech'])),
            ('<NewSpeech>MR. GIKARIA: </Newspeech>', ('MR. GIKARIA: ', ['newspeech'])),
            ('<HEADER>MOTIONS</header>', ('MOTIONS', ['header'])),
            ('<Header>MOTIONS</Header>', ('MOTIONS', ['header'])),
            ('<Header>MOTIONS</HEADER>', ('MOTIONS', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_tag_with_spacing(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where there is spacing
        between the angle brackets and the tag name (e.g. `< Header >`).
        """
        strings = [
            ('< HEADER >MOTIONS</ header >', ('MOTIONS', ['header'])),
            ('< HEADER >MOTIONS</\nheader>', ('MOTIONS', ['header'])),
            ('< Sub HEADER >Question No. 259</\n sub header>', ('Question No. 259', ['subheader'])),
            ('< NewSpeech >MR. GIKARIA: </ Newspeech >', ('MR. GIKARIA: ', ['newspeech'])),
            ('<  NewSpeech>MR. GIKARIA: < /Newspeech>', ('MR. GIKARIA: ', ['newspeech'])),
            ('<NewSpeech >MR. GIKARIA: <  /Newspeech  >', ('MR. GIKARIA: ', ['newspeech'])),
            ('<New Speech >MR. GIKARIA: <  /New speech  >', ('MR. GIKARIA: ', ['newspeech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_wrong_closing(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where the closing tag
        has incorrect syntax (e.g. `<header />`, `<header>`).
        """
        strings = [
            ('<header>MOTIONS<header />', ('MOTIONS', ['header'])),
            ('<header>MOTIONS<header/ >', ('MOTIONS', ['header'])),
            ('<header>MOTIONS<header / >', ('MOTIONS', ['header'])),
            ('<header>MOTIONS<header>', ('MOTIONS', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_missing_bracket(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where there is a
        missing angle bracket (e.g. `<header`).
        """
        strings = [
            ('<header MOTIONS</header>', ('MOTIONS', ['header'])),
            ('<header>MOTIONS /header>', ('MOTIONS', ['header'])),
            ('header>MOTIONS /header>', ('MOTIONS', ['header'])),
            ('header>MOTIONS </header', ('MOTIONS ', ['header'])),
            ('<header>MOTIONS<header / >', ('MOTIONS', ['header'])),
            ('<header>MOTIONS<header>', ('MOTIONS', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_tag_missing_close(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where there is no
        closing tag in the line.
        """
        strings = [
            ('<header>MOTIONS', ('MOTIONS', ['header'])),
            ('<NewSpeech>MR. GIKARIA: ', ('MR. GIKARIA: ', ['newspeech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_missing_open(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where there is no
        opening tag in the line.
        """
        strings = [
            ('MOTIONS</header>', ('MOTIONS', ['header'])),
            ('MOTIONS<header>', ('MOTIONS', ['header'])),
            ('MR. GIKARIA: </NewSpeech>', ('MR. GIKARIA: ', ['newspeech'])),
            ('MR. GIKARIA: <NewSpeech>', ('MR. GIKARIA: ', ['newspeech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_extract_tag_mismatched(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where the
        open and close tags are mis-matched.
        """
        strings = [
            ('<header>MOTIONS</subheader>', ('MOTIONS', ['header', 'subheader'])),
            ('<speech>MOTIONS</subheader>', ('MOTIONS', ['speech', 'subheader'])),
            ('<NewSpeech>MR. GIKARIA: </header>', ('MR. GIKARIA: ', ['header', 'newspeech'])),
            ('<NewSpeech>MR. GIKARIA: </speech>', ('MR. GIKARIA: ', ['newspeech', 'speech'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_dont_extract_erroneous_bracket(self):
        """tests that `extract_flatworld_tags` does not extract tags where an angle bracket
        appears but is not a tag (e.g. `del<i`).

        This occurs sometimes due to OCR errors.
        """
        strings = [
            ('del<i', ('del<i', [])),
            ('del<i>', ('del<i>', [])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


    def test_dont_extract_other_tags(self):
        """tests that `extract_flatworld_tags` does not extract non-flatworld tags.
        (e.g. <i>text</i>).
        """
        strings = [
            ('<i>text</i>', ('<i>text</i>', [])),
            ('<b>text</i>', ('<b>text</i>', [])),
            ('<div>text</div>', ('<div>text</div>', [])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_others_nested(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where there
        are other nested tags.
        """
        strings = [
            ('<newspeech>Mr. Speaker: Anyone <b>here</b> from the Ministry of</newspeech>', ('Mr. Speaker: Anyone <b>here</b> from the Ministry of', ['newspeech'])),
            ('<header><b>MOTIONS</b></header>', ('<b>MOTIONS</b>', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_is_nested(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where the
        Flatworld tag is nested.
        """
        strings = [
            ('<i><newspeech>Mr. Speaker: Anyone here from the Ministry of</newspeech></i>', ('<i>Mr. Speaker: Anyone here from the Ministry of</i>', ['newspeech'])),
            ('<b><header>MOTIONS</header></b>', ('<b>MOTIONS</b>', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])
    

    def test_extract_tag_mid_nested(self):
        """tests that `extract_flatworld_tags` is insensitive to cases where the
        Flatworld tag is nested and there are other tags nested within the
        Flatworld tag (e.g. <b><newspeech><i>text</i></newspeech></b>).
        """
        strings = [
            ('<i><newspeech>Mr. Speaker: Anyone <b>here</b> from the Ministry of</newspeech></i>', ('<i>Mr. Speaker: Anyone <b>here</b> from the Ministry of</i>', ['newspeech'])),
            ('<b><header><i>MOTIONS</i></header></b>', ('<b><i>MOTIONS</i></b>', ['header'])),
        ]
        for s, expected in strings:
            s2, tags = utils.extract_flatworld_tags(s)
            self.assertEqual(s2, expected[0])
            self.assertEqual(tags, expected[1])


if __name__ == '__main__':
    unittest.main()
