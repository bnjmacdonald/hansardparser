"""plenaryparser tests.

Todos:

    TODO: revise metadata test cases by: (a) dropping parliament; (b) revising
        date to datetime.
"""

import unittest
import os
import re
import datetime
import pytz
import calendar
import pandas as pd
from bs4 import BeautifulSoup

import settings
from plenaryparser import utils
from plenaryparser.XmlParser import XmlParser
from plenaryparser.Entry import Entry


# # DEBUGGING:
# fname = 'Hansard 12.04.11.pdf'
# input_dir = os.path.join(settings.BASE_DIR, 'hansardparser', 'test_input')
# file_path = os.path.join(input_dir, fname)
# verbose = 3
# parser = XmlParser(verbose=verbose)
# path = os.path.join(settings.BASE_DIR, 'query/output/temp')
# metadata, contents = parser.process_transcript(
#     file_path=file_path,
#     save_soup=True,
#     path=path,
#     rm_whitespace=True,
#     append_meta=True,
#     to_format='df-long',
#     speeches_only=False
# )

# from hansardparser.Sitting import Sitting
# from bs4 import BeautifulSoup, Tag, NavigableString
# self = parser
# metadata = Sitting()
# max_check = 50
# self.soup = self._convert_pdf(file_path, False)
# self._preprocess_soup()
# metadata = self._process_meta()
# for i, line in enumerate(contents):
#     print('-------------')
#     print(line)
#     if i == 100:
#         break


class UtilsTests(unittest.TestCase):
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

    def test_parse_speaker_name(self):
        raise NotImplementedError

    # def test_clean_speaker_name(self):
    #     for text, truth in self.parenth_names:
    #         answer = utils.clean_speaker_name(text, name_at_begin=True)
    #         self.assertEqual(answer, truth)


class ParseHansardTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.base_dir = '/Volumes/Transcend/HANSARDS'
        # self.base_dir = '/Users/mounted'
        cls.base_dir = os.path.join(settings.PROJECT_DIR, 'tests', 'test_input')
        cls.filenames = [
            '31st march 1998P.pdf',
            '5th November 1998P.pdf',
            '1st December, 1999P.pdf',
            '14th June, 2000A.pdf',
            '3rd October, 2001A.pdf',
            '3rd May, 2006A.pdf',
            '25th April, 2007A.pdf',
            '15th January, 2008P.pdf',
            '13th March, 2008P.pdf',
            '30th July, 2008A.pdf',
            '6th August, 2008A.pdf',
            '7th August 2008P.pdf',
            'Feb 19.09.pdf',
            'May 13.09A.pdf',
            'May14.09.pdf',
            'May 20.09P.pdf',
            'June 24.09P.pdf',
            'November 18.09P.pdf',
            'Hansard 23.06.10P.pdf',
            'September 01.10A.pdf',
            'October 12.10.pdf',
            'December 7.10.pdf',
            'February 08.11P.pdf',
            'Hansard 12.04.11.pdf',
            'Hansard 22.02.12A.pdf',
            'Hansard 03.10.12A.pdf',
            'Hansard 10.10.12P.pdf'
        ]
        try:
            cls.parsers = {}
            for filename in cls.filenames:
                file_path = os.path.join(cls.base_dir, filename)
                parser = XmlParser(verbose=False)
                parser.soup = parser._convert_pdf(file_path, save_soup=False)
                parser._preprocess_soup()
                cls.parsers[filename] = parser
                print('parsed {0}...'.format(filename))
        except FileNotFoundError:
            print('WARNING: could not find files in {0} for ParseHansardTests'.format(cls.base_dir))

    def setUp(self):
        self.date_strings = [day.lower() for day in calendar.day_name] + [month.lower() for month in calendar.month_name]
        self.day_endings = ['th', 'st', 'nd', 'rd']
        # self.base_dir = '/Users/mounted'
        self.parser = XmlParser
        self.meta = {
            '31st march 1998P.pdf':
                {'date': datetime.datetime(1998, 3, 31, tzinfo=pytz.utc),
                 'parliament': 8,
                 'time': None,
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '8'
                 },
            '5th November 1998P.pdf':
                {'date': datetime.datetime(1998, 11, 5, tzinfo=pytz.utc),
                 'parliament': 8,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2145'
                 },
            '1st December, 1999P.pdf':
                {'date': datetime.datetime(1999, 12, 1, tzinfo=pytz.utc),
                 'parliament': 8,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2677'
                 },
            '14th June, 2000A.pdf':
                {'date': datetime.datetime(2000, 6, 14, tzinfo=pytz.utc),
                 'parliament': 8,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '1021'
                 },
            '3rd October, 2001A.pdf':
                {'date': datetime.datetime(2001, 10, 3, tzinfo=pytz.utc),
                 'parliament': 8,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2385'
                 },
            '3rd May, 2006A.pdf':
                {'date': datetime.datetime(2006, 5, 3, tzinfo=pytz.utc),
                 'parliament': 9,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '847'
                 },
            '25th April, 2007A.pdf':
                {'date': datetime.datetime(2007, 4, 25, tzinfo=pytz.utc),
                 'parliament': 9,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '877'
                 },
            '15th January, 2008P.pdf':
                {'date': datetime.datetime(2008, 1, 15, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': None,
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '1'
                 },
            '13th March, 2008P.pdf':
                {'date': datetime.datetime(2008, 3, 13, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '135'
                 },
            '30th July, 2008A.pdf':
                {'date': datetime.datetime(2008, 7, 30, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2177'
                 },
            '6th August, 2008A.pdf':
                {'date': datetime.datetime(2008, 8, 6, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2341'
                 },
            '7th August 2008P.pdf':
                {'date': datetime.datetime(2008, 8, 7, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 'start_page': '2431'
                 },
            '2009/Feb 19.09.pdf':
                {'date': datetime.datetime(2009, 2, 19, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'May 13.09A.pdf':
                {'date': datetime.datetime(2009, 5, 13, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'May14.09.pdf':
                {'date': datetime.datetime(2009, 5, 14, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'May 20.09P.pdf':
                {'date': datetime.datetime(2009, 5, 20, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'June 24.09P.pdf':
                {'date': datetime.datetime(2009, 6, 24, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'November 18.09P.pdf':
                {'date': datetime.datetime(2009, 11, 18, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1',
                 },
            'Hansard 23.06.10P.pdf':
                {'date': datetime.datetime(2010, 6, 23, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'September 01.10A.pdf':
                {'date': datetime.datetime(2010, 9, 1, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'October 12.10.pdf':
                {'date': datetime.datetime(2010, 10, 12, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'December 7.10.pdf':
                {'date': datetime.datetime(2010, 12, 7, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'February 08.11P.pdf':
                {'date': datetime.datetime(2011, 2, 8, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'Hansard 12.04.11.pdf':
                {'date': datetime.datetime(2011, 4, 12, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'Hansard 03.10.12A.pdf':
                {'date': datetime.datetime(2012, 10, 3, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'Hansard 22.02.12A.pdf':
                {'date': datetime.datetime(2012, 2, 22, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '9.00 a.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
            'Hansard 10.10.12P.pdf':
                {'date': datetime.datetime(2012, 10, 10, tzinfo=pytz.utc),
                 'parliament': 10,
                 'time': '2.30 p.m.',
                 'heading': 'NATIONAL ASSEMBLY OFFICIAL REPORT',
                 # 'start_page': '1'
                 },
        }
        self.entry_types = [
            ('<text font="3" height="13" left="381" top="281" width="155"><i>[Mr. Speaker in the Chair]</i></text>', 'scene'),
            ('<text top="257" left="319" width="285" height="16" font="8">(<i>Question of the amendment proposed)</i> </text>', 'scene'),
            ('<text font="2" height="13" left="171" top="766" width="517"><b>The Assistant Minister for Finance </b>(Mr. Arap-Kirui):    I am making my maiden speech!</text>', 'speech_new'),
            ('<text top="174" left="171" width="605" height="13" font="2"><b>Dr. Ali: </b>Mr. Speaker, Sir, I want to inform the House that the Minister is misleading us because there is</text>', 'speech_new'),
            ('<text top="294" left="171" width="86" height="13" font="2"><b>Mr.  Munyao:</b></text>', 'speech_new'),
            ('<text top="563" left="171" width="385" height="13" font="0">Is Mr. Anyona not here? We will leave his Question until the end.</text>', 'speech_ctd'),
            ('<text top="381" left="140" width="640" height="15" font="1">buildings to allow for fire safety, <i>et  cetera</i>. So, our role is basically that of co-ordinating </text>', 'speech_ctd'),
            ('<text top="140" left="120" width="656" height="13" font="0">I am three in one! That is why we are asking the KANU Government to be realistic and agree to reduce the</text>', 'speech_ctd'),
            ('<text top="715" left="191" width="195" height="15" font="1">time when it was committed.&#34; </text>', 'speech_ctd'),
            ('<text top="712" left="243" width="490" height="16" font="8">THAT<b>,  </b>aware  that  in  2013  the  British  Government  agreed  to  pay </text>', 'speech_ctd'),
            ('<text top="326" left="171" width="605" height="13" font="0">Mr. Temporary Deputy Speaker, Sir, while I am happy with the move by the Ministry of Education and</text>', 'speech_ctd'),
            ('<text top="422" left="135" width="21" height="16" font="8">3? </text>', 'speech_ctd'),
            ('<text top="589" left="324" width="272" height="15" font="2"><b>QUESTIONS BY PRIVATE NOTICE</b> </text>', 'header'),
            ('<text top="735" left="409" width="79" height="13" font="4"><i>First Reading</i></text>', 'special_header'),
            ('<text top="770" left="355" width="89" height="10" font="4">AINTENANCE OF</text>', 'subheader'),
            # ('<text top="894" left="243" width="11" height="15" font="1">B</text>', 'subheader'),
            # ('<text top="677" left="478" width="20" height="15" font="1">-M</text>', 'subheader'),
            ('<text top="496" left="400" width="97" height="13" font="3"><i>Question No.086</i></text>', 'subsubheader'),
            ('<text top="496" left="400" width="97" height="13" font="3"><i>Clause 16</i></text>', 'subsubheader'),
        ]
        self.speaker_data = [
            {
                'line': '<text top="224" left="191" width="590" height="15" font="2"><b>Mr.  Chepkitony:  </b>Thank you, Mr. Temporary Deputy Speaker, Sir, for giving me this </text>',
                'prev_entry': Entry(
                    text='Thank you, Mr. Temporary Deputy Speaker.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'Mr. Chepkitony',
                'text': 'Thank you, Mr. Temporary Deputy Speaker, Sir, for giving me this'
            },
            {
                'line': '<text top="106" left="191" width="590" height="15" font="2"><b>The Temporary Deputy Speaker</b> (Prof. Kaloki): Hon. Member, you can proceed but just </text>',
                'prev_entry': Entry(
                    text='mashtaka. Naomba kupinga.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'The Temporary Deputy Speaker (Prof. Kaloki)',
                'text': 'Hon. Member, you can proceed but just'
            },
            {
                'line': '<text top="106" left="191" width="590" height="15" font="2"><b>(Prof. Kaloki)</b> continued saying some stuff... </text>',
                'prev_entry': Entry(
                    text='mashtaka. Naomba kupinga.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'Prof. Kaloki',
                'text': 'continued saying some stuff...'
            },
            {
                'line': '<text top="145" left="191" width="590" height="15" font="2"><b>The  Assistant  Minister,  Ministry  of  State  for  Public  Service</b> (Maj. Sugow): Mr. </text>',
                'prev_entry': Entry(
                    text='take one minute. You can see the mood of the House, according to Mr. Midiwo.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'The Assistant Minister, Ministry of State for Public Service (Maj. Sugow)',
                'text': 'Mr.'
            },
            {
                'line': '<text top="519" left="191" width="590" height="15" font="2"><b>The Minister for Justice, National Cohesion and Constitutional  Affairs </b>(Ms. Karua): </text>',
                'prev_entry': Entry(
                    text='upon to reply, put and agreed to)',
                    entry_type='scene',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'The Minister for Justice, National Cohesion and Constitutional Affairs (Ms. Karua)',
                'text': ''
            },
            {
                'line': '<text top="935" left="171" width="605" height="13" font="2"><b>The  Member  for  Maragwa </b>(Mr. P.K. Mwangi): Thank you, Mr. Temporary Deputy Speaker, Sir, for</text>',
                'prev_entry': Entry(
                    text='Mr. Temporary Deputy Speaker, Sir, with those few remarks, I beg to support.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'The Member for Maragwa (Mr. P.K. Mwangi)',
                'text': 'Thank you, Mr. Temporary Deputy Speaker, Sir, for'
            },
            {
                'line': '<text top="783" left="171" width="283" height="13" font="2"><b>Mr. Mutahi: </b>Wacha mambo ya maiden speech!</text>',
                'prev_entry': Entry(
                    text='I am making my maiden speech!',
                    entry_type='speech_new',
                    speaker='mr. arap-kirui'),
                'entry_type': 'speech_new',
                'speaker': 'Mr. Mutahi',
                'text': 'Wacha mambo ya maiden speech!'
            },
            {
                'line': '<text top="749" left="171" width="430" height="13" font="2"><b>Mr. Munyasia: </b>On a point of order, Mr. Temporary Deputy Speaker, Sir.</text>',
                'prev_entry': Entry(
                    text='older Members, who seem to suffer from a great deal of hangover from the old Seventh Parliament.',
                    entry_type='speech_new',
                    speaker=None),
                'entry_type': 'speech_new',
                'speaker': 'Mr. Munyasia',
                'text': 'On a point of order, Mr. Temporary Deputy Speaker, Sir.'
            },
        ]

    def test_process_meta(self):
        for filename in self.filenames:
            print(filename)
            # parser = self.parser(verbose=False)
            parser = self.parsers[filename]
            metadata = parser._process_meta(max_check=50)
            # sittings[key] = metadata
            for key, value in self.meta[filename].items():
                self.assertEqual(value, metadata.__getattribute__(key))

    def test_process_transcript(self):
        for filename in self.filenames:
            # contents = soup.body.contents
            parser = self.parsers[filename]
            metadata, contents = parser.process_transcript(
                file_path=os.path.join(self.base_dir, filename),
                save_soup=False,
                to_format='df-long',
            )
            print(filename)
            # no speaker names should have a weekday or month name in them.
            self.assertEqual(contents.speaker.apply(lambda x: x in self.date_strings).sum(), 0)
            # no speaker name should be 'th', 'st', 'nd', or 'rd'.
            self.assertEqual(contents.speaker.apply(lambda x: x in self.day_endings).sum(), 0)
            # no speaker names should have digits in them.
            self.assertEqual(contents.speaker.apply(lambda x: bool(re.search(r'\d+', x)) if x is not None else False).sum(), 0)
            # no entries should have text == None.
            self.assertEqual(contents.text.apply(lambda x: pd.isnull(x) or len(x) == 0).sum(), 0)
            # no entry types should be None.
            self.assertEqual(contents.entry_type.isnull().sum(), 0)

    def test_xml_get_entry_type(self):
        for l, true_entry_type in self.entry_types:
            line = BeautifulSoup(l, 'xml').contents[0]
            entry_type = self.parser(verbose=False)._get_entry_type(line)
            # print(line)
            self.assertEqual(entry_type, true_entry_type)

    def test_xml_get_entry(self):
        for data in self.speaker_data:
            line = BeautifulSoup(data['line'], 'xml').contents[0]
            parser = self.parser(verbose=False)
            entry_type = parser._get_entry_type(line)
            speaker_name = parser._get_speaker_name(line, entry_type, data['prev_entry'])
            # print(data['speaker'], '\t', speaker_name)
            text = parser._get_text(line)
            # print(data['text'], '\t', text)
            self.assertEqual(entry_type, data['entry_type'])
            self.assertEqual(speaker_name, data['speaker'])
            self.assertEqual(text, data['text'])


if __name__ == '__main__':
    unittest.main()
