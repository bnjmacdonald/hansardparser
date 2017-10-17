"""extracts bills from speech documents.

Notes
-----

header VERY RARELY contains the name of a bill. Header is usually high-level,
like "bills" or "committee of the whole house".

- Exceptions: ["mediated version of the public audit bill, 2014", ...?]

Examples of transcripts containing information on bills: ["23rd april 1998P.pdf",
"30th april 1998P.pdf", "18th June, 2002P.pdf", "4th July, 2007A.pdf", "7th December, 2004P.pdf", "19th April, 2000A.pdf"]

Todo
----

* divide budget bills by year? (i.e. Finance bill 2006 != Finance bill 2007)
* in update_amending_legislation, if bill is not found then add one.

"""

import re
import argparse
import pytz
import pprint
from string import punctuation
from collections import OrderedDict

from server.hansardparser import utils
from server.rawdatamgr.match import Matcher
from server.db import connect
db = connect()

# prefixes on subheaders to remove.
header_prefixes = sorted(['adoption of mediation committee report on', 'adoption of report on', 'amendments to', 'appointment of members to mediation committee on', 'approval of', 'approval of the mediated version of', 'clarifications on', 'closure of debate on', 'consideration and approval of', 'consideration of', 'consideration of a', 'consideration of senate amendments', 'consideration of the senate amendments to', 'constitutionality of', 'constitutionality of debate on', 'constitutionality of provisions in', 'date on', 'debate on', 'decision of the senate on', 'deferment of committee of the whole house on', 'deferment of committee stage', 'deferment of committee stage of', 'deferment of consideration of presidential memorandum on', 'deferment of debate on', 'deferment of second reading', 'deferrment of', 'deferment of', 'deferrement of', 'delay in allocating time to', 'delay in issuance of ministerial statements on fate of', 'delayed approval of', 'delayed assent to', 'delayed introduction of', 'determination of whether', 'enactment of', 'enactment of legislation to domesticate', 'eservations on', 'eservations to', 'exemption of', 'extension of debate on', 'fast-tracking of', 'fate of', 'guidance on proposed amendments to', 'introduction of', 'joint report on', 'leave to introduce', 'leave to introduce a', 'legality /constitutionality of', 'legality of discussing', 'limitation of debate', 'mediated version of', 'message on', 'move to take', 'notification of intent to withdraw sections of', 'onsiderations of reports and third readings', 'passage of', 'pertinent issues on referral of', 'presidential assent to', 'presidential memorandum on', 'preparation of', 'progress report on', 'publication of', 'reduction of publication period', 'reduction of publication period of', 'reduction of referral period of', 'referral of', 'refusal to assent to', 'reinstatement of', 'removal of', 'report of committee of the whole house on', 'report of the committee of the whole house on', 'report of the mediation committee on', 'report and third reading', 'reports and third reading', 'reservations to', 'ruling on whether the senate should proceed with', 'second reading of', 'senate amendments to', 'senate \' samendments to', 'senate samendments to', 'status of', 'the bill was read a second time and committed to a committee of the whole house tomorrow', 'the memorandum from h. e the president on', 'the president to', 'the president \' sr eservation', 'the president sr eservation', 'third reading', 'unprocedural presentation of', 'waiving of referral period of', 'waving of referral period of', 'withdrawal of'], key=lambda x: len(x), reverse=True)

header_suffixes = ['first reading', 'second reading']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        default=0,
        help='verbosity'
    )
    args = parser.parse_args()
    return args

def main():
    # TEMPORARY.
    db.legislation.delete_many({})
    args = parse_args()
    # max_length = 100
    # min_length = 8
    # subheaders_skip = ['no_subheading', 'in the committee']
    # boundary = 95
    # queries database.
    bill_headers = ['bills', 'bill', 'in the committee', 'committee of the whole house', 'first reading', 'second reading', 'third reading', 'progress reported', 'guillotine']
    bill_subheaders = [r'\bbill\b', r'\bamendment\b', 'first reading', 'second reading', 'third reading']  # r'^the '
    # false_positive_headers = ['questions by private notice', 'question by private notice', 'personal statement']
    # false_positive_subheaders = ['']
    regex_bill_header = re.compile('|'.join(bill_headers), re.IGNORECASE)
    regex_bill_subheaders = re.compile('|'.join(bill_subheaders), re.IGNORECASE)
    # bill_text = ['grant leave to introduce a bill']
    # regex_bill_text = re.compile('|'.join(bill_text), re.IGNORECASE)
    # pipeline1 = [
    #     {'$match': {"text": regex_bill_text}},
    #     {''}
    # ]
    match = {
        "header": regex_bill_header,
        "subheader": regex_bill_subheaders,
    }
    sort = OrderedDict([('date', 1), ('header', 1), ('subheader', 1)])
    group = {
        "_id": {"header": "$header", "subheader": "$subheader", "date": "$date"},
        "speeches": {'$push': '$_id'}
    }
    pipeline = [
        {'$match': match},
        {'$sort': sort},
        {'$group': group},
    ]
    if args.verbosity > 1:
        print('Aggregating speeches...')
    agg = db.speeches.aggregate(pipeline)
    # speeches = db.speeches.find(match, sort=[('header', 1), ('subheader', 1)])
    # titles = []
    for group in agg:
        # print(group['_id']['header'] + ': ' + group['_id']['subheader'])
        date = pytz.utc.localize(group['_id']['date']) if group['_id']['date'] is not None else None
        subheader = group['_id']['subheader']
        if len(group['speeches']) == 0:  # or subheader in subheaders_skip
            if args.verbosity > 1:
                print('Skipping: {0}'.format(group['_id']))
            continue
        # 'communications from the chair'
        # cleans bill title.
        title, prefix, suffix = clean_title(subheader, prefixes=header_prefixes, suffixes=header_suffixes)
        # if len(title) > max_length or len(title) < min_length:
        #     if args.verbosity > 1:
        #         print('Skipping: {0}'.format(group['_id']))
        #     continue
        # if args.verbosity:
        #     print('\nProcessing header: "{0}"'.format(group['_id']['header']))
        #     print('Subheader: "{0}" => Bill title: "{1}"'.format(subheader, title))
        # check if bill exists.
        try:
            bill = get_legislation(title, args.verbosity)
            bill_id = bill['_id']
            # if not exists, insert bill.
        except TypeError:
            bill = dict(
                title=title,
                number=None,
                gazette_num=None,
                actions=[],
                omnibus=is_omnibus_bill(title),
                budget=is_budget_bill(title),
                amendment=is_amendment_bill(title),
                private=None,
                owners=[],
                sponsors=[],
                amending=[],
                # mentions=[{'date': date, 'speeches': group['speeches']}],
                remarks=None,
            )
            
            bill_id = db.legislation.update_one(bill, {'$set': bill}, upsert=True).upserted_id
            if bill_id and args.verbosity:
                print('Created bill with title: {0}'.format(title))
            if bill_id is None:
                raise RuntimeError('A bill should have been created. Something went wrong.')
        # add action to bill.
        speeches = list(db.speeches.find({'_id': {'$in': group['speeches']}}))
        header = group['_id']['header'] + suffix if suffix is not None else group['_id']['header']  # note: kludge.
        action_parser = ParseBillAction(header=header, subheader=group['_id']['subheader'], speeches=speeches)
        action = action_parser.get_action()
        if args.verbosity and action is None:
            print('WARNING: bill action is None.')
            pprint.pprint(bill)
            pprint.pprint(action)
        update_result = db.legislation.update_one({'_id': bill_id}, {'$push': {'actions': action}}, upsert=False)
        if args.verbosity and update_result.modified_count == 0:
            print('WARNING: bill was not updated with action.')
            pprint.pprint(bill)
            pprint.pprint(action)
    # sorts actions by date.
    db.legislation.update_many({}, {'$push': {'actions': {'$each': [], '$sort': {'date': 1}}}}, upsert=False)
    # adds amending legislation to each bill.
    update_amending_legislation(db.legislation.find({'amendment': True}, {'title': 1}))
    if args.verbosity:
        print('Success!')
    return 0

def clean_title(title, prefixes=[], suffixes=[]):
    """cleans bill title.
    
    Arguments:
        title: String representing uncleaned bill title.
            example: "amendments to the health bill, 2015"
        prefixes: list of String representing prefixes that should be removed
            from bill title (e.g. "status of", "amendments to").
        suffixes: list of String representing suffixes that should be removed
            from bill title.
    Note: header_prefixes should be sorted from shortest to longest string so
    that longest gets matched first.

    Returns:
        title: String representing cleaned bill title.
            example: "the health bill, 2015"
    """
    prefix = None
    suffix = None
    prefix_regex = re.compile(r'(?:.*)(?P<prefix>{0})(?P<title>.+)(?P<suffix>{1})'.format('|'.join(prefixes), '|'.join(suffixes)), re.DOTALL)
    regex = re.search(prefix_regex, title)
    if regex:
        title = regex.group('title')
        prefix = regex.group('prefix').strip()
        suffix = regex.group('suffix').strip()
        title = re.sub('^[{0}]'.format(re.escape(punctuation)), '', title).strip()
    return title, prefix, suffix


def get_legislation(title, verbose=0):
    """retrieves a bill based on a title string using fuzzy matching.

    Arguments:
        title: String representing title of a bill.

    Returns:
        bill: dict representing a piece of legislation.
    """
    bill = None
    choices = [t for t in db.legislation.find({}, {'_id': 1, 'title': 1}, sort=[('title', 1)])]
    matcher = Matcher(
        boundary=95,
        limit=10,
        num_return=None,
        scorer='WRatio',
        library='fuzzywuzzy',
        verbose=0
    )
    matched_choice, score, _, _, top_matches = matcher.find_matches(values_dict={'title': title.strip().lower()}, choices=choices)
    if matched_choice and sum([is_amendment_bill(title), is_amendment_bill(matched_choice['title'])]) in [0, 2]:
        # NOTE: kludge. 
        bill = db.legislation.find_one({'_id': matched_choice['_id']})
        if verbose > 1:
            print('{0} matched to {1} (score: {2}).'.format(title, bill['title'], score))
        if sum([is_amendment_bill(title), is_amendment_bill(bill['title'])]) == 1:
            if verbose:
                print('\nWARNING: Amendment bill is being matched to an original bill (score: {0}).\nBill 1: {1}.\nBill 2: {2}'.format(score, bill['title'], title))
            bill = None
    return bill


def update_amending_legislation(bills):
    """for each bill in bills, updates array of legislation that bill is
    amending.
    """
    for bill in bills:
        amending = get_amending_legislation(bill['title'])
        if len(amending):
            update_result = db.legislation.update_one(bill, {'$addToSet': {'amending': {'$each': amending}}}, upsert=False)
            if not update_result.matched_count > 0:  # modified_count
                raise RuntimeError('A bill should have been found. Something went wrong.')
    return 0

def get_amending_legislation(title):
    """retrieves legislation being amended by bill.
    
    Todo
    ----
    * add other rules for finding amending legislation?
    """
    amending_bills = []
    if is_amendment_bill(title):
        amending_regex = re.compile(r'\(amendment\)|amendment', re.IGNORECASE)
        temp_title = re.sub('\s+', ' ', amending_regex.sub('', title))
        amending_bill = get_legislation(temp_title)
        if amending_bill is not None:
            amending_bills.append(amending_bill['_id'])
    return amending_bills

def is_bill_title(s):
    """returns true if String is a bill title, False otherwise."""
    title_regex = re.compile(r'bill|amendment', re.IGNORECASE)
    return bool(title_regex.search(s))

def is_amendment_bill(title):
    """returns True if bill is an amendment bill, False otherwise."""
    regex = re.compile('amendment', re.IGNORECASE)
    return bool(regex.search(title))

def is_omnibus_bill(title):
    """returns True if bill is an omnibus bill.

    Note: should I count budget bills as omnibus bills?
    """
    omnibus_kws = ["statute law", "miscellaneous"]
    omnibus_regex = re.compile('|'.join(omnibus_kws), re.IGNORECASE)
    return bool(omnibus_regex.search(title))

def is_budget_bill(title):
    """returns True if bill is an omnibus bill.

    Note: should I count budget bills as omnibus bills?
    """
    budget_kws = [r"the finance bill", r"appropriation(s)? bill"]
    budget_regex = re.compile('|'.join(budget_kws), re.IGNORECASE)
    return bool(budget_regex.search(title))


class ParseBillAction(object):
    """The ParserBillAction class contains methods for extracting a "bill
    action" from a set of speeches."""
    def __init__(self, header, subheader, speeches):
        # self.bill = bill
        self.header = header
        self.subheader = subheader
        self.speeches = speeches
    def get_action(self):
        """retrieves one or more actions on a bill from a list of speeches.
        """
        action = {}
        is_action = False
        action_tests = [
            ('introduced', self._is_action_introduced),
            ('first_reading', self._is_action_first_reading),
            ('second_reading', self._is_action_second_reading),
            ('third_reading', self._is_action_third_reading),
            ('committee_whole', self._is_action_committee_of_whole),
            # ('assent', self._is_action_assented),
        ]
        while is_action is False and len(action_tests):
            action_name, action_test = action_tests.pop(0)
            is_action = action_test()
        if is_action:
            action['name'] = action_name
            action['date'] = self._get_action_date()
            action['speeches'] = [speech['_id'] for speech in self.speeches]
            action['outcome'] = self._get_action_outcome(action_name)
        self.action = action
        return action
    def _get_action_date(self):
        date = None
        uniq_dates = list(set([speech['date'] for speech in self.speeches]))
        if len(uniq_dates) > 1:
            raise RuntimeError('self.speeches contains multiple unique dates.')
        if uniq_dates[0] is not None:
            date = pytz.utc.localize(uniq_dates[0])
        return date
    def _get_action_outcome(self, action_name):
        """
        Returns:
            result: String representing outcome of action. One of: ["passed",
                "failed", "withdrawn", "dropped", None]. "dropped" represents
                amendments that were not voted on for one reason or another (e.g.
                decision by Chair to drop amendment).
        """
        if action_name == 'introduced':
            outcome = self._get_introduced_outcome()
        elif action_name == 'first_reading':
            outcome = self._get_first_reading_outcome()
        elif action_name == 'second_reading':
            outcome = self._get_second_reading_outcome()
        elif action_name == 'third_reading':
            outcome = self._get_third_reading_outcome()
        elif action_name == 'committee_whole':
            outcome = self._get_committee_whole_outcome()
        else:
            raise RuntimeError('action_name "{0}" not recognized'.format(action_name))
        return outcome
    def _get_introduced_outcome(self):
        """
        TODO:
            * check for other possibilities  (e.g. withdrawn).
        """
        return 'passed'
    def _get_first_reading_outcome(self):
        regex_passed = re.compile(r'orders for first reading(s)? read|read the first time|referred to the relevant departmental committee', re.IGNORECASE)
        # Ordered to be read the Second Time tomorrow
        for speech in self.speeches:
            if speech['type'] == 'scene':
                if regex_passed.search(speech['text']):
                    return 'passed'
                # todo: add tests for negatived, withdrawn, dropped.
        return None
    def _get_second_reading_outcome(self):
        regex_passed = re.compile(r'read a second time|committed to a committee of the whole house', re.IGNORECASE)
        # put and agreed to?
        for speech in self.speeches:
            if speech['type'] == 'scene':
                if regex_passed.search(speech['text']):
                    return 'passed'
                # todo: add tests for negatived, withdrawn, dropped.
        return None
    def _get_third_reading_outcome(self):
        regex_passed = re.compile(r'read (the|a) third time and passed', re.IGNORECASE)
        for speech in self.speeches:
            if speech['type'] == 'scene':
                if regex_passed.search(speech['text']):
                    return 'passed'
                # todo: add tests for negatived, withdrawn, dropped.
        return None
    def _get_committee_whole_outcome(self):
        """
        Returns:
            outcome: string. Either "with amendment", "
                without amendment", or None.
        """
        regex = re.compile(r'beg to move that the committee doth report to the house its consideration of .+ and its approval thereof (?P<outcome>with(out)? amendment)', re.IGNORECASE)
        for speech in self.speeches:
            if speech['type'] == 'scene':
                regex_result = regex.search(speech['text'])
                if regex_result:
                    return regex_result.group('outcome')
        return None
    def _is_action_introduced(self):
        regex_introduced = re.compile(r'grant(s)? leave to introduce a bill', re.IGNORECASE)
        for speech in self.speeches:
            regex_result = regex_introduced.search(speech['text']) if speech['text'] else None
            if regex_result:
                return True
        return False
    def _is_action_first_reading(self):
        regex = re.compile(r'first reading', re.IGNORECASE)
        return bool(regex.search(self.header) or regex.search(self.subheader))
    def _is_action_second_reading(self):
        regex = re.compile(r'second reading', re.IGNORECASE)
        return bool(regex.search(self.header) or regex.search(self.subheader))
    def _is_action_third_reading(self):
        regex = re.compile(r'third reading', re.IGNORECASE)
        return bool(regex.search(self.header) or regex.search(self.subheader))
    def _is_action_committee_of_whole(self):
        regex = re.compile(r'committee of the whole|in the committee', re.IGNORECASE)
        return bool(regex.search(self.header) or regex.search(self.subheader))
    def _check_gazetted(self):
        """
        TODO:
            * implement this.
        """
        raise NotImplementedError
    def _check_assented(self):
        """checks whether a bill has been assented to.

        TODO:
            * write http request to check kenya law database for this?
            * for more recent legislation, check president's website?
        """
        raise NotImplementedError


if __name__ == '__main__':
    main()
