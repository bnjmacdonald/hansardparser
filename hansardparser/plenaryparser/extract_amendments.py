"""extracts records of amendments from speeches.

Legislation schema::
    
    {
        "name": String representing name of legislation.
        "amending": [ObjectId]. Array of ObjectIds representing pieces of
            legislation that this legislation directly affects/amends.
        "actions": [{'date': Date, 'type': String}]. Array of actions
            representing when action was taken on the legislation.
            'date': Date representing date of action.
            'type': String representing type of action. One of: ["introduced",
                "first reading", "second reading", "amended", "third reading",
                "passed", "assented", ...]
        "private": Boolean representing whether this is a private member's
            bill.
        "omnibus": Boolean representing whether this is an omnibus bill.
        "persons": [ObjectId]. Array of ObjectIds representing persons
            who introduced the legislation.
        "sponsors": [ObjectId]. Array of ObjectIds representing persons who
            sponsored the legislation.
    }

Amendment schema::
    
    {
        "legislation": ObjectId representing _id of legislation
        "text": String representing initial text of amendment
        "section": String representing clause/section/part number
        "person": ObjectId representing _id of person who proposed amendment.
        "parent": ObjectId to amendment that is being countered, if this
            amendment is a cunter-amendment (i.e. amendment to an amendment).
            If this is not a counter-amendment, leave null.
        "outcome": String representing result of amendment. See
            `get_vote_result`.
        "vote": ObjectId representing _id of vote on amendment.
        "speeches": [ObjectId]. Array of ObjectIds representing speeches
            associated with amendment.
    }


Notes
-----

Finding PDFs containing amendments:
    
    {"$or": [{"header": {"$regex": "^Clause", "$options": "i"}}, {"subheader": {"$regex": "^Clause", "$options": "i"}}, {"subsubheader": {"$regex": "^Clause", "$options": "i"}}]}
    {"type": "scene", "text": {"$regex": "^clause", "$options": "i"}}
    {"type": {"$in": ["header", "subheader", "subsubheader"]}, "text": {"$regex": "committee"}}
    {"type": {"$in": ["speech_new", "speech_ctd"]}, "text": {"$regex": "^clause", "$options": "i"}}
    {"type": "scene", "text": {"$regex": ^question of the"}}
    {"type": "scene", "text": {"$regex": ^put and agreed to"}}

Examples of PDFs containing amendments:
[
    "Hansard_Report_-_Wednesday_4th_March_2015_A_-1.pdf", 
    "Hansard_Report_-_Wednesday__18th_November_2015A.pdf"
]

Todo
----

- clean subheaders, which is a major problem in leading amendments to fail to link to a bill.
- revise extrction of amendment text by cutting off text after amendments that have the "clause X be deleted" structure.

"""


# PSEUDO-CODE
# for all speeches on [date]:
#     subset to speeches that are part of committee of the whole house.
#     group by bill, clause
#     for each bill...
#         upsert bill
#         for each clause...
#             for each speech/scene in clause...
#                 get class of speech/scene
#                 process class

import argparse
import re
from string import punctuation
import pytz
import pprint

from server.hansardparser.extract_bills import get_legislation, clean_title, header_prefixes
from server.db import connect
db = connect()

question_headers = ['oral answers to questions', 'questions by private notice', 'question by private notice', 'oral answers to', 'oral answer to question', 'questions by private notice registered owner of land parcel lr.10743 in thika']
motion_headers = ['motions', 'motion', 'notices of motions', 'notice of motion']
procedural_headers = ['quorum', 'motion for the adjournment', 'adjournment', 'point of order', 'points of order', 'motion for the adjournment under standing order no.20', 'procedural motion', 'motion for adjournment']
statement_headers = ['ministerial statement', 'ministerial statements', 'statements', 'personal statement', 'statements health status of the right honourable prime minister']
paper_headers = ['papers laid']
report_headers = ['report, consideration of report and third reading', 'report', 'report and third reading']
other_headers = ['communication from the chair', 'communications from the chair', 'committee of the whole house', 'committee of the whole', 'in the committee', 'preliminary', 'prime ministers time', 'noes:', 'abstentions:']
header_type_dict = {'question': question_headers,
    'motion': motion_headers,
    'bill': [],
    'procedural': procedural_headers,
    'statement': statement_headers,
    'paper': paper_headers,
    'report': report_headers,
    'other': other_headers
}

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
    # TEMPORARY
    db.amendments.delete_many({})
    args = parse_args()
    # regex_legislation_header = re.compile('|'.join(bill_headers), re.IGNORECASE)
    # regex_committee_whole_house = re.compile('', re.IGNORECASE)
    amendment_subsubheaders = ['clause']
    regex_amendment_subsubheader = re.compile('|'.join(amendment_subsubheaders), re.IGNORECASE)
    pipeline = [
        {"$match": {
            "$or": [
                # {"header": regex_legislation_header},
                # {"subheader": regex_legislation_header},
                {"subsubheader": regex_amendment_subsubheader}
            ]
        }},
        {"$group": {
            "_id": {"header": "$header", "subheader": "$subheader", "clause": "$subsubheader"},
            "speeches": {
                '$push': {"_id": "$_id", "person": "$person", "text": "$text", "type": "$type", "date": "$date"}
            }
        }}
    ]
    if args.verbosity:
        print('Aggregating speeches...')
    agg = db.speeches.aggregate(pipeline)
    bill = None
    for group in agg:
        bill_title, prefix, suffix = clean_title(group['_id']['subheader'], prefixes=header_prefixes)
        # if encountering different bill, retrieve the bill.
        if bill is None or bill_title != bill['title']:
            if args.verbosity:
                print('Extracting amendments from {0}'.format(bill_title))
            bill = get_legislation(bill_title, args.verbosity)
            if bill is None:
                if args.verbosity:
                    print('WARNING: Bill not found for: {0}'.format(group['_id']))
                continue
            # TODO: create bill.
        bill_section = get_section(group['_id']['clause'])
        amendments = []
        amendment_fields = dict(
            date=pytz.utc.localize(group['speeches'][0]['date']),
            legislation=bill['_id'],
            text=None,
            section=bill_section,
            person=None,
            parent=None,
            outcome=None,
            vote=None,
            speeches=[],
        )
        amendment = amendment_fields.copy()
        for speech in group['speeches']:
            appended = False
            classname, result = get_class(speech)
            # if amendment already has text, then this must be a further amendment to the same clause.
            if classname == 'amendment_text' and amendment['text'] is not None:
                amendments.append(amendment.copy())
                appended = True
                amendment = amendment_fields.copy()
            amendment = process_class(amendment, classname, result, speech, args.verbosity)
            amendment['speeches'].append(speech['_id'])
        if not appended:
            amendments.append(amendment.copy())
        for amendment in amendments:
            update_result = db.amendments.update_one(amendment, {'$set': amendment}, upsert=True)
            # if args.verbosity > 1:
            #     if update_result.upserted_id:
            #         print('Added a new amendment to {0}.'.format(bill_title))
            #         # pprint.pprint(amendment)
            #     else:
            #         print('Modified {0} amendments for {1}.'.format(update_result.modified_count, bill_title))
    if args.verbosity:
        print('Success!')
    return 0

def get_section(name):
    # section_num = None
    # section_regex = re.compile(r'clause\s+(?P<num>\d{1,4})', re.IGNORECASE)
    # result = section_regex.search(name)
    # if result:
    #     section_num = int(result.group('num'))
    name = re.sub('\s+', ' ', name.strip().lower())
    return name

def get_class(speech):
    """retrieves the class of a speech under a clause subsubheading.""
    
    Possible classes:
        amendment_text: text of an amendment (usually begins with "THAT")
        speech: debate and other speeches related to an amendment.
        vote: action taken on an amendment
        # clause: integer representing current clause of bill being debated.
    
    Arguments:
        speech: dict representing a speech object.

    Returns:
        classname: String representing class. One of:
            ["amendment_text", "speech", "proposed", "vote"]
        result: String representing result associated with classname.
    """
    classname = None
    result = None
    amend_text = get_amendment_text(speech['text'])
    if amend_text:
        classname = 'amendment_text'
        result = amend_text
    if classname is None:
        vote_result = get_vote_result(speech)
        if vote_result:
            classname = "vote"
            result = vote_result
    if classname is None:
        proposed = is_proposed(speech)
        if proposed:
            classname = 'proposed'
            result = proposed
    if classname is None:
        result = speech['text']
        classname = 'speech'
    return classname, result


def process_class(amendment, classname, result, speech, verbose):
    assert isinstance(amendment, dict), 'amendment must be a dict.'
    # if is_subamendment(speech):
    if classname == 'amendment_text':
        if 'text' in amendment and amendment['text'] not in [None, '']:
            if verbose:
                print('WARNING: Amendment already has text.')
                pprint.pprint({k: amendment[k] for k in amendment if k != 'speeches'})
                pprint.pprint(speech)
        if amendment['person'] is not None:
            if verbose:
                print('Amendment already has person. Amendment:')
                pprint.pprint({k: amendment[k] for k in amendment if k != 'speeches'})
                pprint.pprint(speech)
        amendment['text'] = result
        amendment['person'] = speech['person']
    elif classname == 'vote':
        if 'outcome' in amendment and amendment['outcome'] not in [None, ''] and result != amendment['outcome']:
            print('\nAmendment already has vote outcome and it differs from current outcome.')
            print('New vote outcome: {0}'.format(result))
            print('Speech: ')
            pprint.pprint(speech)
            print('Amendment: ')
            pprint.pprint({k: amendment[k] for k in amendment if k != 'speeches'})
        amendment['outcome'] = result
    elif classname == 'proposed':
        pass
    elif classname == 'speech':
        pass
    else:
        raise RuntimeError('classname "{0}" not recognized'.format(0))
    return amendment

def get_amendment_text(text):
    if text is None:
        return None
    amend_text = None
    amendment_text_regex = re.compile(r'(?:.*)(I beg to move|THAT)(?P<text>.+)', re.DOTALL)
    regex = re.search(amendment_text_regex, text)
    if regex:
        amend_text = regex.group('text')
        amend_text = re.sub('^[{0}]'.format(re.escape(punctuation)), '', amend_text).strip()
    return amend_text

def get_vote_result(speech):
    """
    Returns:
        result: String representing result of amendment. One of: ["passed",
            "failed", "withdrawn", "dropped"]. "dropped" represents
            amendments that were not voted on for one reason or another (e.g.
            decision by Chair to drop amendment).
    """
    # vote_result = None
    if speech['type'] != 'scene':
        return None
    passed = bool(re.search('agreed', speech['text'], re.IGNORECASE))
    failed = bool(re.search('negatived', speech['text'], re.IGNORECASE))
    withdrawn = bool(re.search('withdraw', speech['text'], re.IGNORECASE))
    dropped = bool(re.search('dropped', speech['text'], re.IGNORECASE))
    test_results = {
        'passed': passed,
        'failed': failed,
        'withdrawn': withdrawn,
        'dropped': dropped
    }
    if sum(test_results.values()) > 1:
        raise RuntimeError('Multiple vote results: {0}'.format(', '.join([k for k, v in test_results.items() if v])))
    # if sum(test_results.values()) == 0:
    #     return None
    for k, v in test_results.items():
        if v:
            return k
    return None

def is_subamendment(speech):
    result = False
    if speech['text'] is None:
        return result
    return bool(re.search(r'further amended', speech['text'], re.IGNORECASE))

def is_proposed(speech):
    if speech['type'] != 'scene':
        return False
    proposed = bool(re.search('proposed', speech['text'], re.IGNORECASE))
    return proposed


if __name__ == '__main__':
    main()
