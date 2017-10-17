"""inserts speeches into the database.

    
Usage
-----

Example::
    
    python -m server.hansardparser.insert_speeches -i /Volumes/Transcend/HANSARDS/{1998..2013} ../data/raw/transcripts -m "../data/manual/matched_names.csv" -v 1

Notes
-----

"""

import os
import sys
import time
import re
import argparse
import pandas as pd
import numpy as np

import settings
from server.rawdatamgr import utils
from server.hansardparser.XmlParser import XmlParser
from server.hansardparser.utils import get_file_paths
from server.db import connect
db = connect()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="output verbosity (integer)"
    )
    parser.add_argument(
        '-i'
        '--input',
        nargs='+',
        dest='input',
        required=True,
        help='List of input directories (where transcripts exist).',
    )
    parser.add_argument(
        '-m'
        '--matchespath',
        type=str,
        dest='matches_path',
        default=os.path.join(settings.BASE_DIR, 'data', 'manual', 'matched_names.csv'),
        help='Output directory (where to save parsed transcripts if --savesoup flag is selected).',
    )
    parser.add_argument(
        '-s'
        '--savesoup',
        action='store_true',
        dest='savesoup',
        default=False,
        help='Save parsed html/txt to disk.',
    )
    parser.add_argument(
        '-o'
        '--soupoutput',
        type=str,
        dest='soupoutput',
        default=os.path.join(settings.BASE_DIR, 'data', 'generated', 'parsed_transcripts'),
        help='Output directory to save parsed transcripts (only used if --savesoup flag is selected).',
    )
    parser.add_argument(
        '-d'
        '--datefmt',
        type=str,
        dest='date_format',
        default='%Y-%m-%d',
        help='Date format.',
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    msg = '\nYou are about to insert documents into the speeches collection. This may affect existing speeches. \n\nDo you wish to continue (yes/no)? '
    r = input(msg)
    if r != 'yes':
        print('cancelled')
        sys.exit(0)
    if r == 'yes':
        # constructs list of input_dirs where transcripts are located.
        # input_dirs = [os.path.join('/Volumes/Transcend/HANSARDS', str(year)) for year in [1998, 1999, 2000, 2001, 2002]]
        verbose = args.verbosity
        input_dirs = []
        for input_dir in args.input:
            if not input_dir.startswith('/'):
                input_dir = os.path.join(settings.BASE_DIR, input_dir)
            input_dirs.append(input_dir)
        file_paths = get_file_paths(input_dirs, args.verbosity)
        matched_names = pd.read_csv(args.matches_path)
        matched_names.replace(np.nan, '', inplace=True)
        matched_names = {(row['speaker'], row['parliament']): row['person__name'] for i, row in matched_names[matched_names.match == '1'].iterrows()}
        time0 = time.time()
        # parses each transcript and inserts speeches.
        for i, file_path in enumerate(file_paths):
            # parses transcript
            parser = XmlParser(verbose=verbose)
            metadata, contents = parser.process_transcript(
                file_path=file_path,
                save_soup=args.savesoup,
                path=args.soupoutput,
                rm_whitespace=True,
                append_meta=True,
                to_format='df-long'
            )
            if contents is None:
                print('WARNING: no speeches returned from parser.process_transcript in {0}'.format(file_path))
                continue
            contents['transcript'] = file_path.split('/')[-1]
            # checks if speeches already exist in database for this data.
            if verbose:
                uniq_dates = [utils.date_from_str(datestr, [args.date_format]) for datestr in contents['date'].unique()]
                for date in uniq_dates:
                    if db.speeches.find({'date': date}).count() > 0:
                        print('WARNING: speeches already exist in database on {0}'.format(date))
            # inserts speeches.
            speeches = list(contents.to_dict(orient='index').values())
            insert_speeches(speeches, matched_names, args.date_format, verbose)
            if verbose:
                time_temp = time.time()
                print('Finished processing {0} of {1} files...'.format(i+1, len(file_paths)))
                print('Elapsed time: {0:.2f} minutes\n'.format(((time_temp - time0)/60.0)))
        if verbose:
            time1 = time.time()
            print('\nSuccessfully parsed filenames:')
            print('\t\n'.join(file_paths))
            print('\nSuccessfully inserted speeches from {0} transcripts.'.format(len(file_paths)))
            print('Total time: {0:.2f} minutes.'.format((time1 - time0)/60.0))
    return 0

def insert_speeches(speeches, matched_names, date_format, verbose):
    """inserts each speech in speeches into database.

    Parameters:
        speeches: list-like iterable or generator that yields a dict-like
            speech containing fields 'text', 'header', et cetera.
            example::

                [
                    {'text': 'blah blah blah', 'speaker_cleaned': 'raila', ...},
                    {'text': 'blah blah blah', 'speaker_cleaned': 'musyoka', ...},
                ]

        matched_names: see `assign_person_to_speech` method.
        date_format: String representing date format to use in 
            utils.date_from_str.
    """
    # date = _get_date(speeches, date_format)
    speech_before_id = None
    for speech in speeches:
        # if date == 'multiple_dates':
        date = utils.date_from_str(speech['date'], [date_format])
        new_speech = dict(
            date=date,
            type=speech['entry_type'],
            text=speech['text'],
            header=speech['header'],
            subheader=speech['subheader'],
            subsubheader=speech['subsubheader'],
            speech_before=speech_before_id,
            page_num=int(speech['page_number']) if speech['page_number'] else None,
            raw_meta={k: speech[k] for k in speech if k not in ['text', 'header', 'subheader', 'subsubheader', 'page_number', 'entry_type', 'date']},
            transcript=speech['transcript']
        )
        # retrieves person and inserts speech.
        person_id = get_person_id(new_speech['raw_meta'], matched_names, verbose)
        new_speech['is_chair'] = get_is_chair(new_speech['raw_meta'])
        new_speech['person'] = person_id
        speech_id = db.speeches.insert_one(new_speech).inserted_id
        speech_before_id = speech_id
    return 0


def get_person_id(speech, matched_names, verbose):
    """given a speech, retrieves a person id.

    Requires an array of matched_names, which contain a mapping of a name of a
    speaker to the name of a person in the database.

    Parameters:
        speech: dict containing speech fields (e.g. 'text', 'header', ...).
        matched_names: dict of mapping from speaker name and parliament to 
            matching person name in database (e.g. 
            {('midiwo', 11): 'washington jakoyo midiwo'}).

    Returns:
        person_id: ObjectId representing id of matched person. If not match is
            found, returns None.
    """
    # constructs key for matched_names dict.
    speaker = speech['speaker']
    # speaker_cleaned = speech['speaker_cleaned']
    # appointment = speech['appointment']
    # title = speech['title']
    try:
        parliament = speech['parliament']
    except TypeError:
        parliament = ''
    except ValueError:
        parliament = ''
    key = (speaker, parliament)
    key = tuple([el if el is not None else '' for el in key])
    try:
        # retrieves matching persons.
        name = matched_names[key]
        persons = db.persons.find({'name': name})
        n_persons = persons.count()
        # retrieves person_id.
        if n_persons == 0:
            person_id = None
            if verbose > 1:
                print('Person match not found for key "{0}"'.format(key))
        elif n_persons == 1:
            person = persons.next()
            person_id = person['_id']
        elif n_persons > 1:
            persons = list(persons)
            person_id = persons[0]['_id']
            if verbose:
                print('WARNING: Found two or more person documents. Returning first person id only. \nSpeaker: {0}\nPersons: {0}'.format(speaker, persons))
    except KeyError:
        person_id = None
        if verbose > 1:
            print('Person not found for key "{0}"'.format(key))
    return person_id

def get_is_chair(speech):
    """returns True if speech is made by speaker of the house or a
    chairperson. Returns False otherwise.
    """
    substrings = ['speaker', 'chair', 'deputy', 'temporary', 'spekaer', 'spika']
    chair_regex = re.compile('|'.join(substrings), re.IGNORECASE)
    # checks that no actual person names accidentally match the regex.
    # assert db.persons.find({'name': {'$regex': chair_regex}}).count() == 0, 'chair regex matches a true person name.'
    fields = ['speaker', 'speaker_cleaned', 'appointment', 'title']
    for field in fields:
        if speech[field] is not None and chair_regex.search(speech[field]):
            return True
    return False

if __name__ == '__main__':
    main()