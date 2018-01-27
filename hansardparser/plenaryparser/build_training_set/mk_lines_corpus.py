"""builds a corpus for training an XML lines classifier and saves to disk.

Useful for constructing an input matrix `X` from text documents for a supervised
or unsupervised learning task. Uses methods defined in the `text2vec` library
to build the corpus.

Saves corpus to the `data/generated/text2vec` folder.

Usage:

    Convert xml lines to padded sequence format (e.g. for deep learning models)::

        python -m hansardparser.plenaryparser.build_training_set.mk_lines_corpus \
            -v 2 \
            --fmt seq \
            --input /Users/bnjmacdonald/Documents/current/projects/hansardlytics/data/raw/hansards/2013-

    Convert xml lines to bag-of-words format (e.g. for sklearn models)::

        python -m hansardparser.plenaryparser.build_training_set.mk_lines_corpus -v 2 --fmt bow --input tests/test_input
"""

import os
import re
import json
import argparse
import datetime
from typing import List, Tuple
import numpy as np
# from nltk.stem.porter import PorterStemmer
from text2vec.corpora.corpora import Corpus, CorpusBuilder
from text2vec.processing.preprocess import preprocess_one  # rm_stop_words_punct, rm_digits
from text2vec.corpora.dictionaries import BasicDictionary

from hansardparser.plenaryparser.build_training_set.extract_line_labels import LineLabelExtractor
from hansardparser.plenaryparser.build_training_set.utils import insert_xml_tag_whitespace
from hansardparser.plenaryparser.classify import split
from hansardparser import settings

SPLIT_SIZES = {'train_size': 0.6, 'dev_size': 0.2, 'test_size': 0.2}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-i', '--input', type=str, help="path to input directory containing Hansard PDFs.")
    parser.add_argument('-f', '--fmt', type=str, help="model type", choices=['bow', 'seq'], required=True)
    parser.add_argument('--max_seqlen', type=int, help="max sequence length")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    # creates directories to save corpus and corpus builder.
    datestr = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H%M%S')
    corpus_path = os.path.join(settings.DATA_ROOT, 'generated', 'text2vec', 'corpora', datestr)
    builder_path = os.path.join(settings.DATA_ROOT, 'generated', 'text2vec', 'builders', datestr)
    if not os.path.exists(builder_path):
        os.makedirs(builder_path)
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    lines, labels = _extract_all_lines(args.input)
    labels, label_codes = _assign_label_codes(labels)
    # saves label codes to disk.
    with open(os.path.join(corpus_path, 'label_codes.json'), 'w') as f:
        json.dump(label_codes, f)
    _split_and_build_corpus(lines, labels, corpus_path, builder_path, args)
    return None


def _extract_all_lines(path: str, verbosity: int = 0) -> Tuple[List[dict], list]:
    """extracts xml lines from all PDFs in a directory.

    wrapper to LineLabelExtractor.extract_labels.

    Arguments:

        path: str. Path to directory containing PDFs.

    Returns:

        all_lines, all_labels: Tuple[List[dict], list].

            First element of tuple is list of dicts, where each dict contains a
                "text" key containing an xml line and an "_id" key that uniquely
                identifies the line. This is the structure required by
                `corpus.build()`.

            Second element of tuple is list of line labels.
    """
    # constructs a list of dicts,
    if verbosity > 0:
        print('extracting xml lines from PDFs...')
    extractor = LineLabelExtractor(verbose=verbosity)
    all_lines = []
    all_labels = []
    for fname in os.listdir(path):
        if fname.endswith('.pdf'):
            soup = extractor.convert_pdf(os.path.join(path, fname), save_soup=False)
            labels, lines = extractor.extract_labels(soup)
            fname_temp = re.sub(r'\s+', '', fname.strip().lower())
            lines = [{'text': str(line), '_id': fname_temp + '_' + str(i)} for i, line in enumerate(lines)]
            all_lines.extend(lines)
            all_labels.extend(labels)
    assert len(all_lines) == len(all_labels), 'lines and labels must have same length.'
    return all_lines, all_labels


def _assign_label_codes(labels: list) -> Tuple[List[int], dict]:
    """converts str labels to int codes.

    Returns:

        all_labels_coded, label_codes: Tuple[List[int], dict]. First element of
            tuple is list of label codes. Second element is dict containing
            mapping of each unique label to its code.
    """
    label_codes = {}
    labels_coded = []
    for label in labels:
        try:
            code = label_codes[label]
        except KeyError:
            code = len(label_codes)
            label_codes[label] = code
        labels_coded.append(code)
    return labels_coded, label_codes


def _split_and_build_corpus(lines: List[dict], labels: list, corpus_path: str, builder_path: str, args: argparse.Namespace) -> None:
    """splits lines into train, dev, and test sets, then builds a corpus for each
    set and saves corpus to disk.

    Arguments:

        lines: List[dict]. List of xml lines, where each element is a dict with a "text"
            key and an "_id" key.

        labels: list. List of line labels.

        corpus_path: str. Directory to save corpus.

        builder_path: str. Directory to save corpus builder.

        args: argparse.Namespace.

    Returns:

        None.
    """
    assert len(lines) == len(labels), 'lines and labels must have same length.'
    # corpus builder options
    options = {}
    if args.max_seqlen:
        options['max_seqlen'] = args.max_seqlen
    # splits data into train, dev, and test sets. Then builds and saves corpus
    # for each set.
    _ids = [line['_id'] for line in lines]
    assert len(_ids) == len(set(_ids)), 'one or more line _ids are not unique.'
    splits = split.train_dev_test_split(_ids, sizes=SPLIT_SIZES)
    # split.save_splits(splits, os.path.join(corpus_path, 'splits'), fmt="%s")
    for split_name, split_ids in splits.items():
        if args.verbosity > 0:
            print('building corpus for {0} set.'.format(split_name))
        if not os.path.isdir(os.path.join(corpus_path, split_name)):
            os.mkdir(os.path.join(corpus_path, split_name))
        if not os.path.isdir(os.path.join(builder_path, split_name)):
            os.mkdir(os.path.join(builder_path, split_name))
        # saves _ids
        np.savetxt(os.path.join(corpus_path, split_name, 'labels.txt'), split_ids, fmt='%s')
        # subsets lines and labels to only those examples in this data split.
        split_ids_set = set(split_ids)
        split_lines = []
        split_labels = []
        for i, line in enumerate(lines):
            if line['_id'] in split_ids_set:
                split_lines.append(line)
                split_labels.append(labels[i])
        # saves labels.
        np.savetxt(os.path.join(corpus_path, split_name, 'labels.txt'), split_labels, fmt='%s')
        # initializes corpus builder.
        builder = CorpusBuilder(
            path=os.path.join(builder_path, split_name),
            fmt=args.fmt,
            dictionary=BasicDictionary(),
            text_transformer=__text_transformer,
            options=options,
            verbosity=args.verbosity
        )
        # builds corpus for this data split
        corpus = Corpus(os.path.join(corpus_path, split_name), builder=builder, verbosity=args.verbosity)
        corpus.build(split_lines, update_dict=True)
        # saves builder config.
        # NOTE: builder must be saved because dictionary and text_transformer will
        # be needed later.
        builder.save()
    if args.verbosity:
        print('\nCorpus saved to {0}\nDictionary saved to {1}\nSuccess!'.format(corpus_path, builder_path))
    return 0


def __text_transformer(text: str) -> List[str]:
    """custom text transformer. Receives a string as the only argument and
    returns a list of tokens."""
    pipeline = [
        {"returns": "str", "function": insert_xml_tag_whitespace},
    ]
    return preprocess_one(text, pipeline=pipeline)


if __name__ == '__main__':
    main()
