
import os
import re
import warnings
import shutil
from typing import List, Tuple
import numpy as np
import pandas as pd
from unidecode import unidecode

from text2vec.corpora.corpora import Corpus, CorpusBuilder


def insert_xml_tag_whitespace(s: str) -> str:
    """inserts whitespace between an xml tag and text.

    Examples::

        >>> s = '<i>hello</i>'
        >>> insert_xml_tag_whitespace(s)
        ' <i> hello </i> '
        >>> s = 'my name is <b>bob</b>'
        >>> insert_xml_tag_whitespace(s)
        'my name is  <b> bob </b> '
    """
    s2 = re.sub(r'<', ' <', s)
    s2 = re.sub(r'>', '> ', s2)
    return s2


def str2ascii_safe(s):
    """converts string to ascii.
    """
    if pd.isnull(s):
        return None
    return unidecode(s)


def build_temp_corpus(documents: List[dict], builder_path: str) -> Tuple[np.array, np.array]:
    """constructs a temporary corpus that can be fed into a trained tensorflow
    model for training or prediction.

    Generates a temporary corpus on disk in the `./corpus_temp0` directory and
    then deletes this directory before the method returns.

    Arguments:

        documents: List[dict]. List of documents to be converted into a corpus.
            If documents is a list of str, then it is converted into a list of
            dict to meet the requirement of CorpusBuilder.

        builder_path: str. Path to corpus builder.

    Returns:

        inputs, seqlens: Tuple[np.array, np.array].

            inputs: np.array with shape (n_documents, n_tokens). Corpus. Each
                row represents a document.

            seqlens: np.array with shape (n_documents,). Sequence length of each
                document.
    """
    # KLUDGE: for required CorpusBuilder input format.
    if isinstance(documents[0], str):
        documents = [{'text': doc, '_id': i} for i, doc in enumerate(documents)]
    corpus_path = './corpus_temp0'
    try:
        i = 1
        while os.path.exists(corpus_path):
            warnings.warn(f'{corpus_path} already exists. Checking if '
                          f'{corpus_path[:-1] + str(i)} can be created instead...')
            corpus_path = corpus_path[:-1] + str(i)
            i += 1
        os.makedirs(corpus_path)
        builder = CorpusBuilder(builder_path)
        # KLUDGE: loads all data into memory rather than reading from disk.
        corpus = Corpus(corpus_path, builder=builder)
        corpus.build(documents, update_dict=False, overwrite=True)
        inputs = np.array([doc for doc in corpus])
        seqlens = np.loadtxt(os.path.join(corpus_path, corpus.seqlens_fname), dtype=np.int64)
        shutil.rmtree(corpus_path)
    except Exception as err:
        shutil.rmtree(corpus_path)
        raise err
    return inputs, seqlens
