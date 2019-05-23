"""splits Hansard transcript lines into train, dev, and test files.

Example::

    python -m build_training_set.split_lines -v 1 \
        --filepath ../../data/tests/manual/speaker_name_hand_labels_w_text20.csv \
        --outpath ../../data/tests/generated/plenaryparser/speaker_name_hand_labels_w_text20_splits \
        --by_sitting
"""

import os
import argparse
from typing import List
import numpy as np
import pandas as pd

from build_training_set import split

ACCEPTABLE_FTYPES = set(['txt', 'csv'])
# SEED = 30197
# np.random.seed(SEED)

def parse_args(args: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-f', '--filepath', type=str, help='Input file containing lines.')
    parser.add_argument('-o', '--outpath', type=str, help='Directory in which to save split data.')
    parser.add_argument('--by_sitting', action='store_true', help='Split lines by sitting/transcript.')
    parser.add_argument('--train_size', type=float, default=0.6, help='Proportion of data set to devote to training set.')
    parser.add_argument('--dev_size', type=float, default=0.2, help='Proportion of data set to devote to dev set.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data set to devote to test set.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    args = parser.parse_args(args)
    return args


def main(filepath: str, outpath: str, by_sitting: bool = False,
         train_size: float = 0.6, dev_size: float = 0.2, test_size: float = 0.2,
         seed: int = None, verbosity: int = 0):
    if seed:
        np.random.seed(seed)
    filetype = filepath.split('.')[-1].strip().lower()
    assert filetype in ACCEPTABLE_FTYPES, \
        f'File type must be one of {ACCEPTABLE_FTYPES}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # splits lines in txt file.
    sizes = {'train_size': train_size, 'dev_size': dev_size, 'test_size': test_size}
    if filetype == 'txt':
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if by_sitting:
            raise NotImplementedError()
        else:
            _ids = list(range(0, len(lines)))
        splits = split.train_dev_test_split(_ids, sizes=sizes)
        for split_name, split_ids in splits.items():
            if by_sitting:
                raise NotImplementedError()
            else:
                these_lines = np.array([lines[_id] for _id in split_ids])
            np.savetxt(os.path.join(outpath, f'{split_name}.txt'), these_lines, fmt='%s')
    # splits lines in csv.
    elif filetype == 'csv':
        lines = pd.read_csv(filepath, encoding='latin')
        if by_sitting:
            _ids = lines.file.unique()
        else:
            _ids = lines.index.values
        splits = split.train_dev_test_split(_ids, sizes=sizes)
        for split_name, split_ids in splits.items():
            if by_sitting:
                these_lines = lines[lines.file.isin(split_ids)]
            else:
                these_lines = lines.loc[split_ids]
            these_lines.to_csv(os.path.join(outpath, f'{split_name}.csv'), index=False)
    else:
        raise RuntimeError(f'File type must be one of {ACCEPTABLE_FTYPES}.')


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
