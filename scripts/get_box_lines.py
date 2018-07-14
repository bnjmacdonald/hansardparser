"""retrieves the text for each line from a dataframe of hand-labeled lines.

Given a dataframe of hand-labeled lines from text files that are stored on Box.com,
retrieves the actual text of the line.
"""

import os
import re
import argparse
import pandas as pd

from vignettes.TxtParser.get_box_files import get_file_ids, get_text_file
from hansardparser import settings

# path to hand labels.
HAND_LABELS_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'hansard_txt_hand_labels.csv')
# path to where hand labels with text should be saved.
NEW_HAND_LABELS_PATH = os.path.join(settings.DATA_ROOT, 'generated', 'hansard_txt_hand_labels_w_text.csv')

def main(verbosity: int = 0) -> None:
    # reads in the hand labels.
    hand_labels = pd.read_csv(HAND_LABELS_PATH)
    # retrieves the id of each file.
    if verbosity > 0: print('retrieving file _ids...')
    file_ids = get_file_ids(hand_labels.year.astype(str).unique())
    # merges file ids onto hand labels.
    file_ids.file = file_ids.file.str.replace(r'.txt$', '')
    hand_labels.rename(columns={'year': 'folder'}, inplace=True)
    hand_labels.folder = hand_labels.folder.astype(str)
    hand_labels = pd.merge(hand_labels, file_ids, on=['folder', 'file'], how='left')
    assert hand_labels._id.isnull().sum() == 0, 'There should be no missing file _ids.'
    # for each file, retrieves the text of each hand-labeled line.
    if verbosity > 0: print('retrieving text of each hand-labeled line...')
    all_lines = []
    for nm, gp in hand_labels.groupby('_id'):
        if verbosity > 1: print(f'retrieving text for file {nm}...')
        text = get_text_file(nm)
        text = re.split(r'\r\n|\r|\n', text)
        try:
            lines = pd.DataFrame([(nm, l, text[l-1]) for l in gp.line.values], columns=['_id', 'line', 'text'])
            all_lines.append(lines)
        except Exception as e:
            print(f'Failed to grab lines from file {nm}.\nError: {e}')
    # merges line text onto hand labels dataframe.
    all_lines = pd.concat(all_lines, axis=0)
    hand_labels = pd.merge(hand_labels, all_lines, on=['_id', 'line'], how='left')
    # saves hand labels dataframe to disk (now with text of each line).
    hand_labels.to_csv(NEW_HAND_LABELS_PATH, index=False)
    if verbosity > 0: print('Success!')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    args = parser.parse_args()
    main(**args.__dict__)
