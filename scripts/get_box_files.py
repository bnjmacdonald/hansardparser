"""Implements methods for retrieving Hansard txt files from Box.com.

This script has been written specifically for the `Outputs` folder on Box.com
that contains Hansard txt transcripts.

Todos:

    TODO: add option to also download PDFs.
"""

import os
import re
import json
import argparse
import requests
# from boxsdk import OAuth2, Client
from typing import List, Generator, Optional
import pandas as pd

from scripts.config import BOX_DEVELOPER_TOKEN
from hansardparser import settings


# FILES_META_PATH = os.path.join(settings.DATA_ROOT, 'manual', 'txt-files-meta.json')
# with open(FILES_META_PATH, 'r') as f:
#     FILES_META = json.load(f)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-f', '--folders', nargs='+', required=True, type=str,
        help='Names of top-level Box.com folders within the `Outputs` folder for '
             'which to download txt files. Example: 1978 1980 1985.')
    parser.add_argument('-o', '--outpath', type=str, required=True,
        help='Directory where you want the downloaded files to be saved.')
    args = parser.parse_args()
    return args


def main(folders: List[str], outpath: str, verbosity: int = 0) -> None:
    """retrieves text files from Box.com and saves them to disk.
    """
    text_files = get_text_files(folders, get_text=True, verbosity=verbosity)
    for text, meta in text_files:
        files_folder = os.path.join(outpath, 'transcripts', meta['folder']['name'])
        meta_folder = os.path.join(outpath, 'meta', meta['folder']['name'])
        if not os.path.exists(files_folder):
            os.makedirs(files_folder)
        if not os.path.exists(meta_folder):
            os.makedirs(meta_folder)
        with open(os.path.join(files_folder, meta['name']), 'w') as f:
            f.writelines(text)
        with open(os.path.join(meta_folder, meta['name'].replace('.txt', '_meta.json')), 'w') as f:
            json.dump(meta, f)
    if verbosity > 0:
        print('Success!')
    return None


def get_text_files(folders: List[str],
                   get_text: bool = True,
                   verbosity: int = 0) -> Generator[Optional[str], None, None]:
    """retrieves text files from box.com.

    Arguments:

        folders: List[str]. List of folder names to retrieve text files from.

        get_text: bool = True. If True, retrieves text content of the file.

    Yields:

        text: Tuple[str, dict]. Text of file, along with metadata for the file.
            If get_text is False, then returns Tuple[None, dict].
    """
    folder_id = "26341197112"
    result = requests.get(f'https://api.box.com/2.0/folders/{folder_id}',
        headers={'Authorization': f"Bearer {BOX_DEVELOPER_TOKEN}"})
    if result.status_code != 200:
        raise RuntimeError('Failed to retrieve top-level folder.')
    top_folder = json.loads(result.content)
    folders = [folder for folder in top_folder['item_collection']['entries'] if folder['name'] in folders]
    # folders = np.random.choice(top_folder['item_collection']['entries'], size=2, replace=False)
    for year_folder_meta in folders:
        if verbosity > 0:
            print(f'Retrieving files for folder "{year_folder_meta["name"]]}"...')
        # retrieves information on year folder
        year_folder_id = year_folder_meta['id']
        result = requests.get(f'https://api.box.com/2.0/folders/{year_folder_id}',
            headers={'Authorization': f"Bearer {BOX_DEVELOPER_TOKEN}"})
        year_folder = json.loads(result.content)
        # retrieves files in the `Output` folder.
        output_folders_meta = [f for f in year_folder['item_collection']['entries'] if f['name'] == 'Output']
        assert len(output_folders_meta) == 1
        output_folder_id = output_folders_meta[0]['id']
        result = requests.get(f'https://api.box.com/2.0/folders/{output_folder_id}',
            headers={'Authorization': f"Bearer {BOX_DEVELOPER_TOKEN}"})
        output_folder = json.loads(result.content)
        for file_meta in output_folder['item_collection']['entries']:
            if verbosity > 1:
                print(f'Retrieving file "{file_meta["name"]}"...')
            assert 'file_version' in file_meta, ('non-file object found in Output'
                f'folder for year {year_folder_meta["name"]}')
            file_meta['folder'] = year_folder_meta
            text = None
            if get_text:
                text = get_text_file(file_meta['id'])
                text = re.split(r'\r\n|\r|\n', text)
            yield text, file_meta


def get_text_file(_id: str) -> str:
    """retreives the contents of a text file from Box.com.
    
    Arguments:

        _id: str. id of text file.

    Returns:

        str: contents of text file.
    """
    res = requests.get(f'https://api.box.com/2.0/files/{_id}/content',
        headers={'Authorization': f"Bearer {BOX_DEVELOPER_TOKEN}"})
    text = res.content.decode(res.apparent_encoding)
    assert len(text) > 0
    return text


def get_file_ids(*args, **kwargs) -> pd.DataFrame:
    """retrieves the id for each text file.

    Wraps to `get_text_files`. Then iterates through each (text, meta) tuple and
    retrieves the file id for each file.

    Arguments:

        *args, **kwargs: arguments to pass to `get_text_files`.
    
    Returns:

        pd.DataFrame. Dataframe with `folder`, `file`, and `id` columns.

    Example::

        >>> get_file_ids(['1940'])
          folder                                        file           _id
        0   1940  XI 1940 26TH NOVEMBER TO 19TH DECEMBER.txt  169311427912
        1   1940       XI 26TH NOVEMBER TO 19TH DECEMBER.txt  169311082992
    """
    files = get_text_files(*args, get_text=False, **kwargs)
    file_ids = []
    for _, meta in files:
        file_ids.append([meta['folder']['name'], meta['name'], meta['id']])
    return pd.DataFrame(file_ids, columns=['folder', 'file', '_id'])


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
