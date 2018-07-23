"""splits data into training, dev, and test sets.

Example usage::
    
    >>> import numpy as np
    >>> from hansardparser.plenaryparser.classify.split import train_dev_test_split
    >>> data = np.unique(np.random.randint(0, high=100000, size=5000))
    >>> seed = 759388
    >>> sizes = {'train_size': 0.8, 'test_size': 0.2}
    >>> out_path = './'
    >>> splits = train_dev_test_split(data, sizes, random_state=seed)
    >>> save_splits(splits, out_path, fmt='%i')

"""

import os
import numpy as np
from sklearn.model_selection import train_test_split


def train_dev_test_split(data: list, sizes: dict = None, **kwargs) -> dict:
    """splits data into training, development (optional), and test sets.
    
    Wrapper to sklearn.model_selection.train_test_split.

    Arguments:

        data: list-like iterable of examples to be passed as the *arrays
            argument in sklearn.model_selection.train_test_split.

        sizes: dict indicating what fraction of the examples should be in the
            training, development, and test sets. dict keys must be
            "train_size", "dev_size" (optional), and "test_size". Values must
            sum to 1.0. If "dev_size" key is not given, data is only split
            into a training set and test seet. 
            Default::
            
                {'train_size': 0.6, 'dev_size': 0.2, 'test_size': 0.2}

        **kwargs: keyword arguments to be passed to
            sklearn.model_selection.train_test_split.

    Returns:

        dict: {'train': [train example], 'dev': [dev example], 'test': [test 
            example]}. dict containing arrays of training examples, development
            examples, and test examples.
    """
    if sizes is None:
        sizes = {'train_size': 0.6, 'dev_size': 0.2, 'test_size': 0.2}
    assert 'train_size' in sizes and 'test_size' in sizes, "sizes must contain 'train_size' and 'test_size' keys."
    assert all(
        [k in ['train_size', 'dev_size', 'test_size']
         for k in sizes]), "only valid keys for sizes arg are train_size, dev_size, and test_size."
    assert sum(sizes.values()) == 1.0, "sizes values must sum to 1.0."
    # assert (len(sizes) == 2 and )
    if 'dev_size' in sizes:
        assert len(sizes) == 3
        train_ids, other_ids = train_test_split(
            data,
            train_size=sizes['train_size'],
            test_size=sizes['dev_size'] + sizes['test_size'],
            **kwargs)
        dev_ids, test_ids = train_test_split(
            other_ids,
            train_size=sizes['dev_size'] / (sizes['dev_size'] + sizes['test_size']),
            test_size=sizes['test_size'] / (sizes['dev_size'] + sizes['test_size']),
            **kwargs)
        splits = {'train': train_ids, 'dev': dev_ids, 'test': test_ids}
    else:
        train_ids, test_ids = train_test_split(
            data, train_size=sizes['train_size'], test_size=sizes['test_size'], **kwargs)
        splits = {'train': train_ids, 'test': test_ids}
    return splits


def save_splits(splits: dict, path: str, **kwargs) -> None:
    """saves data splits to file.

    Arguments:
        
        splits (dict): dict containing the data splits (e.g. return from
            split(). Keys are name of split (e.g. 'train', 'dev', 'test').
            Values should be numpy arrays.
        path (str): directory to save data splits.
        **kwargs: keyword arguments to pass to np.savetxt
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    for split, data in splits.items():
        this_path = os.path.join(path, split + '.txt')
        np.savetxt(this_path, data, **kwargs)
    return None
