
import os
import unittest
import subprocess
import shutil

from hansardparser import settings

class SplitLinesTests(unittest.TestCase):
    """tests the `split_lines` module.
    """

    def setUp(self):
        self.filepath = "data/tests/manual/speaker_name_hand_labels_w_text20.csv"
        self.outpath = "data/tests/generated/plenaryparser/speaker_name_hand_labels_w_text20_splits"


    def tearDown(self):
        if os.path.exists(self.outpath):
            shutil.rmtree(self.outpath)


    def test_split_hand_labels(self):
        """tests that split_lines constructs data splits from hand-labeled data.
        """
        res = subprocess.call([
            'python', '-m',
            'hansardparser.plenaryparser.build_training_set.split_lines',
            '-v', '2', '--filepath', self.filepath,
            '--outpath', self.outpath,
        ])
        # test 1: return code should be 0 (no error encountered)
        self.assertEqual(res, 0, 'Encountered an error.')
        # test 2: there should now be a path to the outpath
        self.assertTrue(os.path.exists(self.outpath))
        # test 3: within the outpath directory, there should be 'train', 'dev', and 'test' files.
        outpath_files = os.listdir(self.outpath)
        self.assertIn('train.csv', outpath_files)
        self.assertIn('dev.csv', outpath_files)
        self.assertIn('test.csv', outpath_files)


if __name__ == '__main__':
    unittest.main()
