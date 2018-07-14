
import os
import unittest
import subprocess
import shutil

from hansardparser import settings

class MkLineCorpusTests(unittest.TestCase):
    """tests mk_lines_corpus.
    """

    def setUp(self):
        self.corpus_path = os.path.join(settings.DATA_ROOT, 'tests', 'generated', 'plenaryparser', 'text2vec', 'corpora')
        self.builder_path = os.path.join(settings.DATA_ROOT, 'tests', 'generated', 'plenaryparser', 'text2vec', 'builders')
        if os.path.exists(self.corpus_path):
            shutil.rmtree(self.corpus_path)
        if os.path.exists(self.builder_path):
            shutil.rmtree(self.builder_path)


    def tearDown(self):
        if os.path.exists(self.corpus_path):
            shutil.rmtree(self.corpus_path)
        if os.path.exists(self.builder_path):
            shutil.rmtree(self.builder_path)


    def test_hand_labels(self):
        """tests that mk_lines_corpus constructs a corpus from hand-labeled data.
        """
        res = subprocess.call([
            'python', '-m',
            'hansardparser.plenaryparser.build_training_set.mk_lines_corpus',
            '-v', '2', '--fmt', 'seq', '--rm_flatworld_tags',
            '--input', os.path.join(settings.DATA_ROOT, 'tests', 'generated', 'plenaryparser', 'hansard_txt_hand_labels_w_text.csv'),
            '--corpus', self.corpus_path,
            '--builder', self.builder_path,
        ])
        # test 1: return code should be 0 (no error encountered)
        self.assertEqual(res, 0, 'Encountered an error.')
        # test 2: there should now be a path to corpora and builders
        self.assertTrue(os.path.exists(self.corpus_path))
        self.assertTrue(os.path.exists(self.builder_path))
        # test 3: there should be one directory within the corpus path and builder path.
        self.assertEqual(len(os.listdir(self.corpus_path)), 1)
        self.assertEqual(len(os.listdir(self.builder_path)), 1)
        # test 4: within the directory in corpus path, there should be 'train', 'dev', and 'test' folders.
        # there should also be a `label_codes.json` file.
        corpus_files = os.listdir(os.path.join(self.corpus_path, os.listdir(self.corpus_path)[0]))
        self.assertIn('train', corpus_files)
        self.assertIn('dev', corpus_files)
        self.assertIn('test', corpus_files)
        self.assertIn('label_codes.json', corpus_files)
        # test 5: within the directory in builder path, there should be a dictionary.json file.
        builder_files = os.listdir(os.path.join(self.builder_path, os.listdir(self.builder_path)[0]))
        self.assertIn('dictionary.json', builder_files)
    

if __name__ == '__main__':
    unittest.main()
