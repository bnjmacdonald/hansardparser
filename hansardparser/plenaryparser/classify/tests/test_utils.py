
import unittest
from hansardparser.plenaryparser.classify import utils

class ParseUnknownArgsTests(unittest.TestCase):
    """Tests utils.parse_unknown_args method."""

    def test_parse_prefix(self):
        """Tests that both '--' and '-' prefixes get detected as arg names."""
        args = ['--foo', 'a', '-f', 'a']
        result = utils.parse_unknown_args(args)
        self.assertEqual(result.foo, 'a')
        self.assertEqual(result.f, 'a')

    def test_parse_type(self):
        """Tests that digits get parsed as the appropriate type."""
        args = [
            (['--foo', '1'],       int),
            (['--foo', '59392'],   int),
            (['--foo', 'abcd'],    str),
            (['--foo', '59.39'],   float),
            (['--foo', 'false'],   str),
            (['--foo', 'False'],   bool),
            (['--foo', 'True'],    bool),
            (['--foo'],            bool),
            (['--foo', '5.'],      float)
        ]
        for arg in args:
            result = utils.parse_unknown_args(arg[0])
            self.assertTrue(isinstance(result.foo, arg[1]))

    def test_list_arg(self):
        """tests that an argument followed by a string with whitespace or an
        array of non-args is parsed as a list."""
        args = [
            # INPUT                           EXPECTED RESULT
            (['--foo', 'one two three four'], {'foo': ['one', 'two', 'three', 'four']}),
            (['--foo', '1 2 three 4.'], {'foo': [1, 2, 'three', 4.0]}),
            (['--foo', '1', '2', 'three', '4.1'], {'foo': [1, 2, 'three', 4.1]}),
        ]
        for arg in args:
            result = utils.parse_unknown_args(arg[0])
            self.assertEqual(len(result._get_kwargs()), 1)
            self.assertTrue(result.foo == arg[1]['foo'])

    def test_mixed_args(self):
        """tests that a mixed list of arguments gets parsed appropriately.
        """
        args = [{
            'input': ['-f', '1', '--foo', 'some thing another', '--bar', 'string', '--bar2', '--foo2', '1.5', '--bar3', 'False'],
            'expected': {'f': 1, 'foo': ['some', 'thing', 'another'], 'bar': 'string', 'bar2': True, 'foo2': 1.5, 'bar3': False}
        }]
        for arg in args:
            result = utils.parse_unknown_args(arg['input'])
            for k, v in arg['expected'].items():
                self.assertEqual(result.__getattribute__(k), v)

    def test_unique_args(self):
        """tests that each argument can only appear once.
        """
        # test 1.
        try:
            args = ['--foo', '1', '--foo', 'a']
            utils.parse_unknown_args(args)
            self.fail('utils.parse_unknown_args should have raised AssertionError.')
        except AssertionError:
            pass
        
        # test 2.
        try:
            args = ['-f', '1', '-f', '1']
            utils.parse_unknown_args(args)
            self.fail('utils.parse_unknown_args should have raised AssertionError.')
        except AssertionError:
            pass

        # test 3.
        try:
            args = ['--foo', '1', '--food', 'a']
            result = utils.parse_unknown_args(args)
        except AssertionError:
            self.fail('utils.parse_unknown_args raised AssertionError unexpectedly.')

if __name__ == '__main__':
    unittest.main()
