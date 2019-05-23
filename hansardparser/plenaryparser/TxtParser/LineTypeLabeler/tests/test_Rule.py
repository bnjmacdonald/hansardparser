"""tests for `RuleLineLabeler`.

Todos:

    TODO: get rid of `LINES` and replace tests with more targeted/specific tests
        that test specific functionality of each RuleLineLabeler method.
"""

import unittest
from TxtParser.LineTypeLabeler.Rule import Rule


class IsSubsubheaderTests(unittest.TestCase):
    """tests the `_is_subsubheader` method.
    """
    def setUp(self):
        self.labeler = Rule(verbosity=2)

    def test_clauses(self):
        """tests that clauses are labeled as subsubheaders.
        """
        lines = [
            ('Clause 32', True),
            ('clause 32', True),
            ('cla use 1', True),
            (' cl ause 1', True),
            ('Recommittal of clause', False),
            ('new clause 9', True),
            ('ne w clause 9', True),
            ('clauses 2 and 3', True),
            ('claus es 2, 3, and 4', True)
        ]
        for s, expected in lines:
            res = self.labeler._is_subsubheader(s)
            self.assertEqual(res, expected, msg=f'{res} != {expected}. Line: {s}')

    def test_division(self):
        """tests that divisions are labeled as subsubheaders.
        """
        lines = [
            ('division', True),
            ('Division', True),
            ('AYES', True),
            ('ayes', True),
            ('noes', True),
            ('Noes', True),
            ('Division of resources', False),
            ('Ayes for recommital', False),
            ('Ends with division', False)
        ]
        for s, expected in lines:
            res = self.labeler._is_subsubheader(s)
            self.assertEqual(res, expected, msg=f'{res} != {expected}. Line: {s}')
    
    def test_vote(self):
        """tests that votes are labeled as subsubheaders.
        """
        lines = [
            ('vote 138', True),
            ('Vote 138', True),
            ('Vote 138 - ministry of health', True),
            ('vote r138', True),
            ('Vote d138', True),
            ('Vote x138', False),
            ('subvote 90', True),
            ('sub vote 90', True),
            ('sub-vote 90', True),
            ('head 220', True),
            ('head 220 - ministry of information', True),
            ('Head d220', True),
            ('head r220', True),
            ('head x220', False),
            ('head of ministry', False),
            ('head of 138 ministry', False)
        ]
        for s, expected in lines:
            res = self.labeler._is_subsubheader(s)
            self.assertEqual(res, expected, msg=f'{res} != {expected}. Line: {s}')

    def test_schedule(self):
        """tests that schedules are labeled as subsubheaders.
        """
        lines = [
            ('Schedule 1', True),
            ('schedule 1', True),
            ('schedule 2 ', True),
            ('first schedule', True),
            ('First schedule', True),
            ('Second schedule ', True),
            ('third Schedule', True),
            ('fourth schedule', True),
            ('fifth schedule', True),
            ('sixth schedule', True),
            ('Schedule for next sitting', False),
            ('Next sitting schedule', False),
        ]
        for s, expected in lines:
            res = self.labeler._is_subsubheader(s)
            self.assertEqual(res, expected, msg=f'{res} != {expected}. Line: {s}')


if __name__ == '__main__':
    unittest.main()
