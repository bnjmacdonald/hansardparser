"""tests for the `Rule` module.

"""

import unittest
from TxtParser.LineSpeakerSpanLabeler.Rule import Rule


class ParseSpeakerNameTests(unittest.TestCase):
    
    def setUp(self):
        self.labeler = Rule(verbosity=2)

    def test_parse_speaker_name(self):
        """tests the `_parse_speaker_name` method.
        """
        strings = [
            # "string", ("expected title", "expected name", "expected appointment")
            ("Mr. Bett ", ("Mr", "Bett", None)),
            ("Bett", (None, "Bett", None)),
            (" Bett", (None, "Bett", None)),
            ("Mr. John Mbadi", ("Mr", "John Mbadi", None)),
            ("Hon. (Mrs.) Shebesh", ("Hon Mrs", "Shebesh", None)),
            ("Hon. J.D. Lotodo", ("Hon", "J.D. Lotodo", None)),
            ("The Minister for Agriculture (Hon. Bett) ", ("Hon", "Bett", "The Minister for Agriculture")),
            ("Hon. Bett (The Minister for Agriculture)", ("Hon", "Bett", "The Minister for Agriculture")),
            ("The Temporary Deputy Speaker", (None, None, "The Temporary Deputy Speaker")),
            ("The Temporary Deputy Speaker (Hon. (Ms.) Mbalu)", ("Hon Ms", "Mbalu", "The Temporary Deputy Speaker")),
            ("The Temporary Deputy Speaker (Ms. Mbalu)", ("Ms", "Mbalu", "The Temporary Deputy Speaker")),
            ("Mrs. Odhiambo-Mabona", ("Mrs", "Odhiambo-Mabona", None)),
            ("Hon. (Mrs.) Odhiambo-Mabona", ("Hon Mrs", "Odhiambo-Mabona", None)),
            ("Hon (Mrs.) Odhiambo-Mabona", ("Hon Mrs", "Odhiambo-Mabona", None)),
            ("Hon. (Mrs) Odhiambo-Mabona", ("Hon Mrs", "Odhiambo-Mabona", None)),
            ("Hon (Mrs) Odhiambo-Mabona", ("Hon Mrs", "Odhiambo-Mabona", None)),
            ("Maj. Gen. Nkaissery", ("Maj. Gen", "Nkaissery", None)),
            ("Eng. Gumbo", ("Eng", "Gumbo", None)),
            ("Hon. (Eng.) Gumbo", ("Hon Eng", "Gumbo", None)),
            ("Hon (Eng.) Gumbo", ("Hon Eng", "Gumbo", None)),
        ]
        for s, expected in strings:
            output = self.labeler._parse_speaker_name(s)
            # print(s, output)
            self.assertEqual(output[0], expected[0])
            self.assertEqual(output[1], expected[1])
            self.assertEqual(output[2], expected[2])


if __name__ == '__main__':
    unittest.main()

