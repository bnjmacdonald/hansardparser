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
            ("Hon. A.B Duale", ("Hon", "A.B Duale", None)),
            ("Hon. A. B. Duale", ("Hon", "A. B. Duale", None)),
            ("Hon. Duale, AB", ("Hon", "Duale, AB", None)),
            ("Hon. Duale, A.B.", ("Hon", "Duale, A.B.", None)),
            ("Hon. Aden Duale", ("Hon", "Aden Duale", None)),
            ("Mr.Bett ", ("Mr", "Bett", None)),
            ("Mrs.shebesh ", ("Mrs", "shebesh", None)),
            ("MRS. SHEBESH", ("MRS", "SHEBESH", None)),
            ("The Minister for Agriculture (Hon. Bett) ", ("Hon", "Bett", "The Minister for Agriculture")),
            ("Hon. Bett (The Minister for Agriculture)", ("Hon", "Bett", "The Minister for Agriculture")),
            ("Hon. F.K. Bett (The Minister for Agriculture)", ("Hon", "F.K. Bett", "The Minister for Agriculture")),
            ("Hon. F. K Bett (The Minister for Agriculture)", ("Hon", "F. K Bett", "The Minister for Agriculture")),
            ("Hon. M. Karua (The Minister for Information, communications, and other stuff)", ("Hon", "M. Karua", "The Minister for Information, communications, and other stuff")),
            ("Hon. (Mrs.) Martha Karua (The Minister for Information, communications, and other stuff)", ("Hon Mrs", "Martha Karua", "The Minister for Information, communications, and other stuff")),
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
            ("Hon. Ichungwah (Kikuyu, JP)", ("Hon", "Ichungwah", "Kikuyu, JP")),
            ("Hon. Kimani Ichungwah (Kikuyu, JP)", ("Hon", "Kimani Ichungwah", "Kikuyu, JP")),
            ("Hon. K. Ichungwah (Kikuyu, JP)", ("Hon", "K. Ichungwah", "Kikuyu, JP")),
            ("Hon. Ichungwah (Kikuyu)", ("Hon", "Ichungwah", "Kikuyu")),
            ("Hon. K. Ichungwah (Kikuyu)", ("Hon", "K. Ichungwah", "Kikuyu")),
            ("Hon. R.K. Otticholo (Sigowet/Soin, WDM-K)", ("Hon", "R.K. Otticholo", "Sigowet/Soin, WDM-K")),
            ("Hon. (Ms) R.K. Otticholo (Sigowet/Soin, WDM-K)", ("Hon Ms", "R.K. Otticholo", "Sigowet/Soin, WDM-K")),
            ("(Hon. Cheptumo)", ("Hon", "Cheptumo", None)),
            ("Hon. Cheptumo)", ("Hon", "Cheptumo", None)),
            ("(Hon. Cheptumo", ("Hon", "Cheptumo", None)),
            ("(Hon. W.K. Cheptumo)", ("Hon", "W.K. Cheptumo", None)),
            ("(Minister for Environment and Natural Resources)", (None, None, "Minister for Environment and Natural Resources")),
            ("(Minister for Environment and Natural Resources", (None, None, "Minister for Environment and Natural Resources")),
            ("Minister for Environment and Natural Resources)", (None, None, "Minister for Environment and Natural Resources"))
        ]
        for s, expected in strings:
            output = self.labeler._parse_speaker_name(s)
            self.assertEqual(output[0], expected[0], msg=f'{output[0]} != {expected[0]}. String: {s}')
            self.assertEqual(output[1], expected[1], msg=f'{output[1]} != {expected[1]}. String: {s}')
            self.assertEqual(output[2], expected[2], msg=f'{output[2]} != {expected[2]}. String: {s}')


if __name__ == '__main__':
    unittest.main()

