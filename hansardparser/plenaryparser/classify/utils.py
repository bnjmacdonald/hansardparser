
import re
import argparse
from typing import List

def parse_unknown_args(args: List[str]) -> argparse.Namespace:
    """parses unknown args returned from second element of parser.parse_known_args().

    Arguments:

        args: List[str]. Unparsed arguments.

    Returns:

        parsed_args: argparse.Namespace. Namespace of parsed arguments.

    Example::

        >>> args = ['-f', '1', '--foo', 'abcd dfd edd', '--bar', 'adv', '--bar2', '--foo2', '1.5']
        >>> parse_unknown_args(args)
        Namespace(bar='adv', bar2=True, f=1, foo=['abcd', 'dfd', 'edd'], foo2=1.5)
    """
    parsed_args = {}
    arg = None
    while len(args) > 0:
        el = args.pop(0)
        if is_arg(el):
            arg = re.sub(r'^-{1,2}', '', el)
            if len(args) == 0 or (len(args) > 0 and is_arg(args[0])):
                parsed_args[arg] = True
        else:
            while len(args) > 0 and not is_arg(args[0]):
                el += ' ' + args.pop(0)
            assert arg is not None, '{0} value encountered before arg name.'.format(el)
            el = el.strip()
            vals = el.split(' ')
            parsed_val = []
            for val in vals:
                try:
                    if '.' in val:
                        val = float(val)
                    elif el in ['True', 'False']:
                        val = val == 'True'
                    else:
                        val = int(val)
                except ValueError:
                    pass
                parsed_val.append(val)
            assert arg not in parsed_args, '{0} arg was passed twice. Each arg can only be passed once.'
            parsed_args[arg] = parsed_val if len(parsed_val) > 1 else parsed_val[0]
    return argparse.Namespace(**parsed_args)


def is_arg(s):
    """given a string (s), returns True if s is a command-line argument (i.e.
    prefixed with '--' or '-'). False otherwise."""
    return bool(re.search(r'^-{1,2}', s))
