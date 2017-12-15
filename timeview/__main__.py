#!/usr/bin/env python3

"""
TimeView CLI and GUI Application
"""

import logging
from pathlib import Path
import argparse

from .api import TimeView


def parse(args):
    # ENHANCE: think about how to specify CLI loading configurations in a .cfg file
    if args.configuration == 'default':  # one object per panel
        app = TimeView()
        print(args.path)
        for i, path in enumerate(args.path):
            print(f'Loading {path}')
            app.add_view_from_file(Path(path), panel_index=i)
        app.start()
    elif args.configuration == 'labeling':
        raise NotImplementedError
    else:
        raise Exception('unhandled configuration')


def main():
    configurations = ['default']  # , 'labeling']
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="© Copyright 2009-2017, TimeView Developers", prog='TimeView')
    parser.add_argument('-c', '--configuration', type=str, default='default', choices=configurations)
    parser.add_argument('path', type=str, nargs='*', help='files to load')
    parser.set_defaults(func=parse)
    args = parser.parse_args()
    args.func(args)


main()
