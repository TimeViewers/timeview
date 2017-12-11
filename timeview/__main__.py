#!/usr/bin/env python3

"""
TimeView CLI and GUI Application
"""

import logging
from pathlib import Path

from .api import TimeView


def parse(args):
    # ENHANCE: think about how to specify CLI loading configurations in a .cfg file
    if args.configuration == 'default':  # one object per panel
        app = TimeView()
        for path in args.path:
            files = Path().glob(path)
            for i, file in enumerate(files):
                logging.info(f'Loading {file}')
                if i:
                    app.add_panel()
                app.add_view_from_file(file)
        app.start()
    elif args.configuration == 'labeling':
        raise NotImplementedError
    else:
        raise Exception('unhandled configuration')


def main():
    import argparse
    configurations = ['default']  # , 'labeling']
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="© Copyright 2009-2017, TimeView Developers", prog='TimeView')
    parser.add_argument('-c', '--configuration', type=str, default='default', choices=configurations)
    parser.add_argument('path', type=str, nargs='*', help='files to load')
    parser.set_defaults(func=parse)
    args = parser.parse_args()
    args.func(args)


main()
