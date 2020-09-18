import argparse
from pathlib import Path
import re

from richere_parser import ANNOTATIONS_DIRECTORY


SOURCE_TYPE_PATTERN = re.compile(r'source_type\s*=\s*"([^"]*)"')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", type=Path)
    return parser.parse_args()


def main(args):
    richere_path = args.data

    source_types = set()
    for annotation_path in (richere_path / ANNOTATIONS_DIRECTORY).glob('*'):
        print(f'In {annotation_path.stem}:')
        with annotation_path.open('r', encoding='utf-8') as f:
            text = ''.join(f.readlines())

        # In Rich ERE there's only one deft_ere tag with exactly one source_type attribute per
        # annotation file, so we don't need to use .finditer().
        match = SOURCE_TYPE_PATTERN.search(text)
        if match:
            source_type = match.group(1)
            source_types.add(source_type)
            print(f'  Source type = {source_type}')
        else:
            print(f'  WARNING: source type not found')
        print()

    print('OVERALL:')
    print(f'  Source types found = {source_types}')


if __name__ == '__main__':
    main(parse_arguments())
