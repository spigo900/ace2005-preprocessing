import argparse
import csv
from pathlib import Path
import random
from typing import List, Tuple

from richere_parser import DOCUMENTS_DIRECTORY


# Seed to use for permuting the list of documents.
RANDOM_SEED = 42

# Fractions of the dataset to use for train, dev, and test datasets.
TRAIN_FRACTION = 0.8
TEST_FRACTION = 0.1

# Header for the output CSV file
DATA_LIST_HEADER = ('type', 'path')


def split_documents(documents: List[str]) -> Tuple[List[str], List[str], List[str]]:
    num_documents = len(documents)

    num_train = int(TRAIN_FRACTION * num_documents)
    num_test = int(TEST_FRACTION * num_documents)
    num_dev = num_documents - num_train - num_test

    dev_start = num_train
    test_start = num_train + num_dev

    return (
        documents[:num_train],
        documents[dev_start:dev_start + num_dev],
        documents[test_start:test_start + num_test],
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="Path of ACE2005 English data", type=Path)
    parser.add_argument('--output', help="Output path", default=Path('./data_list.csv'), type=Path)
    return parser.parse_args()


def sanity_checks(_args):
    assert TRAIN_FRACTION > 0
    assert TEST_FRACTION > 0
    assert TRAIN_FRACTION + TEST_FRACTION < 1


def main(args):
    sanity_checks(args)
    richere_path = args.data
    output_path = args.output

    random.seed(RANDOM_SEED)

    all_documents = [path.stem for path in (richere_path / DOCUMENTS_DIRECTORY).glob('*.xml')]
    permuted_documents = all_documents.copy()
    random.shuffle(permuted_documents)
    train_documents, dev_documents, test_documents = split_documents(permuted_documents)
    documents_with_data_types = (
        ('train', train_documents),
        ('dev', dev_documents),
        ('test', test_documents),
    )

    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(DATA_LIST_HEADER)
        for data_type, document_list in documents_with_data_types:
            for document in document_list:
                writer.writerow((data_type, document))


if __name__ == '__main__':
    main(parse_arguments())
