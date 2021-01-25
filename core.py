import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory

import hpargparse
from hpman.m import _

from pipeline.train_hvd import HvdTrainer


def prepare_parser():
    parser = argparse.ArgumentParser(
        description='The core script of experiment management.')
    subparsers = parser.add_subparsers(dest='command')

    # add_preprocess_parser(subparsers)
    add_train_parser(subparsers)
    add_test_parser(subparsers)
    # add_parallel_train_parser(subparsers)
    # add_table_parser(subparsers)
    # add_server_parser(subparsers)
    # add_client_parser(subparsers)

    return parser



def add_train_parser(subparsers):
    parser = subparsers.add_parser('train', description='Training.')
    _.parse_file('.')
    hpargparse.bind(parser, _)
    # args = parser.parse_args()


def train(args):
    trainer = HvdTrainer()
    trainer.run()


def add_test_parser(subparsers):
    """
    eg.: python core.py preprocess <split_0> <split_1> --width 160 --height 120
    """
    parser = subparsers.add_parser('test', description='Testing.')


def test(args):
    pass


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)
    else:
        pass