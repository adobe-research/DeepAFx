#!/usr/bin/env python3
#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import sys
import argparse
import os


from deepafx.download import download_dataset_distortion
from deepafx.download import download_dataset_nonspeech
from deepafx.download import download_dataset_mastering


all_parser = argparse.ArgumentParser(description='Download all models and datasets.')

parser = argparse.ArgumentParser(description='Command line download datasets and models.')

s = parser.add_subparsers(title='subcommands', dest='subcommand')


s.add_parser('distortion',  
             parents=[download_dataset_distortion.init_parser()], 
             add_help=False,
             help='Download distortion dataset.')
s.add_parser('nonspeech',  
             parents=[download_dataset_nonspeech.init_parser()], 
             add_help=False,
             help='Download nonspeech dataset.')
s.add_parser('mastering',  
             parents=[download_dataset_mastering.init_parser()], 
             add_help=False,
             help='Download mastering dataset.')
s.add_parser('all',  
             parents=[all_parser], 
             add_help=False,
             help='Download all datasets.')


args = parser.parse_args()

if args.subcommand == 'distortion':
    download_dataset_distortion.run_from_args(args)
elif args.subcommand == 'nonspeech':
    download_dataset_nonspeech.run_from_args(args)
elif args.subcommand == 'mastering':
    download_dataset_mastering.run_from_args(args)
elif args.subcommand == 'all':
    download_dataset_distortion.run_from_args(parser.parse_args(["distortion"]))
    download_dataset_nonspeech.run_from_args(parser.parse_args(["nonspeech"]))
    download_dataset_mastering.run_from_args(parser.parse_args(["mastering"]))
else:
    print('\nInvalid subcommand.\n')
    parser.print_help()