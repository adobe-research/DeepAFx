#!/usr/bin/env python3
#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

import os
import sox
from deepafx import utils
import argparse
import glob
    
def download_nonspeech_dataset(input_dir, output_dir, sr=22050):
    """Resample the DAPS dataset to a given value.
    
    The dataset must exist on the current machine.
    """
    
    kPathData = input_dir
    kDataName = os.path.join(kPathData,'daps.tar.gz')
    
    # If folder doesn't exist, then create it.
    if not os.path.isdir(kPathData):
        os.makedirs(kPathData)

    cmd = 'wget -O %s https://zenodo.org/record/4660670/files/daps.tar.gz?download=1 -P %s' % (kDataName, kPathData)
    os.system(cmd)
    
    # Untar
    print('Extracting tar...')
    cmd = 'tar -xf ' + kDataName + ' -C ' + kPathData
    os.system(cmd)

    file_list = glob.glob(os.path.join(input_dir, '**/.*'), recursive=True)
    for fp in file_list:
        try:
            os.remove(fp)
        except OSError:
            print("Error while deleting file")

    # Copy
    cmd = 'mkdir %s ; cp -r %s* %s' % (output_dir, os.path.join(kPathData, 'daps/'), output_dir)
    print(cmd)
    os.system(cmd)

    inputPathFiles = utils.getFilesPath(input_dir, '*.wav')
    outputPathFiles = utils.getFilesPath(output_dir, '*.wav')
    inputPathFiles.sort()
    outputPathFiles.sort()
    tfm = sox.Transformer()

    for i,j in zip(inputPathFiles, outputPathFiles):
        tfm.rate(sr, quality='h')
        tfm.build(i,j)


def init_parser():
    """Initialze a command line parse to use this functionality as a script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                            default='/home/code-base/scratch_space/nonspeech/',
                            type=str,
                            help='Specify input dataset directory of DAPS.')
    parser.add_argument('--output_dir',
                            default='/home/code-base/scratch_space/nonspeech/daps_22050/',
                            type=str,
                            help='Specify outout, resampled dataset directory of DAPS.')
    return parser

def run_from_args(args):
    download_nonspeech_dataset(args.input_dir, args.output_dir)
    
def main():
    args = init_parser().parse_args()
    run_from_args(args)
        

if __name__ == "__main__":
    main()

