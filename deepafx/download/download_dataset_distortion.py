#!/usr/bin/env python3
#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import os 
import sox
import argparse

from deepafx import utils


def download_distortion_dataset(output_dir, sr=22050, verbose=False):
    """Download the distortion dataset from the web.
    """

    kPathData = output_dir
    kDataName = os.path.join(kPathData,'distortion_dataset')

    # If folder doesn't exist, then create it.
    if not os.path.isdir(kPathData):
        os.makedirs(kPathData)

    # Downloading
    if verbose:
        print("Downloading dataset...")

    cmd = 'wget -O %s https://zenodo.org/record/3562442/files/AudioMDPI.zip?download=1 -P %s' % (kDataName, kPathData)
    os.system(cmd)

    if verbose:
        print("Dataset downloaded")

    # Unzipping
    cmd = 'unzip %s -d %s' % (kDataName, kPathData)
    os.system(cmd)

    cmd = 'rm %s' % kDataName
    os.system(cmd)

    # Copy and resample to 22050 kHz

    kPathDataOriginal = os.path.join(kPathData,'AudioMDPI/6176ChannelStrip/')
    kPathDataResampled = os.path.join(kPathData,'6176ChannelStrip_22050/')
    cmd = 'mkdir %s ; cp -r %s* %s' % (kPathDataResampled, kPathDataOriginal, kPathDataResampled)
    os.system(cmd)

    inputPathFiles = utils.getFilesPath(kPathDataOriginal, '*.wav')
    outputPathFiles = utils.getFilesPath(kPathDataResampled, '*.wav')
    inputPathFiles.sort()
    outputPathFiles.sort()
    tfm = sox.Transformer()

    for i,j in zip(inputPathFiles, outputPathFiles):
        tfm.rate(sr, quality='h')
        tfm.build(i,j)

        
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='/home/code-base/scratch_space/distortion/',
                        type=str,
                        help='Output directory to store the dataset.')
    return parser

def run_from_args(args):
    download_distortion_dataset(args.output_dir)
    
def main():
    
    args = init_parser().parse_args()
    run_from_args(args)
        

if __name__ == "__main__":
    main()

