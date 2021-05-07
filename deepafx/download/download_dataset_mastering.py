#!/usr/bin/env python3
#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

import ast
import json
import os
from joblib import Parallel, delayed
from binaryornot.check import is_binary
import tempfile
import sox
import numpy as np
import soundfile as sf
from scipy import signal
from deepafx import utils

import argparse

DOWNLOAD_URL='https://multitracksearch.cambridge-mt.com/ms-mtk-search.htm'

def get_mixing_secrets_html_special(temp_text_file='/scratch-space/mixing_secrets_raw.txt'):
    """ Return a cached version of the HTML
    """
    f = open(temp_text_file, "r")
    stuff = f.read()
    things = stuff.split('\n\n')
    return things

def get_mixing_secrets_html(url):
    """ Download and return the mixing secrets HTML
    """
    print(url)
    # Create a temp file 
    with tempfile.NamedTemporaryFile(suffix='.txt') as fp:
        temp_text_file = fp.name

        # Download the web info
        os.system('curl ' + url + ' --output ' + temp_text_file)

        # Read in curl'd file
        f = open(temp_text_file, "r")
        stuff = f.read()

        # Find the file sets
        start_deliminator = 'var projects = ['
        end_deliminator = '];'
        start = stuff.find(start_deliminator) + len(start_deliminator)
        end = stuff.find(end_deliminator)
        substring = stuff[start:end]

        # split into separate db items
        things = substring.split('\n\n')

    return things



def download_item(item):
    """ Download a given pair of unprocessed and processed files  
    """
    
    url, output_dir, overwrite = item
    output_filepath = os.path.join(output_dir, os.path.basename(url))
    output_filepath_wav = os.path.join(output_dir, os.path.basename(url)[0:-4] + '.wav')
    
    ib = True
    
    extra = ' >/dev/null 2>&1'
    if not os.path.exists(output_filepath) or overwrite:
        
        # Download the file
        cmd = 'wget -O ' + output_filepath + ' ' + url
        ret = os.system(cmd)
            
        ib = is_binary(output_filepath)
        if not ib:
            print("BAD", output_filepath)
            cmd = 'rm ' + output_filepath + extra
            ret = os.system(cmd)
        else:
            
            if '.mp3' in output_filepath:
                # Convert to wav using sox
                cmd = 'sox -v 0.99 ' + output_filepath + ' ' + output_filepath_wav + extra
                ret = os.system(cmd)
                cmd = 'rm ' + output_filepath + extra
                ret = os.system(cmd)
            
        return not ib
    else:
        return False
    
def download_all(mixing_secrets_url, 
                 output_dir, 
                 download=True, 
                 overwrite=False, 
                 num_cpus=28, 
                 verbose=False):
    """Download all unprocessed and processed file pairs."""
    
    output_metadata_json = 'mixing_secrets_mastering.json'
    
    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read the mixing secrets raw web javascript
    things = get_mixing_secrets_html(mixing_secrets_url)
    #things = get_mixing_secrets_html_special()

    database = {}
    for thing in things:
        try:
            # Convert javascript object to python dictionary
            d = ast.literal_eval(thing.strip())[0]
            d['ic'] = ''
            project_type = d['pt']
            artist_project = d['a'] + d['p']
            if artist_project not in database:
                database[artist_project] = {}
            database[artist_project][project_type] = d
        except:
            if verbose:
                print('Bad entry, ignoring')

    # Filter on master/unmaster paired data only
    unmastered_db = {}
    for key in database:
        entries_map = database[key]
        if 'Mstr' in entries_map and 'Full' in entries_map:
            unmastered_db[key] = entries_map

    # Create final metadata
    final_json = {}
    for i, key in enumerate(unmastered_db):
        entries_map = database[key]

        artist = entries_map['Mstr']['a']
        project = entries_map['Mstr']['p']
        unmastered_url = entries_map['Mstr']['pv']
        full_url = entries_map['Full']['pv']
        
        elem = {}
        elem['artist'] = artist
        elem['project'] = project
        elem['unmastered_url'] = unmastered_url
        elem['mastered_url'] = full_url
        elem['unmastered_file'] = os.path.basename(unmastered_url)[0:-4] + '.wav'
        elem['mastered_file'] = os.path.basename(full_url)[0:-4] + '.wav'
        final_json[i] = elem

    # Download files
    items = []
    for key in final_json:
        unmastered_url = final_json[key]['unmastered_url']
        mastered_url = final_json[key]['mastered_url']

        items.append((unmastered_url, output_dir, overwrite))
        items.append((mastered_url, output_dir, overwrite))


    if download:
        results = Parallel(n_jobs=num_cpus)(delayed(download_item)(i) for i in items)  

    # Save metadata
    with open(os.path.join(output_dir, output_metadata_json), 'w') as outfile:
        json.dump(final_json, outfile)


def make_same_shape(x, y):
    """ Make two arrays the same shape
    """
    
    if x.shape[0] > y.shape[0]:
        d = x.shape[0] - y.shape[0]
        pad = np.zeros((d, x.shape[1]))
        y = np.concatenate((y, pad))
        
        
    elif y.shape[0] > x.shape[0]:
        d = y.shape[0] - x.shape[0]
        pad = np.zeros((d, x.shape[1]))
        x = np.concatenate((x, pad))
    
    return x, y


def align_all(input_dir, output_dir):
    """ Align unprocessed and processed files. 
    
    Impose some custom alignment for certain files.
    """

    pre_adjustment = {} # (cut in unmastered, cut in mastered)
    pre_adjustment['Showmonster_UnmasteredWAV.wav'] = (23, 0)
    pre_adjustment['CanYouSayTheSame_UnmasteredWAV.wav'] = (40, 0)
    pre_adjustment['DeadEnemies_UnmasteredWAV.wav'] = (10, 0)
    pre_adjustment['FlecheDOr_UnmasteredWAV.wav'] = (20, 0)
    pre_adjustment['Kaathaadi_UnmasteredWAV.wav'] = (40, 34)
    pre_adjustment['HeartOnMyThumb_UnmasteredWAV.wav'] = (77, 0)

    override_adjustment = {} # (cut in unmastered, cut in mastered)
    override_adjustment['BurningBridges_UnmasteredWAV.wav'] = 0

    post_adjustment = {} #(start, end)
    post_adjustment['JesuJoy_UnmasteredWAV.wav'] = (0, 185.0)
    post_adjustment['NonLoDiroColLabbro_UnmasteredWAV.wav'] = (0, 129.0)
    post_adjustment['OwnWayToBoogie_UnmasteredWAV.wav'] = (4.9, 174.0)
    post_adjustment['DadsGlad_UnmasteredWAV.wav'] = (0, 243)
    post_adjustment['MarshMarigoldsSong_UnmasteredWAV.wav'] = (0, 72)
    post_adjustment['TrudeTheBumblebee_UnmasteredWAV.wav'] = (0, 147)
    post_adjustment['FlecheDOr_UnmasteredWAV.wav'] = (0, 151)  
    post_adjustment['Lament_UnmasteredWAV.wav'] = (5.4, 291)  
    post_adjustment['Air_UnmasteredWAV.wav'] = (6.0, 213) 
    post_adjustment['WeFeelAlright_UnmasteredWAV.wav'] = (0.0, 163) 
    post_adjustment['DragMeDown_UnmasteredWAV.wav'] = (16.6, 254) 
    post_adjustment['GhostBitch_UnmasteredWAV.wav'] = (0, 261) 
    post_adjustment['ThroughMyEyes_UnmasteredWAV.wav'] = (1.9, 202) 
    post_adjustment['BigDummyShake_UnmasteredWAV.wav'] = (1.9, 194) 
    post_adjustment['Kaathaadi_UnmasteredWAV.wav'] = (11.3, 150)  
    post_adjustment['HeartOnMyThumb_UnmasteredWAV.wav'] = (13, 213) 
    post_adjustment['LocationLocation_UnmasteredWAV.wav'] = (6.7, 313)  
    post_adjustment['Mute_UnmasteredWAV.wav'] = (0, 225)  
    post_adjustment['MeuBem_UnmasteredWAV.wav'] = (15, 336) 
    post_adjustment['AyniNehirde_UnmasteredWAV.wav'] = (4, 279) 
    post_adjustment['BananaSplit_UnmasteredWAV.wav'] = (0, 269)
    post_adjustment['LostMyWay_UnmasteredWAV.wav'] = (7, 409)
    post_adjustment['Signs_UnmasteredWAV.wav'] = (7, 245)
    post_adjustment['Flames_UnmasteredWAV.wav'] = (2, 145)
    post_adjustment['DeadEnemies_UnmasteredWAV.wav'] = (12, 254)
    post_adjustment['Chasque_UnmasteredWAV.wav'] = (2.6, 199)

    input_metadata_json = os.path.join(input_dir, 'mixing_secrets_mastering.json')
    output_metadata_json = os.path.join(output_dir, 'mixing_secrets_mastering.json')
    os.makedirs(output_dir, exist_ok=True)

    # Read in metadata
    with open(input_metadata_json) as json_file:
        data = json.load(json_file)

    # Iterate over data
    remove_list = []
    for key in data:
        unmastered_path = os.path.join(input_dir, data[key]['unmastered_file'])
        mastered_path = os.path.join(input_dir, data[key]['mastered_file'])
        try:
            # read in the file pairs
            (unmastered, unmastered_rate) = sf.read(unmastered_path)    
            (mastered, mastered_rate) = sf.read(mastered_path)
        except:
            remove_list.append(key)
            continue
            

        if data[key]['unmastered_file'] in pre_adjustment: 
            (cut_unmaster, cut_master) = pre_adjustment[data[key]['unmastered_file']]
            unmastered = unmastered[int(cut_unmaster*unmastered_rate):,:] 
            mastered = mastered[int(cut_master*mastered_rate):,:] 


        sos = signal.butter(4, 2000.0/(mastered_rate/2.0), btype='high', output='sos')

        unmastered, mastered = make_same_shape(unmastered, mastered)

        center = int(unmastered.shape[0]/2)
        left = center - int(mastered_rate*10)
        right = center + int(mastered_rate*10)

        x = signal.sosfilt(sos, unmastered[left:right,0])
        y = signal.sosfilt(sos, mastered[left:right,0])
        c = signal.correlate(x, y)
        zero_index = int(len(x))-1 
        shift = zero_index - np.argmax(c)

        if data[key]['unmastered_file'] in override_adjustment:
            shift = override_adjustment[data[key]['unmastered_file']]

        if shift < 0:
            pad = np.zeros((abs(shift), mastered.shape[1]))
            mastered = np.concatenate((pad, mastered))
        else:
            pad = np.zeros((abs(shift), unmastered.shape[1]))
            unmastered = np.concatenate((pad, unmastered))

        # Make the same shape
        unmastered, mastered = make_same_shape(unmastered, mastered)

        if data[key]['unmastered_file'] in post_adjustment:
            (start, end) = post_adjustment[data[key]['unmastered_file']]
            unmastered = unmastered[int(start*unmastered_rate):int(end*unmastered_rate),:]
            mastered = mastered[int(start*mastered_rate):int(end*mastered_rate),:]

        data[key]['unmastered_file'] = key + 'a-' + data[key]['unmastered_file']
        data[key]['mastered_file'] = key + 'b-' + data[key]['mastered_file'] 
        unmastered_path_output = os.path.join(output_dir, data[key]['unmastered_file'])
        mastered_path_output = os.path.join(output_dir, data[key]['mastered_file'])

        # Write the data
        sf.write(unmastered_path_output, unmastered, unmastered_rate)
        sf.write(mastered_path_output, mastered, mastered_rate)

    for key in remove_list:
        data.pop(key, None)

    with open(output_metadata_json, 'w') as outfile:
        json.dump(data, outfile)
    
def resample(input_dir, output_dir, sr=22050):
    """ Resample all files to a given sampling rate
    """
    
    os.makedirs(output_dir, exist_ok=True)
    inputPathFiles = utils.getFilesPath(input_dir, '*.wav')
    outputPathFiles = utils.getFilesPath(output_dir, '*.wav')
    inputPathFiles.sort()
    tfm = sox.Transformer()
    
    for f in inputPathFiles:
        output_path = os.path.join(output_dir, os.path.basename(f))
        tfm.rate(sr, quality='h')
        tfm.build(f,output_path)

        
        
def init_parser():
    """Initialze a command line parse to use this functionality as a script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                            type=str, 
                            choices=['download', 'align', 'resample', 'all'], 
                            default='all')
    parser.add_argument('--mixing_secrets_url',
                            default=DOWNLOAD_URL,
                            type=str,
                            help='Specify the mixing secrets dataset URL.')   
    parser.add_argument('--download_dir',
                            default='/home/code-base/scratch_space/mastering/mixing_secrets_raw',
                            type=str,
                            help='Temporary raw output directory to store files.')
    parser.add_argument('--output_dir',
                            default='/home/code-base/scratch_space/mastering/mixing_secrets_mastering',
                            type=str,
                            help='Final output directory of full-quality dataset.')
    parser.add_argument('--resampled_dir',
                            default='/home/code-base/scratch_space/mastering/mixing_secrets_mastering_22050',
                            type=str,
                            help='Final resampled output used for training.')
    return parser
    
def download_mastering_dataset(download_dir, 
                               output_dir, 
                               resampled_dir, 
                               mixing_secrets_url=DOWNLOAD_URL):
    """Download the mastering dataset from the web.
    
    download_dir {str} - folder to store raw downloaded files
    output_dir {str} - folder to store high-quality composite dataset
    resampled_dir {str} - folder to store downsampled composite dataset
    mixing_secrets_url {str} - URL of mixing secrets website
    """
    download_all(mixing_secrets_url, download_dir)
    align_all(download_dir, output_dir)
    resample(output_dir, resampled_dir) 

def run_from_args(args):
    if args.mode == 'download':
        download_all(args.mixing_secrets_url, args.download_dir)
    elif args.mode == 'align':
        align_all(args.download_dir, args.output_dir)
    elif args.mode == 'resample':
        resample(args.output_dir, args.resampled_dir)
    elif args.mode == 'all':
        download_mastering_dataset(args.download_dir, 
                                   args.output_dir, 
                                   args.resampled_dir, 
                                   mixing_secrets_url=args.mixing_secrets_url)   
    

def main():
    args = init_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()

