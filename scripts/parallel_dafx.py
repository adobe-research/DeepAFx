#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

import time
import numpy as np
import scipy as sp
import os
from deepafx import lv2_plugin
import multiprocessing as mp
import librosa

    
    
if __name__ == "__main__": 
    
    ### Define settings
    plugin_uri = 'http://lsp-plug.in/plugins/lv2/compressor_mono'
#     plugin_uri   = 'https://github.com/lucianodato/speech-denoiser' 
    sr = 22050
    hop_samples = sr #64
    signal_length_sec = 1
    wav_in_path  = '/home/code-base/sensei/noisy_speech.wav'
    num_items = 100
    
    # parallelization scheme
    k_pipe = True # If true, use a pipe for communication. If false, use queues.

    ### Read in the audio
    channels, sr = librosa.core.load(wav_in_path, sr=sr, mono=False)
    if len(channels.shape) == 1:
        channels = np.expand_dims(channels, axis=1)
    if signal_length_sec > 0:
        channels = channels[0:(int(sr*signal_length_sec)),:]
    items = []
    for i in range(num_items):
        items.append(channels)
    batch_audio = np.array(items)
        
        
    if True:
        print('RUNNING PARALLEL')
        
        # Create a multi LV2 object
        multi_lv2 = lv2_plugin.Multi_LV2_Plugin(num_items, 
                                     plugin_uri, 
                                     sr, 
                                     hop_samples=hop_samples, 
                                     k_pipe=k_pipe,
                                     verbose=False)

        tic = time.time()
        output_batch = multi_lv2.run_batch(batch_audio)
        par_time = time.time() - tic
        print('Par Time elapsed:', par_time)
        multi_lv2.shutdown()
        del multi_lv2 # trigger the thread join
    

    
    if True:
        print('RUNNING SEQUENCE')
        
        # Create a plugin for each signal
        plugins = []
        for item in items:
            lv2_dafx = lv2_plugin.LV2_Plugin(plugin_uri, 
                                             sr, 
                                             hop_samples=hop_samples,
                                             verbose=False)
            plugins.append(lv2_dafx)

        tic = time.time()
        for signal, lv2_dafx in zip(items, plugins):
            lv2_dafx.runs(signal.transpose())
        seq_time = time.time() - tic
        print('Seq Time elapsed:', seq_time)
        
    print('Delta Improvement', seq_time/par_time)