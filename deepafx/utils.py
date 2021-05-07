#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.python.framework import ops
import numpy as np
import scipy
import math
import librosa
import fnmatch
import os
from functools import partial
import pyloudnorm
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_distances


from deepafx import lv2_plugin

import matplotlib.pyplot as plt 

def db(x):
    """Computes the decible energy of a signal"""
    return 20*np.log10(np.sqrt(np.mean(np.square(x))))

def get_toy_norm_data(num_samples, samples_per, range_db=20):
    """Get data for toy normalization task
        
    """
    # Synthesize random audio signals with random gains, predict normalized signals
    signals = np.random.randn(num_samples, samples_per) 
    x_train = signals.copy()
    y_train = signals.copy()
    for i in range(signals.shape[0]):
        gain_dB = np.random.rand(1)[0]*range_db - range_db/2.0
        gain_linear = 10**((gain_dB)/20)
        temp = x_train[i,:]/np.sqrt(np.mean(np.square(x_train[i,:])))
        x_train[i,:] = temp*gain_linear
        y_train[i,:] = temp
    return x_train, y_train



def melspectrogram(y, mirror_pad=False):
    """Compute melspectrogram feature extraction
    
    Keyword arguments:
    signal -- input audio as a signal in a numpy object
    inputnorm -- normalization of output
    mirror_pad -- pre and post-pend mirror signals 
    
    Returns freq x time
               
    
    Assumes the input sampling rate is 22050Hz
    """
    
    # Extract mel.
    fftsize = 1024
    window = 1024
    hop = 512
    melBin = 128
    sr = 22050

    # mirror pad signal
    # first embedding centered on time 0 
    # last embedding centered on end of signal
    if mirror_pad:
        y = np.insert(y, 0, y[0:int(half_frame_length_sec * sr)][::-1])
        y = np.insert(y, len(y), y[-int(half_frame_length_sec * sr):][::-1])
    
    S = librosa.core.stft(y,n_fft=fftsize,hop_length=hop,win_length=window)
    X = np.abs(S)
    mel_basis = librosa.filters.mel(sr,n_fft=fftsize,n_mels=melBin)
    mel_S = np.dot(mel_basis,X)

    # value log compression
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.astype(np.float32)
    

    return mel_S


def getFilesPath(directory, extension):
    
    n_path=[]
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                n_path.append(os.path.join(path,name))
    n_path.sort()
                
    return n_path



def getRandomTrim(x, length, pad=0, start=None):
    
    length = length+pad
    if x.shape[0] <= length:
        x_ = x
        while(x.shape[0] <= length):
            x_ = np.concatenate((x_,x_))
    else:
        if start is None:
            start = np.random.randint(0, x.shape[0]-length, size=None)
        end = length+start
        if end > x.shape[0]:
            x_ = x[start:]
            x_ = np.concatenate((x_, x[:length-x.shape[0]]))
        else:
            x_ = x[start:length+start]
            
    return x_[:length]

def fadeIn(x, length=128):
    
    w = scipy.signal.hann(length*2, sym=True)
    w1 = w[0:length]
    ones = np.ones(int(x.shape[0]-length))
    w = np.append(w1, ones)
    
    return x*w

def fadeOut(x, length=128):
    
    w = scipy.signal.hann(length*2, sym=True)
    w2 = w[length:length*2]
    ones = np.ones(int(x.shape[0]-length))
    w = np.append(ones, w2)
    
    return x*w


def plotTimeFreq(audio, sr, n_fft=512, hop_length=128):
    
    n = len(audio)
    plt.figure(figsize=(14, 4*n))
    colors = list(plt.cm.viridis(np.linspace(0,1,n)))
    
    X = []
    X_db = []
    maxs = np.zeros((n,))
    mins = np.zeros((n,))
    maxs_t = np.zeros((n,))
    for i, x in enumerate(audio):
        X_ = librosa.stft(x, n_fft=n_fft,hop_length=hop_length)
        X_db_ = librosa.amplitude_to_db(abs(X_))
        X.append(X_)
        X_db.append(X_db_)
        maxs[i] = np.max(X_db_)
        mins[i] = np.min(X_db_)
        maxs_t[i] = np.max(np.abs(x))
    vmax = np.max(maxs)
    vmin = np.min(mins)
    tmax = np.max(maxs_t)
    
    for i, x in enumerate(audio):
        plt.subplot(n, 2, 2*i+1)
        librosa.display.waveplot(x, sr=sr, color=colors[i])
        plt.ylim(-tmax,tmax)
        plt.subplot(n, 2, 2*i+2)
        librosa.display.specshow(X_db[i], sr=sr, x_axis='time', y_axis='log',
                                 hop_length=hop_length, cmap='GnBu', vmax=vmax, vmin=vmin)
#         plt.colorbar(format='%+2.0f dB')


def getLatencyPlugin(plugin_uri, sr, stereo=False):
    
    lv2_dafx = lv2_plugin.LV2_Plugin(plugin_uri, sr, verbose=False)
    
    return lv2_dafx.get_latency_plugin(stereo=stereo)





def slicing(x, win_length, hop_length, center = True, windowing = False, pad = 0):
    # Pad the time series so that frames are centered
    if center:
#         x = np.pad(x, int((win_length-hop_length+pad) // 2), mode='constant')
        x = np.pad(x, ((int((win_length-hop_length+pad)//2), int((win_length+hop_length+pad)//2)),), mode='constant')
        
    # Window the time series.
    y_frames = librosa.util.frame(x, frame_length=win_length, hop_length=hop_length)
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
    else:
        window = 1.0 
    f = []
    for i in range(len(y_frames.T)):
        f.append(y_frames.T[i]*window)
    return np.float32(np.asarray(f)) 


def overlap(x, x_len, win_length, hop_length, windowing = True, rate = 1): 
    x = x.reshape(x.shape[0],x.shape[1]).T
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
        rate = rate*hop_length/win_length
    else:
        window = 1
        rate = 1
    n_frames = x_len / hop_length
    expected_signal_len = int(win_length + hop_length * (n_frames))
    y = np.zeros(expected_signal_len)
    for i in range(int(n_frames)):
            sample = i * hop_length 
            w = x[:, i]
            y[sample:(sample + win_length)] = y[sample:(sample + win_length)] + w*window
    y = y[int(win_length // 2):-int(win_length // 2)]
    return np.float32(y*rate)   

def getParamInRange(params, param_min, param_max):
    """Return the input params (0,1) scaled to (min, max) values."""
    return (param_max-param_min)*params + param_min   

def getParamInEncoderRange(params, param_min, param_max):
    """Return the input params (0,1) scaled to (min, max) values."""
    return (params - param_min)/(param_max-param_min) 


def getParameterSettings(dafx, param_map, new_param_range=None):
    param_min = []
    param_max = []
    param_range = {}

    for i in param_map:
        d, param_min_, param_max_ = dafx.get_param_range(param_map[i])
        param_range[i] = [float(str(param_min_)), float(str(param_max_))]
        
    # not pretty, but translates id from plugin to id in param map and then updates the param range.
    if new_param_range:
        new_param_range_ = {}
        for i in new_param_range:
            key = list(param_map.keys())[list(param_map.values()).index(i)]
            new_param_range_[key] = list(new_param_range[param_map[key]])

        for i in new_param_range_:
            param_range[i] = new_param_range_[i]

    for i in param_range:
        param_min.append(param_range[i][0])
        param_max.append(param_range[i][1])
    param_min = np.asarray(param_min)
    param_max = np.asarray(param_max)
    
    return param_range, param_min, param_max

#  # Iterate over the params vector and map it to the correct control on the dafx  
#     if isinstance(dafx, lv2_plugin.LV2_Plugin):
#         params = get_param_in_range(params, param_min, param_max)
#         for i in range(len(params)):
#             dafx.set_param(param_map[i], params[i])

#         # Process the audio and return
#         if stereo:
#             return dafx.runs_stereo(np.array(signal).transpose()).astype(np.float32).transpose()
#         else:
#             return dafx.runs(np.array(signal).transpose()).astype(np.float32).transpose()
    
#     elif isinstance(dafx, lv2_plugin.LV2_Plugin_Chain):
        
#         idx = 0
#         for j, param_map_plugin in enumerate(param_map):
#             params_plugin = params[idx:len(param_map_plugin)+idx]
#             idx=len(param_map_plugin)
#             params_plugin = get_param_in_range(np.array(params_plugin),
#                                                np.array(param_min[j]),
#                                                np.array(param_max[j]))
#             for i in range(len(params_plugin)):
#                 dafx.set_param(j, param_map_plugin[i], params_plugin[i])

# #         Process the audio and return
#         return dafx.runs(np.array(signal).transpose()).astype(np.float32).transpose()

def processFramesDAFx(dafx, param_map, x_frames, x_parameters, new_param_range=None, stereo=False, greedy_pretraining=None):
    
    if isinstance(dafx, lv2_plugin.LV2_Plugin):

        param_range, param_min, param_max = getParameterSettings(dafx,
                                                                 param_map,
                                                                 new_param_range=new_param_range)

        out = []
        for frame, params in zip(x_frames, x_parameters):

            params = getParamInRange(params, param_min, param_max)
            for i in range(len(params)):
                dafx.set_param(param_map[i], params[i])
            if stereo:
                out_ = dafx.runs_stereo(np.expand_dims(frame,0))
            else:
                out_ = dafx.runs(np.expand_dims(frame,0))

            out_ = np.squeeze(out_)
            out.append(out_)

        out = np.asarray(out) 
        out = out.flatten()
        return out
    
    elif isinstance(dafx, lv2_plugin.LV2_Plugin_Chain):
        
        if greedy_pretraining == 0:
            greedy_pretraining = len(param_map)
            
        out = []
        for frame, params in zip(x_frames, x_parameters):
            
#             forward(signal, params, dafx, param_map, param_min, param_max, stereo, greedy_pretraining=0)
#             idx = 0
#             for j, param_map_plugin in enumerate(param_map):
#                 params_plugin = params[idx:len(param_map_plugin)+idx]
#                 idx=len(param_map_plugin)
#                 params_plugin = get_param_in_range(np.array(params_plugin),
#                                                    np.array(param_min[j]),
#                                                    np.array(param_max[j]))
#                 for i in range(len(params_plugin)):
#                     dafx.set_param(j, param_map_plugin[i], params_plugin[i])

#     #         Process the audio and return
#             return dafx.runs(np.array(signal).transpose(), greedy_pretraining=greedy_pretraining).astype(np.float32).transpose()
            
            
            
            
            idx = 0
            out_ = frame.copy()
            for j, param_map_plugin in enumerate(param_map[:greedy_pretraining]):
                params_plugin = params[idx:len(param_map_plugin)+idx]
                idx+=len(param_map_plugin)
                dafx_plugin = dafx.plugins[j]
                new_param_range_plugin = new_param_range[j]
                param_range, param_min, param_max = getParameterSettings(dafx_plugin,
                                                                         param_map_plugin,
                                                                         new_param_range=new_param_range_plugin)

                params_plugin = getParamInRange(params_plugin, param_min, param_max)
#                 print(dafx_plugin.print_plugin_info())
#                 for p in params_plugin:
#                     print(p)
                for i in range(len(params_plugin)):
                    dafx_plugin.set_param(param_map_plugin[i], params_plugin[i])
            
                if stereo[j]:
                    out_ = dafx_plugin.runs_stereo(np.expand_dims(out_,0))
                else:
                    out_ = dafx_plugin.runs(np.expand_dims(out_,0))
                
                out_ = np.squeeze(out_)
#                 print(j, np.max(np.abs(out_)))
            out.append(out_)
            
            
#             out_ = dafx.runs(np.expand_dims(frame,0), greedy_pretraining=greedy_pretraining)
#             out_ = np.squeeze(out_)
#             out.append(out_)
        out = np.asarray(out) 
        out = out.flatten()
        
        return out
            
            


def highpassFiltering(x_list, f0, sr):

    b1, a1 = scipy.signal.butter(4, f0/(sr/2),'highpass')
    x_f = []
    for x in x_list:
        x_f_ = scipy.signal.filtfilt(b1, a1, x).copy(order='F')
        x_f.append(x_f_)
    return x_f

def lineartodB(x):
    return 20*np.log10(x) 
def dBtoLinear(x):
    return np.power(10,x/20)

def lufs_normalize(x, sr, lufs):

    # measure the loudness first 
    meter = pyloudnorm.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(x)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyloudnorm.normalize.loudness(x, loudness, lufs)
    
    return loudness_normalized_audio


def shift_preset_values(preset, f0 = [16,19,23,26,29,33,37,41], gains = [22,25,28,32,36,40]):
    
        for f in f0:
            coeff = np.random.uniform(low=0.75, high=1.25)
            preset[f] = np.clip(preset[f]*coeff, 10.0, 11000)

        for g in gains:
            coeff = np.random.uniform(low=0.5, high=1.5)
            preset[g] = np.clip(dBtoLinear(lineartodB(preset[g])*coeff), -36, 36)

        return preset
    

    


def getDistances(x,y):

    distances = {}
    distances['mae'] = mean_absolute_error(x, y)
    distances['mse'] = mean_squared_error(x, y)
    distances['euclidean'] = np.mean(paired_distances(x, y, metric='euclidean'))
    distances['manhattan'] = np.mean(paired_distances(x, y, metric='manhattan'))
    distances['cosine'] = np.mean(paired_distances(x, y, metric='cosine'))
   
    distances['mae'] = round(distances['mae'], 5)
    distances['mse'] = round(distances['mse'], 5)
    distances['euclidean'] = round(distances['euclidean'], 5)
    distances['manhattan'] = round(distances['manhattan'], 5)
    distances['cosine'] = round(distances['cosine'], 5)
    
    return distances

def getMFCC(x, sr, mels=128, mfcc=13, mean_norm=False):
    
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
                                     n_fft=1024, hop_length=256,
                                     n_mels=mels, power=2.0)
    melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    if mean_norm:
        mfcc -= (np.mean(mfcc, axis=0))
    return mfcc

        
def getMSE_MFCC(y_true, y_pred, sr, mels=128, mfcc=13, mean_norm=False):
    
    ratio = np.mean(np.abs(y_true))/np.mean(np.abs(y_pred))
    y_pred =  ratio*y_pred
    
    y_mfcc = getMFCC(y_true, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    z_mfcc = getMFCC(y_pred, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    
    return getDistances(y_mfcc[:,:], z_mfcc[:,:]) 
