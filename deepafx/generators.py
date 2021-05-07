#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import librosa
import soundfile as sf

import numpy as np
import scipy
import sox
import logging
logging.getLogger('sox').setLevel(logging.CRITICAL)
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from deepafx import dafx_layer
from deepafx import mix_snr
from deepafx import lv2_plugin
from deepafx import utils

import sys
import os

    
    
class Data_Generator_Default_Values(keras.utils.Sequence):

    def __init__(self,
                 x_audio,
                 dafxs,
                 batch_size,
                 length_samples,
                 steps_per_epoch=1,
                 sr=22050,
                 pad=0,
                 large_frame_length_secs=20,
                 center_frames=False):
        
        self.batch_size = batch_size
        self.x_audio = x_audio
        self.dafxs = dafxs
        self.sr = sr
        self.pad = pad
        self.steps_per_epoch = steps_per_epoch
        self.audio_time_len_samples = length_samples
        self.large_frame_length_secs = large_frame_length_secs
        self.indexes = np.repeat(np.arange(len(self.x_audio)), steps_per_epoch*batch_size)
        np.random.shuffle(self.indexes)
        self.center_frames = center_frames
        
        self.frame_idx = 0
        self.x_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        
 
        for dafx in self.dafxs:
            default = []
            if dafx.fx_chain:
                for i in range(len(dafx.mb.plugin_uri)):
                    d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values[i]),
                                                  np.asarray(dafx.mb.param_min[i]),
                                                  np.asarray(dafx.mb.param_max[i]))
                    default.append(d_)
                self.default_values = [item for sublist in default for item in sublist]
            else:
                d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values),
                                                  np.asarray(dafx.mb.param_min),
                                                  np.asarray(dafx.mb.param_max))
                default.append(d_)
                self.default_values = default
        
        self.default_values = np.asarray(self.default_values)
        self.default_values = np.expand_dims(self.default_values, axis=0)
        self.default_values = np.repeat(self.default_values, self.batch_size, axis=0)
           
   
        
        
      
           

    def __len__(self):
        return self.steps_per_epoch
    
    def on_finishing_sliding(self):
        np.random.shuffle(self.indexes)
        self.x_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        self.frame_idx=0
#         for dafx in self.dafxs:
#             dafx.reset_dafx_state(self.sr*1)

    def on_epoch_end(self):
        self.on_finishing_sliding()
        
    def load_frames(self):
        
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0 
        x_frames = []
        for i, index in enumerate(indexes):
       
            x_audio = self.x_audio[index]
            length_frame = int((self.large_frame_length_secs)*self.sr)
            sample = np.random.randint(0, x_audio.shape[0]-length_frame, size=None)
            
            x_audio_ = utils.getRandomTrim(x_audio,
                                         length_frame,
                                         pad=self.pad,
                                         start=sample)
           
              
            x_audio_ = utils.fadeIn(x_audio_, length=64)
            x_audio_ = utils.fadeOut(x_audio_, length=64)    
            
            length_frame = x_audio_.shape[0]

            x_audio_ = x_audio_[:length_frame]
            x_audio_ = np.pad(x_audio_, ((0, length_frame-x_audio_.shape[0]),), mode='constant')
         
            x_audio_w = utils.slicing(x_audio_,
                                      self.audio_time_len_samples+self.pad,
                                      1024,
                                      center=self.center_frames) 

            x_frames.append(x_audio_w)

            
        x_frames = np.asarray(x_frames)
       
        
        return x_frames
    
    

    def __getitem__(self, idx):
        
        'Generate one batch of data'
        
        num_segments = self.batch_size
        audio_time_len_samples = self.audio_time_len_samples
        
        audio_feat_input = np.zeros((num_segments, audio_time_len_samples+self.pad))
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0   
        
        if self.frame_idx // self.frame_total == 1:
            self.on_finishing_sliding()
            
        for i, index in enumerate(indexes):
            
            x_w = self.x_frames
            
            x = x_w[i, self.frame_idx]
            
            audio_feat_input[i] = x
            
        self.frame_idx += 1
        

        X = [audio_feat_input]
        Y = self.default_values

    
        return X, Y
    
    
class Data_Generator_Stateful_Distortion(keras.utils.Sequence):

    def __init__(self,
                 x_audio,
                 y_audio,
                 dafxs,
                 batch_size,
                 length_samples,
                 steps_per_epoch=1,
                 sr=22050,
                 pad=0,
                 crop=False,
                 output_length=1000,
                 snr_db=30, task=0, center_frames=False, default=False):
        

        
        
        self.batch_size = batch_size
        self.x_audio = x_audio
        self.y_audio = y_audio
        self.dafxs = dafxs
        self.sr = sr
        self.pad = pad
        self.steps_per_epoch = steps_per_epoch
        self.audio_time_len_samples = length_samples
        self.indexes = np.repeat(np.arange(len(self.x_audio)), steps_per_epoch*batch_size)
        self.crop = crop
        self.output_time_len_samples = output_length
        np.random.shuffle(self.indexes)
        self.center_frames = center_frames
        
        self.frame_idx = 0
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        
        self.default = default
        
        dafx = self.dafxs
        default = []
        if dafx.fx_chain:
            for i in range(len(dafx.mb.plugin_uri)):
                d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values[i]),
                                                  np.asarray(dafx.mb.param_min[i]),
                                                  np.asarray(dafx.mb.param_max[i]))
                default.append(d_)
            self.default_values = [item for sublist in default for item in sublist]
        else:
            d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values),
                                                np.asarray(dafx.mb.param_min),
                                                np.asarray(dafx.mb.param_max))
            default.append(d_)
            self.default_values = default
        
        self.default_values = np.asarray(self.default_values)
        self.default_values = np.expand_dims(self.default_values, axis=0)
        self.default_values = np.repeat(self.default_values, self.batch_size, axis=0)
        
      
           

    def __len__(self):
        return self.steps_per_epoch
    
    def on_finishing_sliding(self):
        np.random.shuffle(self.indexes)
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_idx=0
        if self.default is False:
            self.dafxs.reset_dafx_state(self.sr*1)

    def on_epoch_end(self):
        self.on_finishing_sliding()
        
    def load_frames(self):
        
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0 
        y_frames = []
        x_frames = []
        for i, index in enumerate(indexes):
       
            x_audio = self.x_audio[index]
            y_audio = self.y_audio[index]
            
            x_audio_w = utils.slicing(x_audio,
                                      self.audio_time_len_samples+self.pad, self.output_time_len_samples,
                                      center=self.center_frames) 
            y_audio_w = utils.slicing(y_audio,
                                      self.audio_time_len_samples+self.pad, self.output_time_len_samples,
                                      center=self.center_frames) 
            x_frames.append(x_audio_w)
            y_frames.append(y_audio_w)
            
        x_frames = np.asarray(x_frames)
        y_frames = np.asarray(y_frames)
        
        
        return x_frames, y_frames

    def __getitem__(self, idx):
        
        'Generate one batch of data'
        
        num_segments = self.batch_size
        audio_time_len_samples = self.audio_time_len_samples
        
        if self.crop:
            output_time_len_samples = self.output_time_len_samples
        else:
            output_time_len_samples = audio_time_len_samples
        
        audio_feat_input = np.zeros((num_segments, audio_time_len_samples+self.pad))
        audio_feat_output = np.zeros((num_segments, output_time_len_samples))
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0   
        
        if self.frame_idx // self.frame_total == 1:
            self.on_finishing_sliding()
            
        for i, index in enumerate(indexes):
            
            x_w = self.x_frames
            y_w = self.y_frames
            
            x = x_w[i, self.frame_idx]
            y = y_w[i, self.frame_idx]
    
            y = y[self.pad:]
            if self.crop:
                y = y[(audio_time_len_samples-output_time_len_samples)//2:(audio_time_len_samples+output_time_len_samples)//2]
            
            audio_feat_input[i] = x
            audio_feat_output[i] = y
            
        self.frame_idx += 1
        
        X = [audio_feat_input]
        
        if self.default:
            Y = self.default_values
            
        else:
            Y = [audio_feat_output]

        return X, Y
    
    
    
    
class Data_Generator_Stateful_Mastering(keras.utils.Sequence):

    def __init__(self,
                 x_audio,
                 y_audio,
                 dafxs,
                 batch_size,
                 length_samples,
                 steps_per_epoch=1,
                 sr=22050,
                 pad=0,
                 crop=False,
                 output_length=1000,
                 large_frame_length_secs=20,
                 snr_db=30, task=0,
                 center_frames=False, augment=True, default=False):
        
        self.batch_size = batch_size
        self.x_audio = x_audio
        self.y_audio = y_audio
        self.dafxs = dafxs
        self.sr = sr
        self.pad = pad
        self.steps_per_epoch = steps_per_epoch
        self.audio_time_len_samples = length_samples
        self.large_frame_length_secs = large_frame_length_secs
        self.indexes = np.repeat(np.arange(len(self.x_audio)), steps_per_epoch*batch_size)
        self.crop = crop
        self.output_time_len_samples = output_length
        np.random.shuffle(self.indexes)
        self.center_frames = center_frames
        self.augment = augment
        
        # if eq augmentation, run first eq_presets.py
        if self.augment:
            self.eq_hop_samples = 2048
            self.eq = lv2_plugin.LV2_Plugin('http://calf.sourceforge.net/plugins/Equalizer8Band',
                                 self.sr,
                                 hop_samples=self.eq_hop_samples)
            self.eq_presets = np.load('/home/code-base/runtime/deepafx/data/EQ_PRESETS.pkl.npy', allow_pickle=True)
            self.eq_stereo = True        
        
        self.frame_idx = 0
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        
        self.default = default
        
        dafx = self.dafxs
        default = []
        if dafx.fx_chain:
            for i in range(len(dafx.mb.plugin_uri)):
                d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values[i]),
                                                  np.asarray(dafx.mb.param_min[i]),
                                                  np.asarray(dafx.mb.param_max[i]))
                default.append(d_)
            self.default_values = [item for sublist in default for item in sublist]
        else:
            d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values),
                                                np.asarray(dafx.mb.param_min),
                                                np.asarray(dafx.mb.param_max))
            default.append(d_)
            self.default_values = default
        
        self.default_values = np.asarray(self.default_values)
        self.default_values = np.expand_dims(self.default_values, axis=0)
        self.default_values = np.repeat(self.default_values, self.batch_size, axis=0)
        
        
      
           

    def __len__(self):
        return self.steps_per_epoch
    
    def on_finishing_sliding(self):
        np.random.shuffle(self.indexes)
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        self.frame_idx=0
        if self.default is False:
            self.dafxs.reset_dafx_state(self.sr*1)

    def on_epoch_end(self):
        self.on_finishing_sliding()
        
    def load_frames(self):
        
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0 
        y_frames = []
        x_frames = []
        for i, index in enumerate(indexes):
       
            x_audio = self.x_audio[index]
            y_audio = self.y_audio[index]
            if self.augment:
                length_frame = ((self.large_frame_length_secs)*self.sr//self.eq_hop_samples)*self.eq_hop_samples
            else:
                length_frame = (self.large_frame_length_secs)*self.sr
                
            sample = np.random.randint(0, x_audio.shape[0]-length_frame, size=None)
            
            x_audio_ = utils.getRandomTrim(x_audio,
                                         length_frame,
                                         pad=self.pad,
                                         start=sample)
            y_audio_ = utils.getRandomTrim(y_audio,
                                         length_frame,
                                         pad=self.pad,
                                         start=sample)
              
            x_audio_ = utils.fadeIn(x_audio_, length=64)
            x_audio_ = utils.fadeOut(x_audio_, length=64)
            y_audio_ = utils.fadeIn(y_audio_, length=64)
            y_audio_ = utils.fadeOut(y_audio_, length=64)
            
            if self.augment:
            
                self.eq.reset_plugin_state(int((0.5)*self.sr//self.eq_hop_samples)*self.eq_hop_samples,
                                           stereo=self.eq_stereo)

                eq_params = random.choice(self.eq_presets)
                eq_params = utils.shift_preset_values(eq_params)

                for i in eq_params:
                    self.eq.set_param(i, eq_params[i])

                if self.eq_stereo:
                    x_audio_ = self.eq.runs_stereo(np.expand_dims(x_audio_,0))
                    x_audio_ = np.squeeze(x_audio_)
                else:
                    x_audio_ = self.eq.runs(np.expand_dims(x_audio_,0))
                    x_audio_ = np.squeeze(x_audio_)

                x_audio_ = x_audio_[self.eq_hop_samples:-self.eq_hop_samples]
                y_audio_ = y_audio_[self.eq_hop_samples:-self.eq_hop_samples]
            
            length_frame = x_audio_.shape[0]
            x_audio_ = x_audio_[:length_frame]
            y_audio_ = y_audio_[:length_frame]
            x_audio_ = np.pad(x_audio_, ((0, length_frame-x_audio_.shape[0]),), mode='constant')
            y_audio_ = np.pad(y_audio_, ((0, length_frame-y_audio_.shape[0]),), mode='constant')


            x_audio_w = utils.slicing(x_audio_,
                                      self.audio_time_len_samples+self.pad,
                                      self.output_time_len_samples,
                                      center=self.center_frames) 
            y_audio_w = utils.slicing(y_audio_,
                                      self.audio_time_len_samples+self.pad,
                                      self.output_time_len_samples,
                                      center=self.center_frames) 
            x_frames.append(x_audio_w)
            y_frames.append(y_audio_w)
            
            if self.augment:
                x_audio_ = x_audio_[:-self.eq_hop_samples]
                y_audio_ = y_audio_[:-self.eq_hop_samples]
            
        x_frames = np.asarray(x_frames)
        y_frames = np.asarray(y_frames)
        
        
        return x_frames, y_frames
    
    

    def __getitem__(self, idx):
        
        'Generate one batch of data'
        
        num_segments = self.batch_size
        audio_time_len_samples = self.audio_time_len_samples
        
        if self.crop:
            output_time_len_samples = self.output_time_len_samples
        else:
            output_time_len_samples = audio_time_len_samples
        
        audio_feat_input = np.zeros((num_segments, audio_time_len_samples+self.pad))
        audio_feat_output = np.zeros((num_segments, output_time_len_samples))
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0   
        
        if self.frame_idx // self.frame_total == 1:
            self.on_finishing_sliding()
            
        for i, index in enumerate(indexes):
            
            x_w = self.x_frames
            y_w = self.y_frames
            
            x = x_w[i, self.frame_idx]
            y = y_w[i, self.frame_idx]
    
            y = y[self.pad:]
            if self.crop:
                y = y[(audio_time_len_samples-output_time_len_samples)//2:(audio_time_len_samples+output_time_len_samples)//2]
            
            audio_feat_input[i] = x
            audio_feat_output[i] = y
            
        self.frame_idx += 1
        

        X = [audio_feat_input]
        if self.default:
            Y = self.default_values    
        else:
            Y = [audio_feat_output]

        return X, Y
    
    
    
class Data_Generator_Stateful_Nonspeech(keras.utils.Sequence):

    def __init__(self,
                 x_audio,
                 y_audio,
                 dafxs,
                 batch_size,
                 length_samples,
                 steps_per_epoch=1,
                 sr=22050,
                 pad=0,
                 crop=False,
                 output_length=1000,
                 large_frame_length_secs=20,
                 center_frames=False,
                 augment=False, default=False):
        

        
        
        self.batch_size = batch_size
        self.x_audio = x_audio
        self.y_audio = y_audio
        self.dafxs = dafxs
        self.sr = sr
        self.pad = pad
        self.steps_per_epoch = steps_per_epoch
        self.audio_time_len_samples = length_samples
        self.large_frame_length_secs = large_frame_length_secs
        self.indexes = np.repeat(np.arange(len(self.x_audio)), steps_per_epoch*batch_size)
        self.crop = crop
        self.output_time_len_samples = output_length
        np.random.shuffle(self.indexes)
        self.center_frames = center_frames
        self.augment = augment
   
        self.frame_idx = 0
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_total = self.x_frames.shape[1]
        
        self.default = default
        
        dafx = self.dafxs
        default = []
        if dafx.fx_chain:
            for i in range(len(dafx.mb.plugin_uri)):
                d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values[i]),
                                                  np.asarray(dafx.mb.param_min[i]),
                                                  np.asarray(dafx.mb.param_max[i]))
                default.append(d_)
            self.default_values = [item for sublist in default for item in sublist]
        else:
            d_ = utils.getParamInEncoderRange(np.asarray(dafx.mb.default_values),
                                                np.asarray(dafx.mb.param_min),
                                                np.asarray(dafx.mb.param_max))
            default.append(d_)
            self.default_values = default
        
        self.default_values = np.asarray(self.default_values)
        self.default_values = np.expand_dims(self.default_values, axis=0)
        self.default_values = np.repeat(self.default_values, self.batch_size, axis=0)
      
           

    def __len__(self):
        return self.steps_per_epoch
    
    def on_finishing_sliding(self):
        np.random.shuffle(self.indexes)
        self.x_frames, self.y_frames = self.load_frames()
        self.frame_idx=0
        if self.default is False:
            self.dafxs.reset_dafx_state(self.sr*1)

    def on_epoch_end(self):
        self.on_finishing_sliding()
        
    def load_frames(self):
        
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0 
        y_frames = []
        x_frames = []
        for i, index in enumerate(indexes):
       
            x_audio = self.x_audio[index]
            y_audio = self.y_audio[index]
            sample = np.random.randint(0, x_audio.shape[0]-(self.large_frame_length_secs)*self.sr, size=None)
            
            x_audio_ = utils.getRandomTrim(x_audio,
                                         (self.large_frame_length_secs)*self.sr,
                                         pad=self.pad,
                                         start=sample)
            y_audio_ = utils.getRandomTrim(y_audio,
                                         (self.large_frame_length_secs)*self.sr,
                                         pad=self.pad,
                                         start=sample)
            
            if self.augment:
                # gain and noise augmentation. noise between 30 and 50 db SNR
                noise = np.random.normal(0, 1, size=(self.large_frame_length_secs)*self.sr)
                snr = np.random.uniform(low=30, high=50)
                gain = np.random.uniform(low=0.1, high=1)
            
                x_audio_ = mix_snr.mix_speech_and_noise(x_audio_, noise, snr, self.sr,
                             speech_weighting='rms',
                             noise_weighting='rms',
                             quick=True)[0]
                
                # Uncomment to add noise also to the target
#                 y_audio_ = mix_snr.mix_speech_and_noise(y_audio_, noise, snr, self.sr,
#                              speech_weighting='rms',
#                              noise_weighting='rms',
#                              quick=True)[0]
                
                x_audio_ = gain * x_audio_
                y_audio_ = gain * y_audio_
            
            
            x_audio_ = utils.fadeIn(x_audio_, length=64)
            x_audio_ = utils.fadeOut(x_audio_, length=64)
            y_audio_ = utils.fadeIn(y_audio_, length=64)
            y_audio_ = utils.fadeOut(y_audio_, length=64)

            
            x_audio_w = utils.slicing(x_audio_,
                                      self.audio_time_len_samples+self.pad, self.output_time_len_samples,
                                      center=self.center_frames) 
            y_audio_w = utils.slicing(y_audio_,
                                      self.audio_time_len_samples+self.pad, self.output_time_len_samples,
                                      center=self.center_frames) 
            x_frames.append(x_audio_w)
            y_frames.append(y_audio_w)
            
        x_frames = np.asarray(x_frames)
        y_frames = np.asarray(y_frames)
        
        
        return x_frames, y_frames

    def __getitem__(self, idx):
        
        'Generate one batch of data'
        
        num_segments = self.batch_size
        audio_time_len_samples = self.audio_time_len_samples
        
        if self.crop:
            output_time_len_samples = self.output_time_len_samples
        else:
            output_time_len_samples = audio_time_len_samples
        
        audio_feat_input = np.zeros((num_segments, audio_time_len_samples+self.pad))
        audio_feat_output = np.zeros((num_segments, output_time_len_samples))
        
        indexes = self.indexes[0*self.batch_size:(0+1)*self.batch_size]
        assert len(indexes) > 0   
        
        if self.frame_idx // self.frame_total == 1:
            self.on_finishing_sliding()
            
        for i, index in enumerate(indexes):
            
            x_w = self.x_frames
            y_w = self.y_frames
            
            x = x_w[i, self.frame_idx]
            y = y_w[i, self.frame_idx]
    
            y = y[self.pad:]
            if self.crop:
                y = y[(audio_time_len_samples-output_time_len_samples)//2:(audio_time_len_samples+output_time_len_samples)//2]
            
            audio_feat_input[i] = x
            audio_feat_output[i] = y
            
        self.frame_idx += 1
        

        X = [audio_feat_input]
        if self.default:
            Y = self.default_values    
        else:
            Y = [audio_feat_output]

        return X, Y