#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from deepafx import dafx_layer
from deepafx import mix_snr
from deepafx import lv2_plugin
from deepafx import utils
from deepafx import layers
from deepafx import losses
from deepafx import inception_model as inception

def getParameterInRange(param, minimum, maximum):
    """ Test Lambda function to get output of the model into parameter range (0,1) -> (min,max)"""

    return (maximum-minimum)*param + minimum




# Model to debud DAFx plugins, input is audio and parameters

def dafx_debug_mono_sequential(time_samples,
                               sr,
                               hop_samples,
                               batch_size,
                               dafx_params,
                               plugin_uri,
                               param_map,
                               gradient_method,
                               compute_signal_grad,
                               multiprocess,
                               output_length=1000,
                               stereo=False,
                               non_learnable_params_settings={},
                               new_params_range={},
                               fx_chain=False, greedy_pretraining=0,
                               ):
        
    
    dafx = dafx_layer.DAFXLayer(plugin_uri, 
                                 sr, 
                                 hop_samples, 
                                 param_map, 
                                 gradient_method=gradient_method,
                                 compute_signal_grad=compute_signal_grad,
                                 multiprocess=multiprocess,
                                 num_multiprocess=batch_size,
                                 stereo=stereo,
                                 non_learnable_params_settings=non_learnable_params_settings, 
                                 new_params_range=new_params_range,
                                 fx_chain=fx_chain,
                                 greedy_pretraining=greedy_pretraining,
                                 name='dafx')

    
    audio_time = tf.keras.layers.Input(shape=(time_samples,), name='audio_time')
    parameters = tf.keras.layers.Input(shape=(np.sum(dafx_params).item(),), name='parameters')

      
    #process the audio through the DAFX, given estimated params from inception_model
    
    audio_time_ = tf.keras.layers.Reshape(target_shape=(time_samples, 1), name='reshape_audio')(audio_time)
    
    crop_samples = int((time_samples-output_length)/2)
    
    audio_time_ = tf.keras.layers.Cropping1D(cropping=crop_samples,name='cropping_input')(audio_time_)

    dafx_output = dafx([audio_time_, parameters])

    flat = tf.keras.layers.Flatten(name='flat_audio')(dafx_output)
    
    # Compute the model
    full_model = tf.keras.models.Model(inputs=[audio_time,parameters], 
                                       outputs=flat, 
                                       name="full_model")
    
    return full_model, [dafx]

# Model to pretrain to default values
def deepAFx_pretraining(time_samples,
                               sr,
                               hop_samples,
                               batch_size,
                               dafx_params,
                               plugin_uri,
                               param_map,
                               gradient_method,
                               compute_signal_grad,
                               multiprocess,
                               encoder,
                               output_length=1000,
                               stereo=False,
                               non_learnable_params_settings={},
                               new_params_range={},
                               fx_chain=False, greedy_pretraining=0,
                               ):
        
    
    dafx = dafx_layer.DAFXLayer(plugin_uri, 
                                 sr, 
                                 hop_samples, 
                                 param_map, 
                                 gradient_method=gradient_method,
                                 compute_signal_grad=compute_signal_grad,
                                 multiprocess=multiprocess,
                                 num_multiprocess=batch_size,
                                 stereo=stereo,
                                 non_learnable_params_settings=non_learnable_params_settings, 
                                 new_params_range=new_params_range,
                                 fx_chain=fx_chain,
                                 greedy_pretraining=greedy_pretraining,
                                 name='dafx')

    
    logmelgram = layers.LogMelgramLayer(frame_length=1024,
                                        num_fft=1024,
                                        hop_length=256,
                                        num_mels=128,
                                        sample_rate=sr,
                                        f_min=0.0,
                                        f_max=sr // 2,
                                        eps=1e-6,
                                        norm=False, name='logMelgram')
    
    
    audio_time = tf.keras.layers.Input(shape=(time_samples,), name='audio_time')
    x = logmelgram(audio_time)
    
    if encoder is 'inception':
    
        encoder_model, backbone = inception.get_example_model(x.shape[1], np.sum(dafx_params).item())
        
    elif encoder is 'mobilenet':
        
        x = tf.keras.layers.BatchNormalization(name='input_norm')(x)
        encoder_model = tf.keras.applications.MobileNetV2(input_shape=(x.shape[1],x.shape[2],x.shape[3]), alpha=1.0,
                                                   include_top=True, weights=None, 
                                                   input_tensor=None, pooling=None,
                                                   classes=np.sum(dafx_params).item(),
                                                   classifier_activation='sigmoid')
     
    hidden_params = encoder_model(x)
    
      
    #process the audio through the DAFX, given estimated params from encoder_model
    
    # Compute the model
    full_model = tf.keras.models.Model(audio_time, 
                                       hidden_params, 
                                       name="full_model")
    
    return full_model, encoder_model, dafx


# deepAFx
def deepAFx(time_samples,
            sr,
            hop_samples,
            batch_size,
            dafx_params,
            plugin_uri,
            param_map,
            gradient_method,
            compute_signal_grad,
            multiprocess,
            encoder,
            output_length=1000,
            stereo=False,
            non_learnable_params_settings={},
            new_params_range={},
            fx_chain=False, greedy_pretraining=0,
            ):
        
    
    dafx = dafx_layer.DAFXLayer(plugin_uri, 
                                 sr, 
                                 hop_samples, 
                                 param_map, 
                                 gradient_method=gradient_method,
                                 compute_signal_grad=compute_signal_grad,
                                 multiprocess=multiprocess,
                                 num_multiprocess=batch_size,
                                 stereo=stereo,
                                 non_learnable_params_settings=non_learnable_params_settings, 
                                 new_params_range=new_params_range,
                                 fx_chain=fx_chain,
                                 greedy_pretraining=greedy_pretraining,
                                 name='dafx')

    
    logmelgram = layers.LogMelgramLayer(frame_length=1024,
                                        num_fft=1024,
                                        hop_length=256,
                                        num_mels=128,
                                        sample_rate=sr,
                                        f_min=0.0,
                                        f_max=sr // 2,
                                        eps=1e-6,
                                        norm=False, name='logMelgram')
    
    
    audio_time = tf.keras.layers.Input(shape=(time_samples,), name='audio_time')
    x = logmelgram(audio_time)
    
    if encoder == 'inception':
    
        encoder_model, backbone = inception.get_example_model(x.shape[1], np.sum(dafx_params).item())
        
    elif encoder == 'mobilenet':
        
        x = tf.keras.layers.BatchNormalization(name='input_norm')(x)
        encoder_model = tf.keras.applications.MobileNetV2(input_shape=(x.shape[1],x.shape[2],x.shape[3]), alpha=1.0,
                                                   include_top=True, weights=None, 
                                                   input_tensor=None, pooling=None,
                                                   classes=np.sum(dafx_params).item(),
                                                   classifier_activation='sigmoid')
     
    hidden_params = encoder_model(x)
      
    #process the audio through the DAFX, given estimated params from inception_model
    
    audio_time_ = tf.keras.layers.Reshape(target_shape=(time_samples, 1), name='reshape_audio')(audio_time)
    
    crop_samples = int((time_samples-output_length)/2)
    
    audio_time_ = tf.keras.layers.Cropping1D(cropping=crop_samples,name='cropping_input')(audio_time_)

    dafx_output = dafx([audio_time_, hidden_params])

    flat = tf.keras.layers.Flatten(name='flat_audio')(dafx_output)
    
    # Compute the model
    full_model = tf.keras.models.Model(audio_time, 
                                       flat, 
                                       name="full_model")
    
    return full_model, encoder_model, dafx