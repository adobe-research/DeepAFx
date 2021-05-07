#!/usr/bin/env python3
#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import os 
import time

import librosa
import librosa.display
import sox
import sklearn

import soundfile as sf
import numpy as np
import scipy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import json

from deepafx import dafx_layer
from deepafx import lv2_plugin
from deepafx import utils
from deepafx import layers
from deepafx import losses
from deepafx import models


tf.random.set_seed(0)
np.random.seed(0)

import argparse
from pathlib import Path

def inference(path_model, params_path, name_task, input_file_path, output_filepath, dafx_wise=0, max_len_sec=50):
    """Evalue a model, given a params file, the task, and output directory
    """
    
    model_name = os.path.basename(path_model)
    kTask = name_task
    
    # Verify the task
    assert name_task in ['distortion', 'nonspeech', 'mastering'], 'task should be "distortion", "nonspeech" or "mastering"'

    # loads config
    k = np.load(params_path, allow_pickle=True).item()
    
    # Define Constants
    kPathAudio = k['path_audio']
    if kTask != 'mastering':
        kXRecording = k['x_recording']
        kYRecording = k['y_recording']
    kSR = k['sr']
    kNumSamples = k['num_samples']
    kBatchSize = k['batch_size']
    kStepsPerEpoch = k['steps_per_epoch']
    kEpochs = k['epochs']
    kPatience = k['patience']
    kHopSamples = k['hop_samples']
    kGradientMethod = k['gradient_method'] 
    kComputeSignalGradient = k['compute_signal_gradient']
    kMultiprocess = k['multiprocess']
    kParams = k['params']
    kPluginUri = k['plugin_uri']
    kParamMap = k['param_map']
    kOutputLength = k['output_length']
    kStereo = k['stereo']
    kSetNonTrainableParameters = k['set_nontrainable_parameters']
    kNewParameterRange = k['new_parameter_range']
    kFxChain = True
    kGreedyDafxPretraining = k['greedy_dafx_pretraining']
    kDefaultPretraining = k['default_pretraining']
    kEncoder = k['encoder']        
        
        
    # Load test dataset or makes partition, for some tasks random_state seed should be the same as in training script
    data, samplerate = sf.read(input_file_path)
    xTestAudio = librosa.resample(data.T, samplerate, 22050)
    xTestAudio = xTestAudio[:min(kSR*30, len(xTestAudio))]

    # Creates model
    print('Creating model...')
    model, encoder, dafx = models.deepAFx(kNumSamples,
                                            kSR,
                                            kHopSamples,
                                            kBatchSize,
                                            kParams,
                                            kPluginUri,
                                            kParamMap,
                                            kGradientMethod,
                                            kComputeSignalGradient,
                                            kMultiprocess,
                                            kEncoder,
                                            output_length=kOutputLength,
                                            stereo=kStereo, 
                                            non_learnable_params_settings=kSetNonTrainableParameters,
                                            new_params_range=kNewParameterRange,
                                            fx_chain=kFxChain)  
    
    # Loads model weights
    model.load_weights(path_model)
    
    # if fxchain=True loads the weights of the model that contains all plugins
    if kFxChain:
        dafx_wise = len(kPluginUri)
    else:
        dafx_wise = 1
    
    dafx.set_greedy_pretraining(dafx_wise)
        
    model.compile()

    # Computes objective metric and saves audio files and parameter automations
    #   It will process first 50 seconds of each test sample
    #   or full sample for the distortion task
    print('Processing the test file...')
    if True:

        xtest = xTestAudio

        # Gets parameter prediction
        layer_name = 'logMelgram'
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=model.get_layer(layer_name).output)
        
        kBlockSize = 64 # Block size for inference audio plugins
        xtest_w = utils.slicing(xtest, kNumSamples, kOutputLength)
        steps = int(np.ceil(len(xtest_w)/kBatchSize))

        intermediate_output = intermediate_layer_model(xtest_w)
        parameters = encoder.predict(intermediate_output,
                                     batch_size=kBatchSize,
                                     steps=steps,
                                     verbose=1)

        xtest_w = utils.slicing(xtest, kOutputLength, kOutputLength)
        xtest_w_smooth = utils.slicing(xtest, kBlockSize, kBlockSize)

        # creates dafx plugins for inference
        if kFxChain:

            dafx_inference_smooth = lv2_plugin.LV2_Plugin_Chain(kPluginUri, kStereo,
                                         kSR,
                                         hop_samples=kBlockSize)

            dafx_inference_smooth.reset_plugin_state(kBlockSize*100)

            for j, set_nontrainabel_paramenters in enumerate(kSetNonTrainableParameters):
                for i in set_nontrainabel_paramenters:
                    dafx_inference_smooth.set_param(j, i, set_nontrainabel_paramenters[i])

        else:

            dafx_inference_smooth = lv2_plugin.LV2_Plugin(kPluginUri,
                                     kSR,
                                     hop_samples=kBlockSize)

            for i in kSetNonTrainableParameters:
                dafx_inference_smooth.set_param(i, kSetNonTrainableParameters[i])

        # Low pass filter the parameters, whether is a fx_chain or a single effect

        b1, a1 = scipy.signal.butter(4, 0.5, 'low')
        b2, a2 = scipy.signal.butter(4, 0.001, 'low')

        try: 
            parameters_smooth_1 = []
            for i in range(np.sum(kParams)):
                filtered_signal = scipy.signal.filtfilt(b1, a1, parameters[:,i])
                parameters_smooth_1.append(filtered_signal)
            parameters_smooth_1 = np.asarray(parameters_smooth_1).T

            p_original_time = np.repeat(parameters, kOutputLength, axis=0)[:xtest.shape[0],:]
            p_smooth_1_time = np.repeat(parameters_smooth_1, kOutputLength, axis=0)[:xtest.shape[0],:]

            parameters_smooth_2 = []
            for i in range(np.sum(kParams)):
                filtered_signal = scipy.signal.filtfilt(b2, a2, p_smooth_1_time[:,i])
                parameters_smooth_2.append(filtered_signal)
            p_smooth_2_time = np.asarray(parameters_smooth_2).T

            parameters_resampled = scipy.signal.resample(p_smooth_2_time, xtest_w_smooth.shape[0])
            parameters_resampled = np.clip(parameters_resampled, 0, 1)
        
        # If parameters length is too short, it only applies one filter (this is for models with large output frames)
        except:

            p_original_time = np.repeat(parameters, kOutputLength, axis=0)[:xtest.shape[0],:]
            parameters_smooth_2 = []
            for i in range(np.sum(kParams)):
                filtered_signal = scipy.signal.filtfilt(b2, a2, p_original_time[:,i])
                parameters_smooth_2.append(filtered_signal)
            p_smooth_2_time = np.asarray(parameters_smooth_2).T

            parameters_resampled = scipy.signal.resample(p_smooth_2_time, xtest_w_smooth.shape[0])
            parameters_resampled = np.clip(parameters_resampled, 0, 1)


        ztest_smooth = utils.processFramesDAFx(dafx_inference_smooth,
                               kParamMap,
                               xtest_w_smooth,
                               parameters_resampled,
                               new_param_range=kNewParameterRange,
                               stereo=kStereo,
                               greedy_pretraining=dafx_wise)
        ztest_smooth = ztest_smooth[:xtest.shape[0]]
        z_smooth = ztest_smooth.copy()
        
        librosa.output.write_wav(output_filepath, z_smooth, kSR, norm=False)

        dafx.reset_dafx_state(kSR*1)

        
    dafx.shutdown()
    del model, encoder
    
def init_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--model_path', 
                        type=str)
    parser.add_argument('--params_path', 
                        type=str)
    parser.add_argument('--output_file', 
                        type=str, 
                        default='/home/code-base/scratch_space/test.wav')
    parser.add_argument('--dafx',
                            dest='dafx_wise',
                            const=0,
                            default=0,
                            action='store',
                            nargs='?',
                            type=int,
                            help='Optional dafx-wise plugin number to test the model')
    return parser

def run_from_args(args):
    inference(args.model_path, 
             args.params_path, 
             args.task, 
             args.input_file,
             args.output_file, 
             dafx_wise=args.dafx_wise)
    
def main():
    args = init_parser().parse_args()
    run_from_args(args)
    
if __name__ == "__main__":
    main()
