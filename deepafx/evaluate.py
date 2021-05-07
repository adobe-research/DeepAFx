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
from deepafx import mix_snr
from deepafx import lv2_plugin
from deepafx import utils
from deepafx import layers
from deepafx import losses
from deepafx import models
from deepafx import generators
from fnmatch import fnmatch

tf.random.set_seed(0)
np.random.seed(0)

import argparse
from pathlib import Path

def evaluate(path_model, params_path, name_task, output_dir, dafx_wise=0, max_len_sec = 50):
    """Evalue a model, given a params file, the task, and output directory
    """
    
    model_name = os.path.basename(path_model)
    kTask = name_task
    
    # Verify the task
    assert name_task in ['distortion', 'nonspeech', 'mastering'], 'task should be "distortion", "nonspeech" or "mastering"'

    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("creating folder : ", output_dir)

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

    if kTask == 'nonspeech':
        
        print('Loading test partition...')

        xPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kXRecording), '*.wav')
        xTest = []
        for path in xPathFiles:
            if 'f9' in path or 'm9' in path:
                xTest.append(path)
                
        xTestAudio = []
        for path in xTest:
            audio, _ = sf.read(path)
            xTestAudio.append(audio)

        yPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kYRecording), '*.wav')
        yTest = []
        for path in yPathFiles:
            if 'f9' in path or 'm9' in path:
                yTest.append(path)

        yTestAudio = []
        for path in yTest:
            audio, _ = sf.read(path)
            yTestAudio.append(audio)

        xTestAudio = utils.highpassFiltering(xTestAudio, 100, kSR)
        yTestAudio = utils.highpassFiltering(yTestAudio, 100, kSR)

    elif kTask == 'distortion':
        
        xPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kXRecording), '*.wav')
        yPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kYRecording), '*.wav')

        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(xPathFiles,
                                                            yPathFiles,
                                                            test_size=0.10,
                                                            random_state=0)

        xTrain, xValid, yTrain, yValid = sklearn.model_selection.train_test_split(xTrain,
                                                            yTrain,
                                                            test_size=0.11111111111,
                                                            random_state=0)

        xTestAudio = []
        for path in xTest:
            audio, _ = sf.read(path)
            xTestAudio.append(audio)

        yTestAudio = []
        for path in yTest:
            audio, _ = sf.read(path)
            yTestAudio.append(audio)

    elif kTask == 'mastering':
        
        kPathFiles = utils.getFilesPath(kPathAudio, '*.wav')

        xPathFiles = []
        yPathFiles = []
        for path in kPathFiles:
            name = path.split('/')[-1]
            recording = name.split('-')[0]
            if recording[-1] is 'a':
                xPathFiles.append(path)
            elif recording[-1] is 'b':
                yPathFiles.append(path)
        xPathFiles.sort()
        yPathFiles.sort()

        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(xPathFiles,
                                                            yPathFiles,
                                                            test_size=0.10,
                                                            random_state=0)

        xTrain, xValid, yTrain, yValid = sklearn.model_selection.train_test_split(xTrain,
                                                            yTrain,
                                                            test_size=0.11111111111,
                                                            random_state=0)

        xTestAudio = []
        for path in xTest:
            audio, _ = sf.read(path)
            audio = librosa.core.to_mono(audio.T)
            audio = utils.lufs_normalize(audio, kSR, -25.0)
            xTestAudio.append(audio)

        yTestAudio = []
        for path in yTest:
            audio, _ = sf.read(path)
            audio = librosa.core.to_mono(audio.T)
            yTestAudio.append(audio)
            

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
    
    # Creates generators

    if kTask == 'nonspeech':

        genTest = generators.Data_Generator_Stateful_Nonspeech(xTestAudio, yTestAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=True)

    elif kTask == 'distortion':

        genTest = generators.Data_Generator_Stateful_Distortion(xTestAudio, yTestAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               center_frames=True)

    elif kTask == 'mastering':

        genTest = generators.Data_Generator_Stateful_Mastering(xTestAudio, yTestAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=False)
        
    # Loss_function
    spectral_loss = losses.multiScaleSpectralLoss(loss_type='L2',
                                                   mag_weight=1.,
                                                   logmag_weight=1.,
                                                   time_loss=True,
                                                   time_loss_type='L1',
                                                   time_loss_weight=10.0,
                                              fft_sizes=(1024,),
                                             overlap=0.0,
                                             time_shifting=True,
                                             batch_size=kBatchSize)
    
    # Loads model weights
    model.load_weights(path_model)
    
    # if fxchain=True loads the weights of the model that contains all plugins
    if kFxChain:
        dafx_wise = len(kPluginUri)
    else:
        dafx_wise = 1
    
    dafx.set_greedy_pretraining(dafx_wise)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss=spectral_loss,
                       metrics=['mae'])

    # Computes test loss
    print('Computing test loss...')
    metrics = {}
    K.set_learning_phase(0)
    score = model.evaluate(genTest,
                       steps=int(kStepsPerEpoch*0.1),
                       batch_size=kBatchSize,
                       verbose=1,
                       return_dict=True)
    metrics['test_losses'] = score

    
    
    # Computes objective metric and saves audio files and parameter automations
    #   It will process first 50 seconds of each test sample
    #   or full sample for the distortion task
    secs = [0, max_len_sec] 
    mfcc_cosine = []
    print('Processing the test dataset...')
    for idx_track in range(len(xTestAudio)):

        xtest = xTestAudio[idx_track][secs[0]*kSR:secs[1]*kSR]
        ytest = yTestAudio[idx_track][secs[0]*kSR:secs[1]*kSR]

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

        # creates dafx plugins for inference. dafx_inference_smooth runs smoothed parameters
        if kFxChain:

            dafx_inference = lv2_plugin.LV2_Plugin_Chain(kPluginUri, kStereo,
                                         kSR,
                                         hop_samples=kOutputLength)

            dafx_inference_smooth = lv2_plugin.LV2_Plugin_Chain(kPluginUri, kStereo,
                                         kSR,
                                         hop_samples=kBlockSize)

            dafx_inference.reset_plugin_state(kOutputLength*100)
            dafx_inference_smooth.reset_plugin_state(kBlockSize*100)

            for j, set_nontrainabel_paramenters in enumerate(kSetNonTrainableParameters):
                for i in set_nontrainabel_paramenters:
                    dafx_inference.set_param(j, i, set_nontrainabel_paramenters[i])
                    dafx_inference_smooth.set_param(j, i, set_nontrainabel_paramenters[i])

        else:

            dafx_inference = lv2_plugin.LV2_Plugin(kPluginUri,
                                     kSR,
                                     hop_samples=kOutputLength)

            dafx_inference_smooth = lv2_plugin.LV2_Plugin(kPluginUri,
                                     kSR,
                                     hop_samples=kBlockSize)

            for i in kSetNonTrainableParameters:
                dafx_inference.set_param(i, kSetNonTrainableParameters[i])
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

        # praces frames using parameters and smoothed parameters
        ztest = utils.processFramesDAFx(dafx_inference,
                                   kParamMap,
                                   xtest_w,
                                   parameters,
                                   new_param_range=kNewParameterRange,
                                   stereo=kStereo,
                                   greedy_pretraining=dafx_wise)
        ztest = ztest[:xtest.shape[0]]

        ztest_smooth = utils.processFramesDAFx(dafx_inference_smooth,
                               kParamMap,
                               xtest_w_smooth,
                               parameters_resampled,
                               new_param_range=kNewParameterRange,
                               stereo=kStereo,
                               greedy_pretraining=dafx_wise)
        ztest_smooth = ztest_smooth[:xtest.shape[0]]

        
        # Saves audio files and parameter automation
        x = xtest.copy()
        y = ytest.copy()
        z = ztest.copy()
        z_smooth = ztest_smooth.copy()
        
        librosa.output.write_wav(os.path.join(output_dir,f'{idx_track}_input.wav'),
                                 x, kSR, norm=False)
        librosa.output.write_wav(os.path.join(output_dir,f'{idx_track}_target.wav'),
                                 y, kSR, norm=False)
        librosa.output.write_wav(os.path.join(output_dir,f'{idx_track}_output.wav'),
                                 z_smooth, kSR, norm=False)
        np.save(os.path.join(output_dir, f'{idx_track}_parameters'), parameters_resampled)
        
        # Uncomment to save audio output and paramenters withouth smoothing:      
#         librosa.output.write_wav(kPathModels+'results/'+kFullModel+f'_{idx_track}_output_nonsmooth.wav',
#                                  z, kSR, norm=False)
#         np.save(kPathModels+'results/'+kFullModel+f'_{idx_track}_parameters', parameters)

        d = utils.getMSE_MFCC(y, z_smooth, kSR, mean_norm=False)    
        mfcc_cosine.append(d['cosine'])

        dafx.reset_dafx_state(kSR*1)
    
    metrics['mfcc_cosine'] = str(round(np.mean(mfcc_cosine), 5))
    print(metrics)
    print('audio samples saved at '+output_dir)
    
    with open(os.path.join(output_dir, model_name+'_test_losses.json'), 'w') as outfile:
        json.dump(metrics, outfile)
        
    dafx.shutdown()
    del model, encoder, genTest
    
def init_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('params_path', type=str)
    parser.add_argument('output_dir', type=str)
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
    evaluate(args.model_path, 
             args.params_path, 
             args.task, 
             args.output_dir, 
             dafx_wise=args.dafx_wise)
    
def main():
    args = init_parser().parse_args()
    run_from_args(args)
    
if __name__ == "__main__":
    main()
