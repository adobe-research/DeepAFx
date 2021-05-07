#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import os 
import time

import librosa
import sox
import soundfile as sf
import numpy as np
import scipy
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from deepafx import dafx_layer
from deepafx import mix_snr
from deepafx import lv2_plugin
from deepafx import utils
from deepafx import layers
from deepafx import losses
from deepafx import models
from deepafx import generators
from deepafx import config_nonspeech
from deepafx import evaluate

tf.random.set_seed(0)
np.random.seed(0)

k = config_nonspeech.k

# Define Constants
kPathAudio = k['path_audio']
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
kPathModels = k['path_models']
kFxChain = True
kGreedyDafxPretraining = k['greedy_dafx_pretraining']
kDefaultPretraining = k['default_pretraining']
kEncoder = k['encoder']


# If folder doesn't exist, then create it.
if not os.path.isdir(kPathModels):
    os.makedirs(kPathModels)
    print("created folder : ", kPathModels)
    
# Selects model ID and saves config.    
kModelID = 0
while os.path.exists(kPathModels + 'full_model_%s' % kModelID):
    kModelID += 1
kFullModel = 'full_model_%s' % kModelID
kEncoderModel = 'encoder_%s' % kModelID
kPathModels = kPathModels+kFullModel+'/'
os.makedirs(kPathModels)
k['name'] = kFullModel
k['time'] = []
model_weights_path = kPathModels+kFullModel+'.h5'
params_path = kPathModels+kFullModel+'.params'
results_dir = os.path.join(kPathModels,'results')
os.system('touch ' + model_weights_path) # Save placeholder for model weights
os.makedirs(results_dir, exist_ok=True)
np.save(params_path, k) # Save config
params_path=params_path+'.npy' # update output params file name

# Get dataset paths and partition
xPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kXRecording), '*.wav')
xTrain = []
xValid = []
xTest = []
for path in xPathFiles:
    if 'f10' in path or 'm10' in path:
        xValid.append(path)
    elif 'f9' in path or 'm9' in path:
        xTest.append(path)
    else:
        xTrain.append(path)
    
xTrainAudio = []
xValidAudio = []
xTestAudio = []
for path in xTrain:
    audio, _ = sf.read(path)
    xTrainAudio.append(audio)
for path in xValid:
    audio, _ = sf.read(path)
    xValidAudio.append(audio)
for path in xTest:
    audio, _ = sf.read(path)
    xTestAudio.append(audio)
    
yPathFiles = utils.getFilesPath(os.path.join(kPathAudio,kYRecording), '*.wav')
yTrain = []
yValid = []
yTest = []
for path in yPathFiles:
    if 'f10' in path or 'm10' in path:
        yValid.append(path)
    elif 'f9' in path or 'm9' in path:
        yTest.append(path)
    else:
        yTrain.append(path)
    
yTrainAudio = []
yValidAudio = []
yTestAudio = []
for path in yTrain:
    audio, _ = sf.read(path)
    yTrainAudio.append(audio)
for path in yValid:
    audio, _ = sf.read(path)
    yValidAudio.append(audio)
for path in yTest:
    audio, _ = sf.read(path)
    yTestAudio.append(audio)

# Highpass filter to remove mich knocks and thuds
xTrainAudio = utils.highpassFiltering(xTrainAudio, 100, kSR)
yTrainAudio = utils.highpassFiltering(yTrainAudio, 100, kSR)
xValidAudio = utils.highpassFiltering(xValidAudio, 100, kSR)
yValidAudio = utils.highpassFiltering(yValidAudio, 100, kSR)
xTestAudio = utils.highpassFiltering(xTestAudio, 100, kSR)
yTestAudio = utils.highpassFiltering(yTestAudio, 100, kSR)
    
# Default pretraining
if kDefaultPretraining:

    

    model, encoder, dafx = models.deepAFx_pretraining(kNumSamples,
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
    
    genTrain = generators.Data_Generator_Stateful_Nonspeech(xTrainAudio, yTrainAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=False, default=True)
    
    genValid = generators.Data_Generator_Stateful_Nonspeech(xValidAudio, yValidAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=False, default=True)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss='mse'
                 )
    model.summary()
    print('Pretraining to default values...')

    es = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=4,
                    verbose=1, 
                    mode='min',
                    baseline=None, 
                    restore_best_weights=True)

    chk = tf.keras.callbacks.ModelCheckpoint(filepath=kPathModels+kFullModel+'_pretraining_chk.h5',
                                                monitor='val_loss',
                                                mode='min',
                                                verbose=1,
                                                save_best_only=True, 
                                                save_weights_only=False)

    t = time.time()

    hist1 = model.fit(genTrain,
                           steps_per_epoch=kStepsPerEpoch,
                           epochs=100, 
                           verbose=1, 
                           callbacks=[es,chk],
                      validation_data=genValid, 
                       validation_steps=int(kStepsPerEpoch*0.1),
                         )
    timeElapsed = (time.time() - t)/60/60
    k['time'].append(timeElapsed)
    print('Time elapsed:', timeElapsed)
    np.save(kPathModels+kFullModel+'.history_pretraining', hist1.history)
    np.save(kPathModels+kFullModel+'.params', k)

    dafx.shutdown()
    del model, encoder, genTrain, genValid
    
    
# Create model
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


# Create generators

genTrain = generators.Data_Generator_Stateful_Nonspeech(xTrainAudio, yTrainAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=True)

genValid = generators.Data_Generator_Stateful_Nonspeech(xValidAudio, yValidAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=True)



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

# # Compile

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss=spectral_loss,
                       metrics=['mae']
             )

model.summary()
encoder.summary()


if kDefaultPretraining:
    model.load_weights(kPathModels+kFullModel+'_pretraining_chk.h5')

if kGreedyDafxPretraining:
    start_dafx = 1
else:
    start_dafx = len(kPluginUri)
    
for i in range(start_dafx,len(kPluginUri)+1):

    # update dafx layers
    print('Adding ' + kPluginUri[i-1])
    dafx.set_greedy_pretraining(i)
        

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss=spectral_loss,
                       metrics=['mae'],
             )


    # Create/reset callbacks

    es = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                min_delta=0, 
                patience=kPatience,
                verbose=1, 
                mode='min',
                baseline=None, 
                restore_best_weights=False)

    chk = tf.keras.callbacks.ModelCheckpoint(filepath=kPathModels+kFullModel+f'_chk_{i}.h5',
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True, 
                                            save_weights_only=False)
    
    loss_chk = tf.keras.callbacks.ModelCheckpoint(filepath=kPathModels+kFullModel+f'_loss_chk_{i}.h5',
                                            monitor='loss',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True, 
                                            save_weights_only=False)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=kPatience-5,
                                      cooldown=1,
                                      verbose=1,
                                      mode='min',
                                      min_lr=0.0000016)

    # Training
    t = time.time()

    hist1 = model.fit(genTrain,
                       steps_per_epoch=kStepsPerEpoch,
                       epochs=kEpochs, 
                       verbose=1, 
                       callbacks=[es,chk,loss_chk,reduce_lr],
                       validation_data=genValid, 
                       validation_steps=int(kStepsPerEpoch*0.1),
                     )
    timeElapsed = (time.time() - t)/60/60
    k['time'].append(timeElapsed)
    print('Time elapsed:', timeElapsed)

    # Save the model

    model.save(kPathModels+kFullModel+f'_{i}.h5', save_format='tf') 
    encoder.save(kPathModels+kEncoderModel+f'_{i}.h5', save_format='tf')
    np.save(kPathModels+kFullModel+f'.history_{i}', hist1.history)
    np.save(kPathModels+kFullModel+'.params', k)
    
model.save(kPathModels+kFullModel+'.h5', save_format='tf') 
encoder.save(kPathModels+kEncoderModel+'.h5', save_format='tf')


if dafx_layer.kLogGradient:
    import pickle
    np.save(kPathModels+kFullModel+'.gradient', [])
    with open(kPathModels+kFullModel+'.gradient.npy', "wb") as fp:   #Pickling
        pickle.dump(dafx_layer.kGradient, fp)


# Delete the model

dafx.shutdown()
del model, encoder, genTrain, genValid

for param in k.keys():
    print(param, k[param])
    
evaluate.evaluate(model_weights_path, params_path, 'nonspeech', results_dir)

