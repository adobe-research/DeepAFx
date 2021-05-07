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
from deepafx import config_mastering
from deepafx import evaluate

tf.random.set_seed(0)
np.random.seed(0)

k = config_mastering.k

# Define Constants
kPathAudio = k['path_audio']
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
    
xTrainAudio = []
xValidAudio = []
xTestAudio = []
for path in xTrain:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    audio = utils.lufs_normalize(audio, kSR, -25.0) # audio input is normalized to -25 lufs
    xTrainAudio.append(audio)
for path in xValid:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    audio = utils.lufs_normalize(audio, kSR, -25.0)
    xValidAudio.append(audio)
for path in xTest:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    audio = utils.lufs_normalize(audio, kSR, -25.0)
    xTestAudio.append(audio)
    
yTrainAudio = []
yValidAudio = []
yTestAudio = []
for path in yTrain:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    yTrainAudio.append(audio)
for path in yValid:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    yValidAudio.append(audio)
for path in yTest:
    audio, _ = sf.read(path)
    audio = librosa.core.to_mono(audio.T)
    yTestAudio.append(audio)
    
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
    
    genTrain = generators.Data_Generator_Stateful_Mastering(xTrainAudio, yTrainAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=False, default=True)
    
    genValid = generators.Data_Generator_Stateful_Mastering(xValidAudio, yValidAudio, dafx,
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

genTrain = generators.Data_Generator_Stateful_Mastering(xTrainAudio, yTrainAudio, dafx,
                                               kBatchSize,
                                               kNumSamples,
                                               steps_per_epoch=kStepsPerEpoch,
                                               sr=kSR,
                                               pad=0,
                                               crop=True,
                                               output_length=kOutputLength,
                                               large_frame_length_secs=10, augment=False)

genValid = generators.Data_Generator_Stateful_Mastering(xValidAudio, yValidAudio, dafx,
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
np.save(kPathModels+kFullModel+'.gradient', [])

if dafx_layer.kLogGradient:
    import pickle
    with open(kPathModels+kFullModel+'.gradient.npy', "wb") as fp:   #Pickling
        pickle.dump(dafx_layer.kGradient, fp)

# Delete the model

dafx.shutdown()
del model, encoder, genTrain, genValid

for param in k.keys():
    print(param, k[param])
    
# Testing
evaluate.evaluate(model_weights_path, params_path, 'mastering', results_dir)  
    
