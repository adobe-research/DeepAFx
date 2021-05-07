#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
import time


# self-includes
from deepafx import dafx_layer
from deepafx import utils

tf.random.set_seed(0)
np.random.seed(0)


 

def multA(ip):
    """ Test Lambda function to apply a scalar gain to an input signal"""
    
    signal = ip[0]
    param = ip[1]
    
    return signal*K.repeat(param, 64)

def get_toy_norm_model(time_samples, gradient_method, compute_signal_grad):
    
    num_params = 1 # number of parameters to predict
    num_basis = 32 # time to frequency expansion parametrization
    
    # Create a DAFX layer
    plugin_uri = 'http://lsp-plug.in/plugins/lv2/compressor_mono'
    sr=16000
    hop_samples=64
    lv2_port_index = 21
    param_map = {0:lv2_port_index} # map 0th element of parameter vector to param id 21 DAFX
    dafx1 = dafx_layer.DAFXLayer(plugin_uri, sr, hop_samples, 
                                 param_map, 
                                 gradient_method=gradient_method,
                                 compute_signal_grad=False,
                                 name='compressor')
    d, param_min, param_max = dafx1.get_dafx_param_range(lv2_port_index)
    dafx1.set_dafx_param(18, 1)

    param_map = {0:21, 1:4} 
    dafx2 = dafx_layer.DAFXLayer(plugin_uri, sr, hop_samples, 
                                 param_map, 
                                 gradient_method=gradient_method,
                                 compute_signal_grad=True,
                                 name='compressor2')
    d2, param_min2, param_max2 = dafx2.get_dafx_param_range(4)
    dafx2.set_dafx_param(18, 1)
    
 
    audio_time = tf.keras.layers.Input(shape=(time_samples,1), name='audio_time')
    
    # Define the analyzer model
    x = tf.keras.layers.Dense(num_basis, activation='linear')(audio_time)
    x = tf.keras.layers.Reshape((time_samples,num_basis,1))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
  
    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global average over time axis
    #x = tf.math.reduce_mean(x, axis=1)
    x = tf.keras.backend.mean(x, axis=1, keepdims=False)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(.1)(x)
    hidden1 = tf.keras.layers.Dense(1, activation='linear')(x)
    hidden2 = tf.keras.layers.Dense(2, activation='linear')(x)
    
    # get min and max values from the DAFX layer
    hidden1 = tf.keras.backend.clip(hidden1, float(param_min), float(param_max))
    hidden2 = tf.keras.backend.clip(hidden2, float(param_min), float(param_max))

    
    # Create the analyzer model and invoke it
    analyzer_model = tf.keras.models.Model(inputs=[audio_time], 
                                           outputs=[hidden1, hidden2], 
                                           name="analyzer_model")
    hidden_params, hidden_params2 = analyzer_model(audio_time)
      
    #process the audio through the DAFX, given estimated params from analyzer
    dafx_output = dafx1([audio_time, hidden_params])
    # dafx_output = tf.keras.layers.Lambda(multA)([audio_time, hidden1])
    
    dafx_output = dafx2([dafx_output, hidden_params2])
    

    # Flatten for simple reshape to 1D
    flat = tf.keras.layers.Flatten()(dafx_output)
    
    # Compute the model
    full_model = tf.keras.models.Model(audio_time, 
                                       flat, 
                                       name="full_model")

    # Return both the full model and analyzer-only mode (no DAFX)
    return full_model, analyzer_model


# Synthesize toy data
num_samples = 10000
samples_per = 64*1 # 64*100
gradient_method = 'spsa' # or 'spsa'
compute_signal_grad = True
x_train, y_train = utils.get_toy_norm_data(num_samples, samples_per, range_db=50)

# Create the model and print
model, analyzer_model = get_toy_norm_model(samples_per, gradient_method, compute_signal_grad)
model.summary()
analyzer_model.summary()


full_model_filepath = 'output_full_model.h5'
analyzer_model_filepath = 'output_analyzer_model.h5'
log_filepath = 'output_log.csv'

# Train the model
if True:
    
    # Early stopping callback
    es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=0, 
            patience=5,
            verbose=0, 
            mode='min',
            baseline=None, 
            restore_best_weights=True)
    
    # define the checkpoint
    model_cp = tf.keras.callbacks.ModelCheckpoint(full_model_filepath, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min')
    
    csv_logger = CSVLogger(log_filepath, append=True, separator=';')
 

    #opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,loss='mean_squared_error')
    
    t = time.time()
    model.fit(x_train, 
              y_train, 
              epochs=10, 
              validation_split=0.3, 
              batch_size=16,
              shuffle=True, 
              verbose=True, 
              callbacks=[es, model_cp, csv_logger])
    print('Time elapsed:', time.time() - t)
    
    # Save the model
    model.save(full_model_filepath, save_format='tf') 
    analyzer_model.save(analyzer_model_filepath, save_format='tf') 

    

# Print out the dB energy of the first few examples to spot check
for i in range(20):
    print("x_train: {:.2f}".format(utils.db(x_train[i,:])), 
          "y_pred: {:.2f}".format(utils.db(y_pred[i,:])), end = ' '
         )
    
    sum_db = 0
    for j in range(y_pred_param.shape[0]):
        sum_db = sum_db + utils.db(y_pred_param[j,i])
        print("param_db{:d}".format(j)  + ": {:.2f}".format(utils.db(y_pred_param[j,i])), end = ' ')
    print(' sum_db:{:.2f}'.format(sum_db))


# Test reload the mode and inference
if False:
    
    #Load the model
    # Load the keras model with a custom object layer
    model = tf.keras.models.load_model(full_model_filepath, 
                                        custom_objects={'DAFXLayer': dafx_layer.DAFXLayer})
    analyzer_model = tf.keras.models.load_model(analyzer_model_filepath)
    
    # Test a few predict examples
    y_pred = model.predict( x_train )
    y_pred_param1 = analyzer_model.predict( x_train )

#     # Print out the dB energy of the first few examples to spot check
#     for i in range(20):
#         print("x_train: {:.2f}".format(utils.db(x_train[i,:])), 
#               "y_pred: {:.2f}".format(utils.db(y_pred[i,:])), 
#               "y_truth: {:.2f}".format(utils.db(y_train[i,:])), 
#               "param_db: {:.2f}".format(utils.db(y_pred_param1[i,:])), 
#               )
              






