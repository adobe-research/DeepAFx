#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/

# Custom gradient example with a batch of vectors (signal) and vectors (of parameter) arguments
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


tf.random.set_seed(0)
np.random.seed(0)

def dafx_gain(signal, params):
    """dafx function
        
    Applies any (potentially non-differentiable) signal processing operator 
    on the signal as a function of the parameters
    """
    return signal*params


@tf.custom_gradient
def foo_custom_grad_numeric_batch(x, y):
    
    epsilon = 0.001

    def _func(xe, ye):
        """Function applied to each element of the batch"""
        return dafx_gain(xe, ye)
    
    def func(x, y):
    
        # Iterate over batch item
        z = []
        for i in range(x.shape[0]):
            z.append(_func(x[i], y[i]))
        z = tf.stack(z)
        return z

    def grad_fn(dy):
        """Gradient applied to each batch"""

        def _grad_fn(dye, xe, ye):
            """Gradient applied to each element of the batch"""
            
            # Grad w.r.t x. NOTE: this is approximate and should +-epsilon for each element
            J_plus = _func(xe + epsilon, ye)
            J_minus = _func(xe - epsilon, ye)
            gradx = (J_plus -  J_minus)/(2.0*epsilon) 
            vecJxe = gradx * dye

            # Grad w.r.t y
            yc = ye.numpy()

            # pre-allocate vector * Jaccobian output
            vecJye = np.zeros_like(ye)

            # Iterate over each parameter and compute the output
            for i in range(ye.shape[0]):

                yc[i] = yc[i] + epsilon
                J_plus = _func(xe, yc)
                yc[i] = yc[i] - 2*epsilon
                J_minus = _func(xe, yc)
                grady = (J_plus -  J_minus)/(2.0*epsilon) 
                yc[i] = yc[i] + 1*epsilon
                vecJye[i] = np.dot(np.transpose(dye), grady)

            return vecJxe, vecJye

        dy1 = []
        dy2 = []
        for i in range(dy.shape[0]):
            vecJxe, vecJye = _grad_fn(dy[i], x[i], y[i])
            dy1.append(vecJxe)
            dy2.append(vecJye)
        return tf.stack(dy1), tf.stack(dy2)

    return func(x, y), grad_fn


# Create a Keras Layer for the black-box audio FX
class DAFXLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DAFXLayer, self).__init__()

    def build(self, input_shape):
        self.oshape = input_shape
        super(DAFXLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs[0]
        params = inputs[1]     
        ret = tf.py_function(func=foo_custom_grad_numeric_batch, 
                             inp=[x, params], 
                             Tout=tf.float32)
        ret.set_shape(x.get_shape())
        return ret
    
def multA(ip):
    x = ip[0]
    a = ip[1]
    return x * K.repeat(a, 20)

def get_model(time_samples):
    
    num_params = 1
    num_basis = 32
 
    audio_time = keras.layers.Input(shape=(time_samples,1), name='audio_time')
    
    # Define the analyzer model
    x = tf.keras.layers.Dense(num_basis, activation='linear')(audio_time)
    x = keras.layers.Reshape((time_samples,num_basis,1))(x)
    x = keras.layers.Conv2D(64, kernel_size=2, activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
  
    x = keras.layers.Conv2D(64, kernel_size=2, activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = keras.layers.Conv2D(64, kernel_size=2, activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.BatchNormalization()(x)
    
    x = tf.math.reduce_mean(x, axis=1)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(.1)(x)
    hidden1 = tf.keras.layers.Dense(1, activation='linear')(x)
      
        
    #process the audio through the DAFX, given the input params
    dafx_output = DAFXLayer()([audio_time, hidden1])

    # Lambda example
    #dafx_output = tf.keras.layers.Lambda(multA)([audio_time, hidden1])
    
    # direct multiplication
    # dafx_output = audio_time  *  K.repeat(hidden1, 20) #Multiply()([audio_time, hidden1])
    
    flat = keras.layers.Flatten()(dafx_output)
    
    # Compute the model
    model = keras.models.Model(inputs=[audio_time], outputs=flat, name="full_model")

    return model

def db(x):
    return 20*np.log10(np.sqrt(np.mean(np.square(x))))

def get_data():

    # Synthesize random audio signals with random gains, predict normalized signals
    signals = np.random.randn(num_samples, samples_per) 
    x_train = signals.copy()
    y_train = 10*signals.copy()
    for i in range(signals.shape[0]):
        gain_dB = np.random.rand(1)[0]*40 - 20
        gain_linear = 10**((gain_dB)/20)
        temp = x_train[i,:]/np.sqrt(np.mean(np.square(x_train[i,:])))
        x_train[i,:] = temp*gain_linear
        y_train[i,:] = temp
    return x_train, y_train


num_samples = 10000
samples_per = 20

# Create the model
model = get_model(samples_per)

# Print the model summary
model.summary()

# Synthesize data
x_train, y_train = get_data()

es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min',
        baseline=None, restore_best_weights=True)

#opt = keras.optimizers.SGD(learning_rate=0.001)
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='mean_squared_error')
model.fit(x_train, y_train, 
          epochs=20, 
          validation_split=0.2, 
          batch_size=128,
          shuffle=True, 
          verbose=True, 
          callbacks=[es])

# Test a few predict examples
y_pred = model.predict( x_train )

for i in range(20):
    print('x_train, y_pred, y_truth', db(x_train[i,:]), db(y_pred[i,:]), db(y_train[i,:]) )

if False:

    with tf.GradientTape(persistent=True) as tape:

        a = 2*np.ones((2,5,1)) # signal, shape = batch x time x 1
        b = 3*np.ones((2,1,1)) #b[0,1:2,0] = 1 # params, shape = batch x params x 1
        x = tf.constant(a, dtype=tf.float64)
        y = tf.constant(b, dtype=tf.float64)
        tape.watch(x)
        tape.watch(y)

        z4 = foo_custom_grad_numeric_batch(x, y)**2
        z1a = tf.py_function(func=foo_custom_grad_numeric_batch, inp=[x, y], Tout=tf.float32)**2
        z1b = tf.py_function(func=foo_custom_grad_numeric_batch, inp=[x, y], Tout=tf.float32)**2


    print('\n')
    print('Grad w.r.t. x') 
    print("foo_custom_grad_numeric", tape.gradient(z4, x))  
    print("py_function", tape.gradient(z1a, x))  

    print('\n')
    print('Grad w.r.t. y')  
    print("foo_custom_grad_numeric", tape.gradient(z4, y))   
    print("py_function", tape.gradient(z1b, y))  

