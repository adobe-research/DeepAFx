#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np
import librosa

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import functools


class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(
        self, frame_length, num_fft, hop_length, num_mels, sample_rate, f_min, f_max, eps, norm=False, **kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self.num_freqs = num_fft // 2 + 1
        self.norm = norm
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mels,
            num_spectrogram_bins=self.num_freqs,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        stfts = tf.signal.stft(
            input,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            fft_length=self.num_fft,
            pad_end=False,  # librosa test compatibility
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(  # assuming channel_first, so (b, c, f, t)
            tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0]
        )
        log_melgrams = _tf_log10(melgrams + self.eps)
        log_melgrams = tf.expand_dims(log_melgrams, 3)
        
#         if self.norm:
            # TO DO; normalization function
            
        return log_melgrams

    def get_config(self):
        config = {
            'frame_length': self.frame_length,
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'eps': self.eps,
            'norm': self.norm
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))
    
# LogMel loss function    
def logMelgramLoss(frame_length=1024, 
                    num_fft=1024,
                    hop_length=512,
                    num_mels=64,
                    sample_rate=22050,
                    f_min=0.0,
                    f_max=22050 // 2,
                    eps=1e-6,
                    norm=False):
    
    logmelgram = LogMelgramLayer(frame_length=frame_length, 
                                num_fft=num_fft,
                                hop_length=hop_length,
                                num_mels=num_mels,
                                sample_rate=sample_rate,
                                f_min=f_min,
                                f_max=f_max,
                                eps=eps, 
                                norm=norm)

    def loss(y_true,y_pred):
        
        Y_true = logmelgram(y_true)
        Y_pred = logmelgram(y_pred)
        
        return tf.keras.losses.mean_squared_error(Y_true, Y_pred)
    
   # Return a function
    return loss



# Tensor function that applies high freq emphasis filter
def high_freq_pre_emphasis(y_true, y_pred, coeff):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
        
    is_1d = (len(y_true.shape) == 1)
    if is_1d:
        y_true = y_true[tf.newaxis, :] 
        y_pred = y_pred[tf.newaxis, :] 
        
    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    
    paddings = tf.constant([[0,0,],[1,0],[0,0]])
    
    y_pred_ = tf.keras.layers.Cropping1D(cropping=(0,1))(y_pred)
    y_pred_ = tf.pad(y_pred_, paddings, 'CONSTANT')
    y_true_ = tf.keras.layers.Cropping1D(cropping=(0,1))(y_true)
    y_true_ = tf.pad(y_true_, paddings, 'CONSTANT')
    
    y_pred_ = tf.math.scalar_mul(coeff, y_pred_)
    y_true_ = tf.math.scalar_mul(coeff, y_true_)
    
    y_true = tf.math.subtract(y_true, y_true_)
    y_pred = tf.math.subtract(y_pred, y_pred_)
    
    y_true = y_true[:,:,0]
    y_pred = y_pred[:,:,0]

    
    return y_true, y_pred


# Tensor function that computes time shifting
# NOTE: for mastering tasks, a better result is obtained when samples_delay_max=300 (due to a longer group delay)
def compute_time_shifting(y_true, y_pred, batch_size=100, samples_delay_max=100):
    
    
    @tf.function
    def shift(x, y, time_shift):
        
        x_frame = x
        y_frame = y
        
            
        if time_shift > 0 and time_shift < samples_delay_max:
            
            x_frame = x_frame[:-time_shift]
            y_frame = y_frame[time_shift:]
            
        return x_frame, y_frame

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
        
    is_1d = (len(y_true.shape) == 1)
    if is_1d:
        y_true = y_true[tf.newaxis, :] 
        y_pred = y_pred[tf.newaxis, :] 
        
    frame_length = y_pred.shape[1]
    zeroes = frame_length//2
    fft_length = frame_length + 2*zeroes
    paddings = tf.constant([[0,0,],[zeroes,zeroes]])
    y_true_padded = tf.pad(y_true, paddings, mode='CONSTANT', constant_values=0)
    y_pred_padded = tf.pad(y_pred, paddings, mode='CONSTANT', constant_values=0)

    A = tf.signal.stft(
        y_true_padded,
        frame_length=fft_length,
        frame_step=fft_length,
        window_fn=None,
        pad_end=False,  # librosa test compatibility
        )
    Ar = tf.math.conj(A)
    B = tf.signal.stft(
        y_pred_padded,
        frame_length=fft_length,
        frame_step=fft_length,
        window_fn=None,
        pad_end=False,  # librosa test compatibility
        )
        
    C = tf.math.multiply(Ar,B)
    c = tf.signal.inverse_stft(C,
                                fft_length,
                                fft_length,
                                fft_length=None,
                                window_fn=None,
                                name=None)
        
    time_shift = fft_length-tf.math.argmax(tf.abs(c),axis=-1)
          
    y_true_frame = []
    y_pred_frame = []
    for i in range(batch_size):
 
        y_true_frame_, y_pred_frame_ = shift(y_true[i],y_pred[i],time_shift[i])
        
        paddings = tf.constant([[0,samples_delay_max,]])
        y_true_frame_ = tf.pad(y_true_frame_, paddings, mode='CONSTANT', constant_values=0)
        y_pred_frame_ = tf.pad(y_pred_frame_, paddings, mode='CONSTANT', constant_values=0)
        y_true_frame_ = y_true_frame_[:frame_length]
        y_pred_frame_ = y_pred_frame_[:frame_length]
        y_true_frame.append(y_true_frame_)
        y_pred_frame.append(y_pred_frame_)
               
    return tf.convert_to_tensor(y_true_frame), tf.convert_to_tensor(y_pred_frame)



