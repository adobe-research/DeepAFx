# Copyright 2021 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Note:
# Original DDSP code follows the original license above. 
# Small modifications follows the LICENSE file within this repo.

import numpy as np
import librosa

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import functools

from deepafx.layers import compute_time_shifting


def tf_float32(x):
    
    """Ensure array/tensor is a float32 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
    else:
        return tf.convert_to_tensor(x, tf.float32)
                                    
def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    
    """Differentiable stft in tensorflow, computed in batch."""
    audio = tf_float32(audio)
    assert frame_size * overlap % 2.0 == 0.0
    s = tf.signal.stft(
          signals=audio,
          frame_length=int(frame_size),
          frame_step=int(frame_size * (1.0 - overlap)),
          fft_length=int(frame_size),
          pad_end=pad_end)
    return s
                                    

def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    mag = tf.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    return tf_float32(mag)
                                    
def safe_log(x, eps=1e-5):
    return tf.math.log(x + eps)

                                    
def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20):
    
    """Make vector equal to the expected length.
    Feature extraction functions like `compute_loudness()` or `compute_f0` produce
    feature vectors that vary in length depending on factors such as `sample_rate`
    or `hop_size`. This function corrects vectors to the expected length, warning
    the user if the difference between the vector and expected length was
    unusually high to begin with.
    Args:
        vector: Numpy 1D ndarray. Shape [vector_length,]
        expected_len: Expected length of vector.
        pad_value: Value to pad at end of vector.
        len_tolerance: Tolerance of difference between original and desired vector
          length.
        use_tf: Make function differentiable by using tensorflow.
    Returns:
        vector: Vector with corrected length.
    Raises:
        ValueError: if `len(vector)` is different from `expected_len` beyond
        `len_tolerance` to begin with.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError('Vector length: {} differs from expected length: {} '
                         'beyond tolerance of : {}'.format(vector_len,
                                                           expected_len,
                                                           len_tolerance))

    is_1d = (len(vector.shape) == 1)
    vector = vector[tf.newaxis, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        vector = tf.pad(
            vector, ((0, 0), (0, n_padding)),
            mode='constant',
            constant_values=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector
                                    
                                    
                                    
def mean_difference(target, value, loss_type='L1', weights=None):
    
    """Common loss functions.
    Args:
        target: Target tensor.
        value: Value tensor.
        loss_type: One of 'L1', 'L2', or 'COSINE'.
        weights: A weighting mask for the per-element differences.
    Returns:
        The average loss.
    Raises:
        ValueError: If loss_type is not an allowed value.
    """
    difference = target - value
    weights = 1.0 if weights is None else weights
    loss_type = loss_type.upper()
    if loss_type == 'L1':
        return tf.reduce_mean(tf.abs(difference * weights))
    elif loss_type == 'L2':
        return tf.reduce_mean(difference**2 * weights)
    elif loss_type == 'COSINE':
        return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
    else:
        raise ValueError('Loss type ({}), must be '
                         '"L1", "L2", or "COSINE"'.format(loss_type))
        
        
def diff(x, axis=-1):
    
    """Take the finite difference of a tensor along an axis.
    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.
    Returns:
        d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
        ValueError: Axis out of range for tensor.
    """
    shape = x.shape.as_list()
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                         (axis, len(shape)))

    begin_back = [0 for _ in range(len(shape))]
    begin_front = [0 for _ in range(len(shape))]
    begin_front[axis] = 1

    shape[axis] -= 1
    slice_front = tf.slice(x, begin_front, shape)
    slice_back = tf.slice(x, begin_back, shape)
    d = slice_front - slice_back
    return d

def compute_loudness(audio,
                     sample_rate=22080,
                     frame_rate=345,
                     n_fft=2048,
                     range_db=120.0,
                     ref_db=20.7):
    
    """Perceptual loudness in dB, relative to white noise, amplitude=1.
    Function is differentiable if use_tf=True.
    Args:
        audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
          [batch_size,].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: Fft window size.
        range_db: Sets the dynamic range of loudness in decibles. The minimum
          loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by
          (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
          corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
          slight dependence on fft_size due to different granularity of perceptual
          weighting.
        use_tf: Make function differentiable by using tensorflow.
    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    if sample_rate % frame_rate != 0:
        raise ValueError(
            'frame_rate: {} must evenly divide sample_rate: {}.'
            'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
            .format(frame_rate, sample_rate))

    # Make inputs tensors for tensorflow.
    audio = tf_float32(audio) 


    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    audio = audio[tf.newaxis, :] if is_1d else audio

    # Take STFT.
    hop_size = sample_rate // frame_rate
    overlap = 1 - hop_size / n_fft
    s = stft(audio, frame_size=n_fft, overlap=overlap, pad_end=True)

    # Compute power
    amplitude = tf.abs(s)
    log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) 
    amin = 1e-20  # Avoid log(0) instabilities.
    power_db = log10(tf.maximum(amin, amplitude))
    power_db *= 20.0

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[tf.newaxis, tf.newaxis, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = tf.maximum(loudness, -range_db)
    mean = tf.reduce_mean

    # Average over frequency bins.
    loudness = mean(loudness, axis=-1)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    # TO DO FIX THIS
    if audio.shape[-1] is None:
        len_audio=n_fft
    else:
        len_audio=audio.shape[-1]
        
    n_secs = len_audio/float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)


    # Pad with `-range_db` noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(
          loudness, expected_len, -range_db)
    return loudness

class SpectralLoss(tf.keras.layers.Layer):
    
    """Multi-scale spectrogram loss."""

    def __init__(self,
                   fft_sizes=(2048, 1024, 512, 256, 128, 64),
                   loss_type='L1',
                   mag_weight=1.0,
                   overlap=0.75,
                   delta_time_weight=0.0,
                   delta_delta_time_weight=0.0,
                   delta_freq_weight=0.0,
                   delta_delta_freq_weight=0.0,
                   logmag_weight=0.0,
                   loudness_weight=0.0,
                   sr = 22080,
                   time_loss=False,
                   time_loss_type='L2',
                   time_loss_weight=1.0,
                   time_shifting=False,
                   high_freq_emphasis=False,
                   batch_size=100,
                   name='spectral_loss'):
        super().__init__(name=name)
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.overlap = overlap
        self.delta_time_weight = delta_time_weight
        self.delta_delta_time_weight = delta_delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.delta_delta_freq_weight = delta_delta_freq_weight
        self.logmag_weight = logmag_weight
        self.loudness_weight = loudness_weight
        self.sr = sr
        self.time_loss=time_loss
        self.time_loss_type=time_loss_type
        self.time_loss_weight=time_loss_weight
        self.time_shifting=time_shifting
        self.batch_size=batch_size
        self.high_freq_emphasis=high_freq_emphasis

    def call(self, target_audio, audio):

        loss = 0.0
        loss_ops = []

        for size in self.fft_sizes:
            loss_op = functools.partial(compute_mag,
                                        size=size, overlap=self.overlap)
            loss_ops.append(loss_op)

        if self.time_shifting:
                
            target_audio, audio = compute_time_shifting(target_audio,
                                                       audio,
                                                       batch_size=self.batch_size)

        if self.high_freq_emphasis:
            
            target_audio, audio = high_freq_pre_emphasis(target_audio, audio, 0.95)
                
        for loss_op in loss_ops:
            
            target_mag = loss_op(target_audio)
            value_mag = loss_op(audio)

          # Add magnitude loss.
            if self.mag_weight > 0:
                loss += self.mag_weight * mean_difference(target_mag, value_mag,
                                                      self.loss_type)

            if self.delta_time_weight > 0:
                target = diff(target_mag, axis=1)
                value = diff(value_mag, axis=1)
                loss += self.delta_time_weight * mean_difference(
                    target, value, self.loss_type)

            if self.delta_delta_time_weight > 0:
                target = diff(diff(target_mag, axis=1), axis=1)
                value = diff(diff(value_mag, axis=1), axis=1)
                loss += self.delta_delta_time_weight * mean_difference(
                    target, value, self.loss_type)

            if self.delta_freq_weight > 0:
                target = diff(target_mag, axis=2)
                value = diff(value_mag, axis=2)
                loss += self.delta_freq_weight * mean_difference(
                    target, value, self.loss_type)

            if self.delta_delta_freq_weight > 0:
                target = diff(diff(target_mag, axis=2), axis=2)
                value = diff(diff(value_mag, axis=2), axis=2)
                loss += self.delta_delta_freq_weight * mean_difference(
                    target, value, self.loss_type)

              # Add logmagnitude loss, reusing spectrogram.
            if self.logmag_weight > 0:
                target = safe_log(target_mag)
                value = safe_log(value_mag)
                loss += self.logmag_weight * mean_difference(target, value,
                                                             self.loss_type)

        if self.loudness_weight > 0:
            
            print(target_audio, audio)
            target = compute_loudness(target_audio, sample_rate=self.sr,
                                                    n_fft=self.fft_sizes[0])
            
            value = compute_loudness(audio, sample_rate=self.sr,
                                                  n_fft=self.fft_sizes[0])
            
            loss += self.loudness_weight * mean_difference(target, value,
                                                         self.loss_type)
        
        if self.time_loss:
                   
            if self.time_shifting:
            
                loss_plus = self.time_loss_weight * mean_difference(target_audio, audio,
                                                                 self.time_loss_type)
                loss_minus = self.time_loss_weight * mean_difference(target_audio, -1*audio,
                                                                 self.time_loss_type)
                
                loss += tf.math.minimum(loss_plus, loss_minus)
                
            else:

                loss += self.time_loss_weight * mean_difference(target_audio, audio,
                                                             self.time_loss_type)


        return loss
    
    
def multiScaleSpectralLoss(fft_sizes=(2048, 1024, 512, 256, 128, 64),
                   loss_type='L1',
                   mag_weight=1.0,
                   overlap=0.75,
                   delta_time_weight=0.0,
                   delta_delta_time_weight=0.0,
                   delta_freq_weight=0.0,
                   delta_delta_freq_weight=0.0,
                   logmag_weight=0.0,
                   loudness_weight=0.0, 
                   sr=22080,
                   time_loss=False,
                   time_loss_type='L1',
                   time_loss_weight=100.0,
                   time_shifting=False,
                   high_freq_emphasis=False,
                   batch_size=100,
                   name='spectral_loss'):
    
    spectral_loss = SpectralLoss(fft_sizes=fft_sizes,
                   loss_type=loss_type,
                   mag_weight=mag_weight,
                   overlap=overlap,
                   delta_time_weight=delta_time_weight,
                   delta_delta_time_weight=delta_delta_time_weight,
                   delta_freq_weight=delta_freq_weight,
                   delta_delta_freq_weight=delta_delta_freq_weight,
                   logmag_weight=logmag_weight,
                   loudness_weight=loudness_weight,
                   sr=sr,
                   time_loss=time_loss,
                   time_loss_type=time_loss_type,
                   time_loss_weight=time_loss_weight,
                   time_shifting=time_shifting,
                   high_freq_emphasis=high_freq_emphasis,
                   batch_size=batch_size,
                   name=name)

    def loss(y_true,y_pred):
        
        return spectral_loss(y_true, y_pred)
    
   # Return a function
    return loss


