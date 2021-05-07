"""
MIT License

Copyright (c) 2018 Brecht De Man

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Python implementation of sample rate independent, integrated loudness measurement according to EBU R128 / ITU-R BS.1770.

When using this in part or full, including modified versions, please consider acknowledging the author by citing

Brecht De Man, "Evaluation of Implementations of the EBU R128 Loudness Measurement,"
145th International Convention of the Audio Engineering Society, October 2018.
This paper also provides background behind this implementation, particularly the filter coefficients.
"""

import numpy as np
import scipy as sp
import scipy.signal
import warnings
import copy


def calculate_loudness(signal, fs, prefiltered=False, gated=True):


    def K_filter(signal, fs, debug=False):

        # pre-filter 1
        f0 = 1681.9744509555319
        G = 3.99984385397
        Q = 0.7071752369554193
        # TODO: precompute
        K = np.tan(np.pi * f0 / fs)
        Vh = np.power(10.0, G / 20.0)
        Vb = np.power(Vh, 0.499666774155)
        a0_ = 1.0 + K / Q + K * K
        b0 = (Vh + Vb * K / Q + K * K) / a0_
        b1 = 2.0 * (K * K - Vh) / a0_
        b2 = (Vh - Vb * K / Q + K * K) / a0_
        a0 = 1.0
        a1 = 2.0 * (K * K - 1.0) / a0_
        a2 = (1.0 - K / Q + K * K) / a0_
        signal_1 = sp.signal.lfilter([b0, b1, b2], [a0, a1, a2], signal)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 9))
            # ax1 = fig.add_subplot(111)
            w, h1 = sp.signal.freqz([b0, b1, b2], [a0, a1, a2], worN=8000)  # np.logspace(-4, 3, 2000))
            plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h1)))
            plt.title('Pre-filter 1')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Gain [dB]')
            plt.xlim([20, 20000])
            plt.ylim([-10, 10])
            plt.grid(True, which='both')
            ax = plt.axes()
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            plt.show()

        # pre-filter 2
        f0 = 38.13547087613982
        Q = 0.5003270373253953
        K = np.tan(np.pi * f0 / fs)
        a0 = 1.0
        a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
        a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
        b0 = 1.0
        b1 = -2.0
        b2 = 1.0
        signal_2 = sp.signal.lfilter([b0, b1, b2], [a0, a1, a2], signal_1)

        if debug:
            plt.figure(figsize=(9, 9))
            # ax1 = fig.add_subplot(111)
            w, h2 = sp.signal.freqz([b0, b1, b2], [a0, a1, a2], worN=8000)
            plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h2)))
            plt.title('Pre-filter 2')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Gain [dB]')
            plt.xlim([10, 20000])
            plt.ylim([-30, 5])
            plt.grid(True, which='both')
            ax = plt.axes()
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.show()

        return signal_2  # return signal passed through 2 pre-filters



    G = [1.0, 1.0, 1.0, 1.41, 1.41]

    if len(signal.shape) == 1:  # if shape (N,), then make (N,1)
        signal = signal.reshape((signal.shape[0], 1))


    # filter or not
    if prefiltered:
        if len(signal.shape) == 1:  # if shape (N,), then make (N,1)
            signal_filtered = copy.copy(signal.reshape((signal.shape[0], 1)))
        else:
            signal_filtered = copy.copy(signal)

        for i in range(signal_filtered.shape[1]):
            signal_filtered[:, i] = K_filter(signal_filtered[:, i], fs)
    else:
        signal_filtered = signal

    # mean square
    T_g = 0.400  # 400 ms gating block
    Gamma_a = -70.0  # absolute threshold: -70 LKFS
    overlap = .75  # relative overlap (0.0-1.0)
    step = 1 - overlap

    T = signal_filtered.shape[0] / fs  # length of measurement interval in seconds
    j_range = np.arange(0, (T - T_g) / (T_g * step)).astype(int)
    z = np.ndarray(shape=(signal_filtered.shape[1], len(j_range)))

    # write in explicit for-loops for readability and translatability
    for i in range(signal_filtered.shape[1]):  # for each channel i
        for j in j_range:  # for each window j
            lbound = np.round(fs * T_g * j * step).astype(int)
            hbound = np.round(fs * T_g * (j * step + 1)).astype(int)
            z[i, j] = (1 / (T_g * fs)) * np.sum(np.square(signal_filtered[lbound:hbound, i]))

    G_current = np.array(G[:signal_filtered.shape[1]])  # discard weighting coefficients G_i unused channels
    n_channels = G_current.shape[0]
    l = [-.691 + 10.0 * np.log10(np.sum([G_current[i] * z[i, j.astype(int)] for i in range(n_channels)])) \
         for j in j_range]

    if gated:
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # throw out anything below absolute threshold:
            indices_gated = [idx for idx, el in enumerate(l) if el > Gamma_a]
            z_avg = [np.mean([z[i, j] for j in indices_gated]) for i in range(n_channels)]
            Gamma_r = -.691 + 10.0 * np.log10(np.sum([G_current[i] * z_avg[i] for i in range(n_channels)])) - 10.0

            # throw out anything below relative threshold:
            indices_gated = [idx for idx, el in enumerate(l) if el > Gamma_r]
            z_avg = [np.mean([z[i, j] for j in indices_gated]) for i in range(n_channels)]
            L_KG = -.691 + 10.0 * np.log10(np.sum([G_current[i] * z_avg[i] for i in range(n_channels)]))
    else:
        Gamma_a = -np.Inf
        Gamma_r = -np.Inf

        indices_gated = [idx for idx, el in enumerate(l) if el > Gamma_r]
        z_avg = [np.mean([z[i, j] for j in indices_gated]) for i in range(n_channels)]
        L_KG = -.691 + 10.0 * np.log10(np.sum([G_current[i] * z_avg[i] for i in range(n_channels)]))


    return L_KG, max(Gamma_r, Gamma_a)


def generate_K_filter_coefficients(fs, debug=False):

    """ Implementation of sample rate independent, integrated loudness measurement via EBU R128 / ITU-R BS.1770.

    :param fs: sampling rate
    :param debug: True of False
    :return: filter coefficients
    """


    # pre-filter 1
    f0 = 1681.9744509555319
    G = 3.99984385397
    Q = 0.7071752369554193
    K = np.tan(np.pi * f0 / fs)
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.499666774155)
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K - Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_

    pre_filter1 = [[b0, b1, b2], [a0, a1, a2]]

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 9))
        w, h1 = sp.signal.freqz([b0, b1, b2], [a0, a1, a2], worN=8000)  # np.logspace(-4, 3, 2000))
        plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h1)))
        plt.title('Pre-filter 1')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim([20, 20000])
        plt.ylim([-10, 10])
        plt.grid(True, which='both')
        plt.show()

    # pre-filter 2
    f0 = 38.13547087613982
    Q = 0.5003270373253953
    K = np.tan(np.pi * f0 / fs)
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0

    pre_filter2 = [[b0, b1, b2], [a0, a1, a2]]

    if debug:
        plt.figure(figsize=(9, 9))
        w, h2 = sp.signal.freqz([b0, b1, b2], [a0, a1, a2], worN=8000)
        plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h2)))
        plt.title('Pre-filter 2')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim([10, 20000])
        plt.ylim([-30, 5])
        plt.grid(True, which='both')
        plt.show()

    return [pre_filter1, pre_filter2]


def lufs_loudness(signal, fs):

    error = False

    # Compute offline integrated loudness and peak loudness
    LUFS_speech_level, LUFS_threshold = calculate_loudness(signal, fs, prefiltered=False)

    # Compute peak
    integrated_peak = 20 * np.log10(np.max(np.abs(signal)))


    if np.isnan(LUFS_speech_level) or np.isinf(LUFS_speech_level):
        LUFS_speech_level = 10 * np.log10(1e-32 + np.mean(np.abs(signal[:]) ** 2))
        if np.isinf(LUFS_speech_level) or np.isnan(LUFS_speech_level):
            error = True
            LUFS_speech_level = 0
            integrated_peak = 1

    return LUFS_speech_level, integrated_peak, error


def rms_loudness(signal, fs):
    simple_loudness = np.reshape(10 * np.log10(np.mean(np.square(signal))), (1, 1))
    return float(simple_loudness)

def peak_loudness(signal):
    integrated_peak = 20 * np.log10(np.max(np.abs(signal)))
    return float(integrated_peak)
