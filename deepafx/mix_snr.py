#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
import numpy as np
import scipy as sp


from . import loudness as loudness

def aweighting_filter(fs):
    """
    Computes an A-weighting filter for a given sampling rate

    :param fs: sampling rate
    :return: b, a - filter coefficients
    """

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    NUM = np.array([((2 * np.pi * f4) ** 2) * (10 ** (A1000 / 20)), 0.0, 0.0, 0.0, 0.0])
    DEN = np.convolve([1, +4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1.0, +4 * np.pi * f1, (2.0 * np.pi * f1) ** 2.0])
    DEN = np.convolve(np.convolve(DEN, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]);

    [b, a] = sp.signal.bilinear(NUM, DEN, fs)
    return b, a

def rms(x):
    """
    Computes the room mean square energy level

    :param x: input audio signal
    :return: root mean square energy value
    """
    return np.sqrt(np.mean(np.square(x)))

def rmsa(x, fs, aweighting=True):
    """
    Compute the A-weighted root mean square of an input signal

    :param x: input signal, typically diffuse noise
    :param fs: sampling rate of the input signal
    :param aweighting: flag to decide if the A-weighting filter is applied
    :return: rms_level - root mean square amplitude level
    """

    if aweighting:
        [b, a] = aweighting_filter(fs)
        y = sp.signal.lfilter(b, a, x)
    else:
        y = x

    return rms(y)

def p56(x, fs):
    """
    Compute the active speech level with the ITU P.56 method B.

    This is very slow. TODO: optimize
    This is a Python port of the Loizou Noise Suppression book CD-ROM asl.

    :param x: input speech signal
    :param fs: sampling rate of the speech signal
    :return: Active speech level root mean square (amplitude)
    """

    def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):
        """
        Interpolate a bin value
        :param upcount:
        :param lwcount:
        :param upthr:
        :param lwthr:
        :param Margin:
        :param tol:
        :return:
        """
        tol = np.abs(tol)
        asl_ms_log = 0
        cc = 0

        # Check if extreme counts are not already the true active value
        iterno = 1
        if np.abs(upcount - upthr - Margin) < tol:
            asl_ms_log = upcount
            cc = upcount
            return asl_ms_log, cc

        if np.abs(lwcount - lwthr - Margin) < tol:
            asl_ms_log = lwcount
            cc = lwthr
            return asl_ms_log, cc

        # Intialize first middle for given (initial) bounds
        midcount = (upcount + lwcount) / 2.0
        midthr = (upthr + lwthr) / 2.0

        while True:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break

            # if tolerance is not met up to 20 iterations, then relax the tolerance by 10%
            iterno = iterno + 1

            if iterno > 20:
                tol = tol * 1.1

            if diff > tol:
                midcount = (upcount + midcount) / 2.0
                midthr = (upthr + midthr) / 2.0
            elif diff < -tol:
                midcount = (midcount + lwcount) / 2.0
                midthr = (midthr + lwthr) / 2.0

        asl_ms_log = midcount
        cc = midthr
        return asl_ms_log, cc

    eps = 2.220446049250313e-16
    nbits = 16  # assumed bit depth in the audio signal

    assert(len(x.shape) == 1)  # Make sure the signal is a vector

    T = 0.03  # time constant of smoothing, in seconds
    H = 0.2  # hangover time in seconds
    M = 15.9
    thres_no = nbits - 1    # margin in dB, number of thresholds
    I = np.ceil(fs * H)     # hangover in samples
    g = np.exp(-1.0 / (fs * T))  # smoothing factor in envelop detection
    c = 2. ** np.linspace(-15, thres_no- 16, thres_no)
    a = np.zeros(thres_no)
    hang = I*np.ones(thres_no)

    sq = np.sum(np.square(x))
    x_len = len(x)
    x_abs = np.abs(x)
    p = sp.signal.lfilter([1-g], [1, -g], x_abs)
    q = sp.signal.lfilter([1-g], [1, -g], p)

    for k in range(x_len):
        for j in range(thres_no):
            if q[k] >= c[j]:
                a[j] = a[j] + 1
                hang[j] = 0
            elif hang[j] < I:
                a[j] = a[j] + 1
                hang[j] = hang[j] + 1
            else:
                break

    # asl = 0
    asl_rms = eps
    if a[0] == 0:
        return asl_rms
    else:
        AdB1 = 10 * np.log10(sq / a[0] + eps)


    CdB1 = 20*np.log10(c[0] + eps)
    if AdB1 - CdB1 < M:
        return asl_rms

    AdB = np.zeros(thres_no)
    CdB = np.zeros(thres_no)
    Delta = np.zeros(thres_no)
    AdB[0] = AdB1
    CdB[0] = CdB1
    Delta[0] = AdB1 - CdB1

    for j in range(1, thres_no):
        AdB[j] = 10 * np.log10(sq/(a[j] + eps) + eps)
        CdB[j] = 20 * np.log10(c[j] + eps)

    for j in range(1, thres_no):
        if a[j] != 0:
            Delta[j] = AdB[j] - CdB[j]
            if Delta[j] <= M:
                # interpolate to find the asl
                asl_ms_log, cl0 = bin_interp(AdB[j], AdB[j-1], CdB[j], CdB[j-1], M, 0.5)
                asl_ms = 10**(asl_ms_log/10.0)
                # asl = (sq / x_len) / asl_ms
                asl_rms = np.sqrt(asl_ms)
                #c0 = 10.0**( cl0 / 20)
                break

    return asl_rms

def match_length(x, desired_len_samples, fs):
    """
    Match the length of the noise signal to be the same as the speech signal.
    If the noise signal is shorter than the speech signal, the noise signal is repeated with a crossfade.

    :param x:
    :param desired_len_samples:
    :param fs:
    :return:
    """
    noise = x
    noise_len = len(noise)
    speech_len = desired_len_samples
    if noise_len < speech_len:

        noise_xfade = np.copy(noise)

        window_len_samples  = int(.01*fs)
        window_len_samples_half = int(window_len_samples/2)
        window = sp.signal.hann(window_len_samples, sym=False)
        fade_in = window[0:window_len_samples_half]
        fade_out = window[window_len_samples_half:]

        # Create version of noise signal with fade in and out
        noise_xfade[0:window_len_samples_half] = noise_xfade[0:window_len_samples_half]*fade_in
        noise_xfade[-window_len_samples_half:] = noise_xfade[-window_len_samples_half:]*fade_out
        factor = int(np.ceil((speech_len - window_len_samples_half)/(noise_len - window_len_samples_half)))
        noise_new = np.zeros(noise_len * factor)

        # Overlap-add the noise signal to extend
        left = 0
        right = noise_len
        for i in range(factor):
            noise_new[left:right] = noise_new[left:right] + noise_xfade
            left = left + noise_len - window_len_samples_half
            right = right + noise_len - window_len_samples_half

        # Cut to speech length
        noise_new = noise_new[0:speech_len]

        # Undo fade in at the start
        noise_new[:window_len_samples_half] = noise[:window_len_samples_half]

    # cut the noise short
    else:
        noise_new = noise[0:speech_len]

    return noise_new

def measure_level(x, fs, weighting='p56'):
    """
    Measure the level of an audio signal using various weighting schemes.

    :param x: input audio signal
    :param fs: sampling rate
    :param weighting: weighting scheme
    :param fixed_level_db: fixed
    :return: active (speech) level in linear energy units (mean-square)
    """
    if weighting == 'p56':
        asl_ms = p56(x, fs) ** 2
    elif weighting == 'rms':
        asl_ms = rms(x) ** 2
    elif weighting == 'rmsa':
        asl_ms = rmsa(x, fs, aweighting=True) ** 2
    elif weighting == 'lufs':
        integrated_level, threshold = loudness.calculate_loudness(x, fs)
        asl_ms = (10 ** (integrated_level / 20)) ** 2
    else:
        assert 0, 'Unknown speech weighting'

    # TODO: change this to output dB level?
    return asl_ms

def scale_to_desired_level(x, fs, desired_level_db, weighting='p56'):
    """
    Scale an audio signal to a desired level using a given weighting scheme

    :param x: input audio signal
    :param fs: sampling rate
    :param desired_level_db: desired level of the signal in decibels
    :param weighting: weighting scheme (See measure_level)
    :return:
    """

    asl_ms = measure_level(x, fs, weighting=weighting)

    als_ms_desired = (10 ** (desired_level_db / 20)) ** 2
    scale = np.sqrt(als_ms_desired / asl_ms)
    asl_ms = als_ms_desired

    y = x * scale

    level_db = 10*np.log10(asl_ms)

    return y, scale, level_db

def mix_speech_and_noise(speech, noise, snr_db, fs,
                         rescale_speech=False,
                         desired_speech_asl_db=-30.0,
                         speech_weighting='p56',
                         fixed_speech_level_db=-30,
                         noise_weighting='rmsa',
                         fixed_noise_level_db=-30,
                         quick=False):
    """
    Mixes speech and noise signals of the same length, given the required SNR

    :param speech: input speech signal
    :param noise: input noise signal
    :param snr_db: desired SNR dB
    :param fs: input audio sampling rate
    :param rescale_speech:  flag to allow rescaling the speech level
    :param desired_speech_asl_db: if rescaling speech, set the speech to a specific level
    :param speech_weighting: weighting scheme used to measure speech
    :param fixed_speech_level_db: if the speech weighting is fixed, assume the user provides the level in dB
    :param noise_weighting: weighting scheme used to measure noise
    :param fixed_noise_level_db: if the noise weighting is fixed, assume the user provides the level in dB
    :param quick: Boolean to measure level only on the first 10 seconds or less
    :return:
    """


    speech_quick = speech
    if quick:
        speech_quick = speech[0:min([int(fs*10), len(speech)-1])]

    # Measure the speech level
    if speech_weighting == 'fixed':
        asl_ms = (10 ** (fixed_speech_level_db / 20)) ** 2
    else:
        asl_ms = measure_level(speech_quick, fs, weighting=speech_weighting)

    speech_scale = 1.0
    if rescale_speech:
        als_ms_desired = (10 ** (desired_speech_asl_db / 20)) ** 2
        speech_scale = np.sqrt(als_ms_desired/asl_ms)
        asl_ms = als_ms_desired

    # Measure the noise
    if noise_weighting == 'fixed':
        noise_energy = (10 ** (fixed_noise_level_db / 20)) ** 2
    else:
        noise_energy = measure_level(noise, fs, weighting=noise_weighting)

    snr_scale = 10.0**(-snr_db/20.0)
    noise_scale = np.sqrt(asl_ms/noise_energy)*snr_scale
    scaled_speech = speech_scale*speech
    scaled_noise = noise_scale*noise
    mix = scaled_speech + scaled_noise

    return mix, scaled_speech, scaled_noise, speech_scale, noise_scale
