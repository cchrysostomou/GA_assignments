"""
Small helper functions used to process the wav data files
"""

import soundfile as sf
from scipy.signal import butter, lfilter, freqz
from scipy import fftpack, signal
import numpy as np
import pandas as pd


def load_wav(wav_path):
    """
        Load a wav file into a numpy array

        Args:
            wav_path (string): location of the wav file

        Returns:
            sound_arr (numpy array): a 1D array of the sound file
            samplerate (int): The sample rate of the sound file
    """
    sound_arr, samplerate = sf.read(wav_path)

    return sound_arr, samplerate

    sample_file = 'sample_data/normal__201104141251.wav'

    sound_arr, samplerate = load_wav(sample_file)

    print('The sample rate is: ' + str(samplerate) +' frames/second')


def butter_lowpass(cutoff, fs, order=5):
    #   creating-lowpass-filter-in-scipy-understanding-methods-and-units
    # Taken from  https://stackoverflow.com/questions/25191620/
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_noise(signal, cutoff, samplerate):
    """
    Runs a low pass filter on a signal. High frequencies are defined as those >=cutoff

    Args:
        signal (numpy array): The signal file that needs to be filtered
        cutoff (int): A frequency value (in hertz) to treat as the high frequency cutoff
        samplerate (int): The framerate of the file
    """
    # Filter the data, and plot both the original and filtered signals.
    return butter_lowpass_filter(signal, cutoff, samplerate)


def get_fourier(signal, samplerate=1):
    """
    Convert a signal from the time domain to the frequency domain

    Args:
        signal (numpy array): array to convert
        samplerate (int): framerate of the signal

    Returns:
        four_series (pd.Series): A series of values. Index represents a frequency, values represents amplitude in each freq
    """
    tmp = signal[~np.isnan(signal)]
    fft_signal = fftpack.fftshift(fftpack.fft(tmp))
    fft_freqs = fftpack.fftshift(fftpack.fftfreq(tmp.shape[0], 1.0/samplerate))
    return pd.Series(fft_signal, index=fft_freqs)


def smooth_data(vals, window=50, lim=1000, normalize=True, cutoff=0.05):
    """
    Smooth data from noisy plots
    """
    tmp = vals.rolling(window=window).mean().iloc[list(np.arange(0, vals.shape[0], window))].dropna()
    if normalize:
        tmp = tmp/(1.0 * np.abs(tmp.values).max())
    tmp[tmp<cutoff] = 0
    return tmp.loc[0:lim]


def np_pad(arr, arr_len):
    """
        Add zeros to an array to force the length of an array to be arr_len
    """
    new_arr = np.zeros(arr_len)
    new_arr[:arr.shape[0]] = arr
    return new_arr

def get_local_frequency(signal, center_point, window, normalize=True, cutoff=0, samplerate=1):
    """
        Given a signal and an identified region of interest (center_point). Return the freuqency domain of the data within a
        specific window size centered around the center_point

        Args:
            signal (np array): signal of interest
            center_point (int): Position/coordinate within the signal that will be in the center of the window
            window (int): Size of the window/# of datapoints merged

        Returns:
            four_arr (pd.Series): Series of the fourier signal
    """
    tmp = signal[~np.isnan(signal)]
    # return the left most point in the window (cannot be smaller than 0)
    left_point = int(max(0, center_point - window*1.0/2))
    # return the right most point in the window (cannot be larger than array size)
    right_point = int(min(center_point + window*1.0/2, tmp.shape[0] - 1))
    # take the window slice in that region
    subset = tmp[range(left_point, right_point + 1)]

    if subset.shape[0] < window:
        # its too small...we want to make sure all signals are equal length
        subset = np_pad(subset, window)
    elif subset.shape[0] > window:
        subset = subset[:window]
    # calculate the frequencies (only take the first n/2 samples (the remainder are the same frequency but on the - scale (due to symmetry)))

    freqs = fftpack.fftfreq(subset.shape[0], 1.0/samplerate)[:subset.shape[0]/2]
    amplitude = np.abs(fftpack.fft(subset))[:subset.shape[0]/2]
    if normalize:
        # divid all amplitudes over the max amplitdude
        # amplitude = amplitude/amplitude.max()
        amplitude = MinMaxScaler().fit_transform(amplitude.reshape(-1, 1)).squeeze()
    amplitude[amplitude<cutoff] = 0
    return pd.Series(amplitude, index=freqs).fillna(0)