import librosa
import numpy as np

from pypesq import pesq
from pystoi.stoi import stoi


def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(clean_signal, noisy_signal, sr)


def mag_and_phase(y, to_db=False):
    """
    Extract magnitude and phase of speech waveform.

    Args:
        y: waveform
        to_db: amplitude to dB

    Returns:
        (mag, phase)
    """
    length = len(y)
    librosa.util.fix_length(y, length + 512 // 2)
    mag, phase = librosa.magphase(librosa.stft(y, n_fft=512, hop_length=512 // 2, win_length=512))
    if to_db:
        mag = librosa.amplitude_to_db(mag)
    return mag, phase, length


def z_score_matrix(mag):
    mean = np.mean(mag, axis=1).reshape((mag.shape[1], 1))
    std = np.std(mag, axis=1).reshape((mag.shape[1], 1))
    mag = (mag - mean) / std

    return mag


def inverse_z_score_matrix(mag):
    pass