from data_objects.params_data import *
from pathlib import Path
from typing import Optional, Union
import numpy as np
import librosa

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):                            #source sampling rate
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr = None)                #Preserving the native sampling rate, wav will be np.ndarray(Audio-time series) and sampling rate is also returned. In this case, same SR is returned.
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)               # If the source SR and sampling_rate in params_data is not same, then resample it

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)

    return wav


def wav_to_spectrogram(wav):
    frames = np.abs(librosa.core.stft(
        wav,
        n_fft=n_fft,                                                #Frame Size
        hop_length=int(sampling_rate * window_step / 1000),                 #Hop length to overlap the frame
        win_length=int(sampling_rate * window_length / 1000),   #Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft
    ))
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
