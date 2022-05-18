"""

Tools for reading and processing audio files and extracting pitches

"""

from itertools import product
import os
from pathlib import Path
import sys

import crepe
import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12
import pandas as pd
from pydub import AudioSegment
import seaborn as sns
from scipy.interpolate import CubicSpline
from scipy.signal import argrelmax, argrelmin, argrelextrema
from scipy.spatial import distance_matrix
from scipy.stats import lognorm, kstest, ks_2samp, linregress
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN
import soundfile as sf
import vamp

import wavfile


def convert_mp3_to_wav_folder(path):
    for f in path.glob('*mp3'):
        convert_mp3_to_wav(f)


def convert_mp3_to_wav(f):
    sound = AudioSegment.from_mp3(f)
    sound.export(os.path.splitext(f)[0] + '.wav', format='wav')


def read_wav(f, return_mono=True):
    fr, wav = wavfile.read(f)[:2]
    if len(wav.shape)>1:
        return fr, wav.mean(axis=1)
    else:
        return fr, wav


def load_mp3file(path):
    """MP3 to numpy array"""
    a = AudioSegment.from_mp3(path)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(axis=1)
    return a.frame_rate, y


def load_audio(path, stereo=False):
    try:
        ext = path.suffix
    except:
        ext = Path(path).suffix

    if ext[1:].upper() in sf.available_formats():
        wav, fr = sf.read(path)
    elif ext[1:].lower() == 'mp3':
        fr, wav = load_mp3file(str(path))
    else:
        raise Exception(f"File extension '{ext}' not recognized\n{path}")

    if stereo:
        return fr, wav
    if len(wav.shape)>1:
        return fr, wav.mean(axis=1)
    else:
        return fr, wav



###############################################################################
####  Published pitch-tracking algorithms:


def extract_pitch_from_wav(fr, wav, alg='pyin_smooth', fmin=40, fmax=8000, lowamp=0.1, hop=256, window=2048):
    # Often wav gets read as an int;
    # a lot of this code uses modules written in C, and I guess they
    # are not flexible regarding variable types;
    # this avoids those errors
    wav = wav.astype(float)

    # pYIN
    # Viterbi - smoothed pitch track
    if alg=='pyin_smooth':
        data = vamp.collect(wav, fr, "pyin:pyin", output='smoothedpitchtrack', parameters={'outputunvoiced': 2, 'lowampsuppression':lowamp})
        melody = data['vector'][1]
        tm = np.arange(len(melody)) * float(data['vector'][0])

    # pYIN?
    # No smoothing - just returns the most probable f0candidates
    # i.e., this is probably just YIN...
    elif alg=='pyin':
        f0 = vamp.collect(wav, fr, "pyin:pyin", output='f0candidates')
        prob = vamp.collect(wav, fr, "pyin:pyin", output='f0probs')
        tm, melody = [], []
        for f, p in zip(f0['list'], prob['list']):
            if 'values' in f.keys():
                freq = f['values'][np.argmax(p['values'])]
                if freq <= fmax:
#               if freq <= fmax and p['values'].max()>0.001:
                    tm.append(float(f['timestamp']))
                    melody.append(freq)
        tm = np.array(tm)
        melody = np.array(melody)

    # pYIN
    # Same as "pyin_smooth" but written in python instead of C
    # Gives slightly different results to "pyin_smooth"
    elif alg=='pypyin':
        pyin = PyinMain()
        pyin.initialise(inputSampleRate=fr, lowAmp=lowamp,
                        fmin=fmin, fmax=fmax)
        for i, frame in enumerate(frame_generator(wav, fr, window=window, hop=hop)):
            fs = pyin.process(frame)
        melody = pyin.getSmoothedPitchTrack()
        tm = np.arange(melody.size) * hop / fr

    # Returns note frequncies and onsets
    elif alg=='pyin_notes':
        data = vamp.collect(wav, fr, "pyin:pyin", output='notes')
        tm, melody = [], []
        for d in data['list']:
            tm.extend([d['timestamp'], d['timestamp']+d['duration']])
            melody.extend([d['values'][0]]*2)

    elif alg=='crepe':
        time, frequency, confidence, activation = crepe.predict(wav, fr)
        return time, frequency

    elif alg=='yin':
        p, h, a, t = yin.compute_yin(wav, fr, f0_min=fmin, f0_max=fmax, harmo_thresh=0.1, w_step=32, w_len=1024)
        p, t = np.array(p), np.array(t)
        melody = p[p>0]
        tm = t[p>0]

    elif alg=='melodia':
        data = vamp.collect(wav, fr, "mtg-melodia:melodia")
        melody = data['vector'][1][data['vector'][1]>0]
        tm = (np.arange(len(data['vector'][1])) * float(data['vector'][0]) + 8*128/44100)[data['vector'][1]>0]

    elif alg=='aubio':
        data = vamp.collect(wav, fr, "vamp-aubio:aubiopitch")
        tm = np.array([d['timestamp'] for d in data['list']])
        melody = np.array([d['values'][0] for d in data['list']])

    else:
        raise Exception('Incorrect algorithm name passed as argument')

    return tm, melody


