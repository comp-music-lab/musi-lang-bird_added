"""

General-purpose functions

"""
from collections import defaultdict
from itertools import product
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pydub import AudioSegment
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, set_link_color_palette
from scipy.interpolate import CubicSpline
from scipy.io import wavfile
from scipy.signal import argrelmax, argrelmin, argrelextrema
from scipy.spatial import distance_matrix
from scipy.spatial.distance import jensenshannon, pdist
from scipy.stats import lognorm, kstest, ks_2samp, linregress
from scipy.optimize import curve_fit
import statsmodels.nonparametric.api as smnp
import vamp


import audio_tools

N_PROC = 25



######################################################## 

def get_cents_from_ratio(r):
    return np.log2(r)*1200


def exp_gauss(x, a, x0, sigma, b):
    gauss = a*np.exp(-(x-x0)**2/(2*sigma**2))
    exp = np.exp(-abs(x-x0) / b) / (2.*b)
    return exp + gauss


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))


def gauss_fixed(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2)) / (sigma * (2*np.pi)**0.5)


def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def poly(x, a, b, c):
    return a * x**2 + b * x + c


def fit_gaussian(X, Y, sigma=20, norm=False):
    mean = np.mean(X)
    if norm:
        popt, pcov = curve_fit(gauss_fixed, X, Y, p0=[mean, sigma])
        return gauss_fixed(X, *popt), popt
    else:
        amp = np.max(Y)
        popt, pcov = curve_fit(gauss, X, Y, p0=[amp, mean, sigma])
        return gauss(X, *popt), popt


def fit_exp_gauss(X, Y, sigma=20, b=100):
    mean = np.mean(X)
    amp = np.max(Y)
    popt, pcov = curve_fit(exp_gauss, X, Y, p0=[amp, mean, sigma, b])
    return exp_gauss(X, *popt), popt


def fit_lognormal(X):
    mag = np.arange(0, max(X))
    param = lognorm.fit(X)
    return mag, lognorm.pdf(mag, param[0], loc=param[1], scale=param[2])


def fit_poly(X, Y):
    return curve_fit(poly, X, Y, p0=[1,1,1])

def distances(X):
    xi, yi = np.meshgrid(X, X)
    return np.abs(xi - yi)
    

def plot_fit(X, Y, coef):
    fig, ax = plt.subplots()
    ax.plot(X, Y, ':')
    for c in coef:
        G = gauss(X, *c)
        ax.plot(X, G)
    

def random_bird_1():
    counts = {1: 3, 2: 2, 3: 7, 4: 5, 5: 5, 6: 5, 7: 1, 8: 1}
    scales = []
    for c, v in counts.items():
        for i in range(v):
            ints = [int(np.random.normal(0, 5))]
            if c > 1:
                ints.extend(list(np.random.randint(1200, size=c)))
            scales.append(ints)
    return scales
            
def random_bird_2(coefs, cutoff=-100):
    idx = np.array([np.random.choice(len(coefs), size=len(coefs), replace=False) for i in range(3)]).T
    X = np.arange(-50, 1155, 5)
    Y = np.zeros(X.size, dtype=float)
    for i, j, k in idx:
        Y += gauss(X, coefs[i,0], coefs[j,1], coefs[k,2])
    return Y / Y.sum()
        

def ratio2cents(ratio):
    return np.log(ratio)/np.log(2)*1200


def smooth_dist_kde(X, hist=False):
    kde = smnp.KDEUnivariate(np.array(X, dtype=float))
    kde.fit(kernel='gau', bw='scott', fft=1, gridsize=10000, cut=20)
    grid = np.linspace(0, 1200, num=1201)
    y = np.array([kde.evaluate(x) for x in grid]).reshape(1201)
    if hist:    
        hist, edges = np.histogram(X, bins=grid)
        xxx = grid[:-1] + (grid[1] - grid[0]) * 0.5    
        return grid, y, xxx, hist
    else:
        return grid, y


def find_maximum_freq(freq):
    X, Y = smooth_dist_kde(freq)
    return X[np.argmax(Y)]


def autocorr(signal):
    signal = signal - signal.mean()
    return np.correlate(signal, signal, mode='full')


def int_to_bin_vec(integer, n=13):
    vec = np.zeros(n, dtype=bool)
    for i in range(n,0,-1):
        if integer == 0:
            return vec
        if integer >= 2**i:
            vec[i-1] = True
            integer -= 2**i
    return vec


def downsample_histogram(X, Y, dx):
    old_dx = np.diff(X[:2])
    if old_dx >= dx:
        print("Wrong bin size chosen")
        return
    binsize = int(dx/old_dx)
    steps = int(len(X) / binsize)
    Xnew = np.array([np.mean(X[i*binsize:(i+1)*binsize])for i in range(steps)])
    Ynew = np.array([np.mean(Y[i*binsize:(i+1)*binsize])for i in range(steps)])
    return Xnew, Ynew
        

def cluster_fn(li, i):
    try:
        return fcluster(li, li[-i,2], criterion='distance')
    except:
        return None


def cluster_inp_generator(li, idx):
    for i in idx:
        yield li, i


def cluster_dist_mat(dist, N):
    li = linkage(pdist(dist), 'ward')
    if N > 0:
        return fcluster(li, li[-N,2], criterion='distance')
    elif N == 0:
        with Pool(N_PROC) as pool:
            return np.array(list(pool.starmap(cluster_fn, cluster_inp_generator(li, range(2,len(dist)-1)), 5)))


def aligned_distance(s1, s2, alg='sum'):
    if alg=='sum':
        return np.sum([np.min(np.abs(s1-s)**2) for s in s2])**0.5
    elif alg=='mean':
        return np.mean([np.min(np.abs(s1-s)**2) for s in s2])**0.5
        

def scale_alignment(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    gap = abs(n1 - n2)
    if gap == 0:
        return np.mean(np.square(s2-s1))**0.5, gap
    else:
        small, large = [[s1, s2][fn([n1, n2])] for fn in [np.argmin, np.argmax]]
        return aligned_distance(large, small), gap


def resample(tm, freq, dx):
    Xgrid = np.round(np.arange(min(tm), max(tm)+dx, dx), int(-np.log10(dx) + 1))
    X, Y = [], []

    match = defaultdict(list)
    for t, f in zip(tm, freq):
        i = np.argmin(np.abs(Xgrid - t))
        match[Xgrid[i]].append(f)

    for k, v in match.items():
        X.append(k)
        Y.append(np.mean(v))

    return np.array(X), np.array(Y)


def join_notes(notes):
    duration = np.sum([n[0][1] - n[0][0] for n in notes])
    beg = notes[0][0][0]
    end = notes[-1][0][1]
    f0 = np.mean([n[1][0] for n in notes])
    return [[beg, end], [f0, f0], duration] 


def remove_unison_notes_from_seq(notes, cut=25):
    # Group adjacent notes together in a list
    # if they are within 'cut' cents of the mean frequency
    group = [[notes[0]]]
    for n in notes[1:]:
        prev_note = np.mean([x[1][0] for x in group[-1]])
        cents = abs(np.log2(n[1][0] / prev_note) * 1200)
        if cents > cut:
            group.append([n])
        else:
            group[-1].append(n)

    # Average out any unison notes to get a mean frequency
    # Note the start and end of the combined unison note
    # as the start of the first note and the end of the last
    # Include the actual sum of the invidual note durations
    # as the sum
    out = []
    for g in group:
        if len(g) == 1:
            duration = g[0][0][1] - g[0][0][0]
            out.append(list(g[0]) + [duration])
        else:
            out.append(join_notes(g))
    return out


def autocorrelation_function_1d(X, Y, bins):
    Y = (Y - np.mean(Y)) / np.std(Y)
    X_dist = distance_matrix(X.reshape(X.size,1), X.reshape(X.size,1))
    YY = np.outer(Y, Y)

    dx = np.diff(bins[:2])
    bin_idx = np.array(X_dist / dx, dtype=int)
    corr = np.zeros(bins.size, dtype=float)
    for i in range(bins.size):
        corr[i] = np.mean(YY[np.where(bin_idx==i)])

    return corr 


def acf_2(Y, N):
    acf = np.array([Y[i:i+N] * Y[i] for i in range(Y.size-N)])
    return np.nanmean(acf, axis=0)



def noteToFreq(note):
    a = 440 #frequency of A (common value is 440Hz)
    return (a / 32) * (2 ** ((note - 9) / 12))


def clip_and_save(path, fr, wav, beg, end, e1=0.5, e2=4.):
    ibeg = int(beg * fr)
    iend = int(end * fr)
    imid = int(ibeg + (iend - ibeg) / 3)
    envelope = np.ones(iend - ibeg, float)
    envelope[:imid-ibeg] = np.linspace(0, 1, imid-ibeg)**e1
    envelope[imid-ibeg:iend-ibeg] = 1 - np.linspace(0, 1, iend-imid)**e2
    clip = wav[ibeg:iend] * envelope
    wavfile.write(path, fr, clip)


def choose_excerpt(mel, fr, dur=2.):
    is_signal = mel > 0
    window = int(dur * fr)
    kernel = np.ones(window, float)
    start = np.argmax(np.convolve(is_signal, kernel, mode="valid"))
    return start / fr
    

def extract_clips(max_clips=10):
    clip_durations = [2, 1.5, 1, 0.75, 0.5]
    for i, root in enumerate(['WEL', 'QUE', 'ACO', 'ARA', 'HAD', 'JEN', 'MBE', 'MES']):
        for j, ID in enumerate('CD'):
            files = sorted(Path('../Data/IDS/IDS-corpus-raw/').glob(f"{root}*{ID}*wav"))
            for f in files[:max_clips]:
                fr, wav = audio_tools.load_audio(f)
                tm, mel = audio_tools.extract_pitch_from_wav(fr, wav)
                start = choose_excerpt(mel, fr)
                for dur in clip_durations:
                    path = f"../Data/IDS/Samples/{f.stem}_{dur:4.2f}.wav"
                    try:
                        clip_and_save(path, fr, wav, start, start + dur)
                    except Exception as e:
                        print(f, e)




        

