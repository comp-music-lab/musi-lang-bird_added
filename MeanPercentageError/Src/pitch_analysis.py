from itertools import product
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit, cuda
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set2_8
import pandas as pd
import seaborn as sns
from scipy.interpolate import CubicSpline
from scipy.io import wavfile
from scipy.signal import argrelmax, argrelmin, argrelextrema
from scipy.spatial import distance_matrix
from scipy.spatial.distance import jensenshannon, cdist, pdist
from scipy.stats import lognorm, kstest, ks_2samp, linregress, pearsonr, entropy
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture


import audio_tools
import bss_utils as utils


N_PROC = 20


##########################################################
### Pitch segmenation 


def segment_signal(X, Y, xcut=0.07, ycut=0.05):
    x, y = [], []
    idx = [0]
    for i in range(1, len(X)):
        if X[i] - X[i-1] < xcut and abs(Y[i] - Y[i-1])/Y[i] < ycut:
            idx.append(i)
        else:
            x.append(X[idx])
            y.append(Y[idx])
            idx = [i]
    if len(idx):
        x.append(X[idx])
        y.append(Y[idx])
    return x, y


def segment_signal_by_spline(X, Y, xcut=0.07, ycut=0.05, dycut=500, mcut=0.05):
    x, y = [], []
    cs = CubicSpline(X, Y)
    Y = cs(X)
    dY = cs(X, 1)
    idx = [0]
    for i in range(1, len(X)):
        if X[i] - X[i-1] < xcut and abs(Y[i] - Y[i-1])/Y[i] < ycut and dY[i] < dycut:
            idx.append(i)
        else:
            if not len(x):
                x.append(X[idx])
                y.append(Y[idx])
            else:
                if (abs(np.mean(y[-1]) - np.mean(Y[idx]))/np.mean(y[-1]) < mcut) and (X[idx[0]] - x[-1][-1] < xcut):
                    x[-1] = np.append(x[-1], X[idx])
                    y[-1] = np.append(y[-1], Y[idx])
                else:
                    x.append(X[idx])
                    y.append(Y[idx])
            idx = [i]
    if len(idx):
        x.append(X[idx])
        y.append(Y[idx])
    return x, y



##########################################################
### Pitch-discreteness


def analyse_note_curvature(X, Y, name='none', deg=2, xcut=0.07, ycut=0.05, dycut=50, mcut=0.05, spline=False, ax='', plot=False):
    if plot or name!='none':
        cols = np.array(Set2_8.hex_colors)[:8]
        if isinstance(ax, str):
            fig, ax = plt.subplots()
        ax.plot(X, Y, 'o', c='k', alpha=1.0, ms=5)

    curves = []

    if spline:
        X, Y = segment_signal_by_spline(X, Y, xcut=xcut, ycut=ycut, dycut=dycut, mcut=mcut)
    else:
        X, Y = segment_signal(X, Y, xcut=xcut, ycut=ycut)
    for i, (x, y) in enumerate(zip(X, Y)):
        if len(x) < 3:
            continue
        z = np.polyfit(x, y, deg)
        y2 = np.poly1d(z)(x)
        if plot or name!='none':
            ax.plot(x, y, 'o', c=cols[i%8], alpha=1.0, ms=3)

        imin = np.argmin(y2)
        imax = np.argmax(y2)
        d2 = np.polyder(np.poly1d(z), 2)[0]
        deriv = (y2[-1] - y2[0]) / (x[-1] - x[0])

        if 0 < imin < len(y2)-1:
            extreme = y2[imin]
        elif 0 < imax < len(y2)-1:
            extreme = y2[imax]
        else:
            extreme = 0
        log_range = np.log(max(y2)/min(y2))/np.log(2)*1200
        curves.append([y2[0], y2[-1], extreme, np.mean(y2), max(y2) - min(y2), log_range, x[-1]-x[0], deriv, d2])

    out = pd.DataFrame(columns=['start', 'end', 'extreme', 'mean', 'range', 'log_range', 'dur', 'df', 'd2f'], data=curves)
    out['ex_diff'] = np.min([np.abs(out['extreme']-out[s]) for s in ['start', 'end']], axis=0)
    out.loc[out.extreme==0, 'ex_diff'] = 0

    if plot or name!='none':
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Frequency / Hz')
        if name != 'none':
            fig.savefig(PATH_FIG.joinpath('Segment', f"{name}.pdf"))
            fig.savefig(PATH_FIG.joinpath('Segment', f"{name}.png"))
            fig.close()

    return out


def fit_error(X, Y, deg, name='none', xcut=0.07, ycut=0.05, dycut=50, mcut=0.05, spline=True, err_typ='err'):
    if spline:
        X, Y = segment_signal_by_spline(X, Y, xcut=xcut, ycut=ycut, dycut=dycut, mcut=mcut)
    else:
        X, Y = segment_signal(X, Y, xcut=xcut, ycut=ycut)
    err = 0.0
    err = []
    yold, ynew = [], []
    for x, y in zip(X, Y):
        if len(x) < 3:
            continue
        z = np.polyfit(x, y, deg)
        p = np.poly1d(z)
        if err_typ=='err':
            err.extend([abs(py-ry)/ry for py, ry in zip(p(x), y)])
        elif err_typ=='corr':
            yold.extend(list(y))
            ynew.extend(list(p(x)))
    if err_typ=='err':
        return np.mean(err)
    elif err_typ=='corr':
        corr = pearsonr(yold, ynew)[0]
        return corr


def error_scaling(speech, vocal, bird, ycut=0.05, dycut=500):
    err = np.zeros(3*10)
    for i, (dat, deg) in enumerate(product([speech, vocal, bird], range(10))):
        err[i] = fit_error(dat[:,0], dat[:,1], deg, ycut=ycut, dycut=dycut)
    return err.reshape(3,10)


def plot_error_scaling(err, ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    for i, l in enumerate(['speech', 'vocal', 'bird']):
        ax.plot(range(err.shape[1]), err[i] / err[i].min(), label=l)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(range(10))
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('Normalised fitting error')


def error_scaling_params(speech, vocal, bird):
    fig, ax = plt.subplots(4,4, figsize=(18,12))
    fig.subplots_adjust(wspace=0.3)
    ycut = [5, 10, 20, 50]
    dycut = [10, 20, 50, 100]
    for i, j in product(range(4), range(4)):
        plot_error_scaling(error_scaling(speech, vocal, bird, ycut=ycut[i], dycut=dycut[j]), ax=ax[i,j])
    for i in range(4):
        ax[0,i].annotate(f"$\\Delta dy$ cutoff = {dycut[i]}", (0.3, 1.15),  xycoords='axes fraction')
        ax[i,0].annotate(f"$\\Delta y$ cutoff = {ycut[i]}", (-0.4, 0.3),  xycoords='axes fraction', rotation=90)

    fig.savefig(f"poly_fit_error.pdf", bbox_inches='tight')


def view_fitting_params(data, name=''):
    fig, ax = plt.subplots(4,4, figsize=(18,12))
    fig.subplots_adjust(wspace=0.3)
    ycut = [5, 10, 20, 50]
    dycut = [10, 20, 50, 100]
    for i, j in product(range(4), range(4)):
        analyse_note_curvature(data[:,0], data[:,1], ycut=ycut[i], dycut=[j], spline=True, ax=ax[i,j])
    for i in range(4):
        ax[0,i].annotate(f"$\\Delta dy$ cutoff = {dycut[i]}", (0.3, 1.15),  xycoords='axes fraction')
        ax[i,0].annotate(f"$\\Delta y$ cutoff = {ycut[i]}", (-0.3, 0.3),  xycoords='axes fraction', rotation=90)

    if name:    
        fig.savefig(f"{name}_fit_params.pdf", bbox_inches='tight')


def error_scaling_params_2(speech, vocal, bird):
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.3)
    ycut = [10, 20]
    lbls = ['Speech', 'Vocal', 'Bird']
    for i in range(2):
        plot_error_scaling(error_scaling(speech, vocal, bird, ycut=ycut[i]), ax=ax[i])
    for i in range(2):
        ax[i].set_title(f"$\\Delta y$ cutoff = {ycut[i]}")
        ax[i].set_ylim(0,230)

    fig.savefig(f"poly_fit_error.pdf", bbox_inches='tight')


def view_fitting_params_2(speech, vocal, bird, name=''):
    fig, ax = plt.subplots(3,2, figsize=(15,9))
    fig.subplots_adjust(wspace=0.15)
    ycut = [10, 20]
    lbls = ['Speech', 'Vocal', 'Bird']
    ft = 12
    for i, data in enumerate([speech, vocal, bird]):
        for j in range(2):
            analyse_note_curvature(data[:,0], data[:,1], ycut=ycut[j], spline=True, ax=ax[i,j])
        ax[i,0].annotate(lbls[i], (-0.15, 0.4),  xycoords='axes fraction', rotation=90, fontsize=ft)

    for i in range(2):
        ax[0,i].annotate(f"$\\Delta y$ cutoff = {ycut[i]}", (0.4, 1.15),  xycoords='axes fraction', fontsize=ft)

    fig.savefig(f"poly_fits.pdf", bbox_inches='tight')


def extract_ints(df, start='mean', end='mean'):
    return [utils.ratio2cents(df.loc[i+1, end] / df.loc[i, start]) for i in range(len(df)-1)]


def bird_song_intervals(spline=False, start='mean', end='mean', plot=False):
    files = PATH_HMN.glob("*wav")
    ints = []
    for i, f in enumerate(files):
        print(f)
        fr, wav = audio_tools.read_wav(f)
        tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float))
        if not len(mel):
            continue
        if plot:
            df = analyse_note_curvature(tm, mel, spline=spline, name=f"bird_{i:02d}")
        else:
            df = analyse_note_curvature(tm, mel, spline=spline)
        ints.extend(extract_ints(df, start=start, end=end))
    return ints


def error_scaling_new(X, Y, ycut=0.05, dycut=0.2, name='', kwargs=''):
    err = np.zeros(10)
    for i, deg in enumerate(range(10)):
        if isinstance(kwargs, str):
            err[i] = fit_error(X, Y, deg, ycut=ycut, dycut=dycut, name=name)
        else:
            err[i] = fit_error(X, Y, deg, **kwargs)
#       err[i] = 2*(i+1) - 2 * np.log(fit_error(X, Y, deg, ycut=ycut, dycut=dycut, err_typ='corr'))
    return err


def plot_fitting_file(f, spline=True, alg='pyin_smooth'):
    fr, wav = audio_tools.read_wav(f)
    tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg=alg)
    tm, mel = tm[mel>0], mel[mel>0]
    df = analyse_note_curvature(tm, mel, spline=spline, plot=True)


def get_error_file(f, name='', kwargs=''):
    print(f)
    print(name)
    fr, wav = audio_tools.read_wav(f)
    tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg='pyin_smooth')
    tm, mel = tm[mel>0], mel[mel>0]
    if len(name):
        try:
            analyse_note_curvature(tm, mel, name=name, spline=False, plot=True)
        except Exception as e:
            print(e)
            print(f)
    if not len(mel):
        return []
    return error_scaling_new(tm, mel, name=name, kwargs=kwargs)


def plot_error_scaling_new():
    fig, ax = plt.subplots()
    lbls = ['Bird', 'Vocal', 'Speech', 'HapBir', 'Inst']
    cols = np.array(Paired_12.hex_colors)[[5,9,1,7,3]]
    output = {l:[] for l in lbls}
    for i, path in enumerate([PATH_BIRD_DISC, PATH_HMN, PATH_SPE, PATH_HPB, PATH_INST]):
#       err_list = []
        for j, f in enumerate(path.glob("*wav")):
            name = f"{lbls[i]}_{f.stem}"
            name = ''
            err = get_error_file(f, name=name)
            if not len(err):
                continue
            output[lbls[i]].append((f, err / err.min()))
#           err_list.append(err)
            if not j:
                ax.plot(range(err.size), err / err.min(), color=cols[i], label=lbls[i], alpha=.4)
#               ax.plot(range(err.size), err, color=cols[i], label=lbls[i])
            else:
                ax.plot(range(err.size), err / err.min(), color=cols[i], alpha=.4)
#               ax.plot(range(err.size), err, color=cols[i])
#       err = np.mean(np.array(err_list), axis=0)
        print(err.shape)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(range(10))
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('Normalised fitting error')
#   ax.set_xscale('log')
    ax.set_yscale('log')

    return output


def bird_freq_ratio():
    fig, ax = plt.subplots(5,4)
    ax = ax.reshape(ax.size)
    for j, f in enumerate(PATH_BIRD_DISC.glob("*wav")):
        fr, wav = audio_tools.read_wav(f)
        tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float))
        if not len(tm):
            continue
        sns.distplot(mel/mel[0], ax=ax[j])


def get_pitch_histogram(freq, redu=False, alg='mode', norm=False, bin_shift=0):
    if not len(freq):
        return []

    # Find most common frequency
    if alg=='mode':
        denom = utils.find_maximum_freq(freq)
    elif alg=='min':
        denom = freq.min()

    # Convert to cents using the most common frequency
    cents = np.log2(freq/denom) * 1200

    # Get cents histogram
    if redu:
        # Collapse the histogram onto an octave
        # Use bin_shift to change the starting point of the histogram
        # if you want it to go from e.g., -50 to 1150 rather than 0 to 1200
        cents = np.array([(x+6000-bin_shift)%1200 for x in cents])
        dx = 5
        bins = np.arange(0, 1200+dx, dx)
        X = bins[:-1] + 0.5 * np.diff(bins[:2]) + bin_shift
    else:
        bins = np.linspace(cents.min(), cents.max(), 100)
        print(f"Range = {cents.max() - cents.min()}")
        dx = np.diff(bins[:2])
        X = bins[:-1] + 0.5 * np.diff(bins[:2])
    hist = np.histogram(cents, bins=bins)[0]

    # Normalise histogram
    if norm:
        hist = hist / hist.sum()


    # Shift Y values so that the maximum is found at 0 cents
    i = np.argmax(hist)
    if bin_shift != 0:
        i -= int(bin_shift/dx)
    hist = np.roll(hist, -i)

    return X, hist


def pitch_class_histogram_all_samp():
    df = pd.DataFrame(columns=['path', 'time', 'freq'])
    for root, dirs, files in os.walk(PATH_DATA1):
        for f in files:
            if f[-4:] == '.txt':
                path = os.path.join(root, f)
                data = np.loadtxt(path).T
                X, Y = data[:2]
#               X, Y, Z = np.loadtxt(path).T
#               df.loc[len(df)] = [path, X[Y>0], Y[Y>0], Z[Y>0]]
                df.loc[len(df)] = [path, X[Y>0], Y[Y>0]]
    return df


def extract_scale_from_pitch_trace(k, v, max_points=0, min_int=80, amp_cut=0.001):
    try:
        print(k)
        if not max_points:
            max_points = len(v[1])
        X, hist = get_pitch_histogram((v[1][v[1]>0])[:max_points], redu=True, norm=True)
        return extract_scale(X, hist, min_int=min_int, amp_cut=amp_cut)
    except Exception as e:
        print(f"{k}\n{e}")
        return None


def ex_sc_inp_generator(data, max_points=5000, min_int=80, amp_cut=0.001):
    for k, v in data.items():
        yield k, v, max_points, min_int, amp_cut


def extract_all_scales(data, max_points=5000, min_int=80, amp_cut=0.001, mp=True):
    if mp:
        with Pool(N_PROC) as pool:
            return {k: x for x, k in zip(pool.starmap(extract_scale_from_pitch_trace, ex_sc_inp_generator(data, min_int=min_int, amp_cut=amp_cut), 10), data.keys())}
    else:
        all_scale = []
        for k, v in data.items():
            try:
                print(k)
                X, hist = get_pitch_histogram((v[1][v[1]>0])[:max_points], redu=True, norm=True, bin_shift=0)
                all_scale.append(extract_scale(X, hist))
    #           scale = extract_scale(X, hist)[:,1]
    #           all_scale.append(scale - min(scale))
            except Exception as e:
                print(f"{k}\n{e}")
    return all_scale


def ex_ph_fn(data, max_points=5000):
    try:
        return get_pitch_histogram((data[1][data[1]>0])[:max_points], redu=True, norm=True, bin_shift=0)[1]
    except:
        return []


def extract_all_pitch_histograms(data, max_points=5000):
    with Pool(N_PROC) as pool:
        return {k: x for x, k in zip(pool.map(ex_ph_fn, data.values(), 10), data.keys()) if len(x)}


def process_scales_into_dataframe(scales):
    df = pd.DataFrame(columns=["ID", "n_notes", "scale", "ints", "amp", "var"])
    for k, v in scales.items():
        if isinstance(v, type(None)):
            continue
        if len(v) == 0:
            continue
        amp, sca, var = v.T
        sca -= min(sca)
        sca = np.round(sca).astype(int)
        ints = np.diff(np.array(list(sca)+[sca[0]+1200]))
        df.loc[len(df)] = [k, len(sca), sca, ints, amp, var]
    return df


def octave_equivalence(freq, max_points=5000):
    cents, hist = get_pitch_histogram((freq[freq>0])[:max_points])
    autocorr = utils.autocorr(hist)
    N = int(len(autocorr)/2)
    return np.arange(N+1)*np.diff(cents[:2]), autocorr[N:]


def calculate_drift(data, f0=100):
    try:
        f = data[1]
        f = f[f>0]
        i = int(f.size/2)
        bins = np.linspace(0, 1200, 60)
        hist1 = np.histogram((np.log2(f[:i]/f0)+12000)%1200, bins=bins)[0]
        hist2 = np.histogram((np.log2(f[i:]/f0)+12000)%1200, bins=bins)[0]
#       X, hist1 = get_pitch_histogram(f[:int(f.size/2)], redu=True, norm=True, bin_shift=0)
#       X, hist2 = get_pitch_histogram(f[int(f.size/2):], redu=True, norm=True, bin_shift=0)
        return jensenshannon(hist1, hist2)
    except Exception as e:
        print(e)
        return None


def investigate_pitch_drift(df, data):
    with Pool(N_PROC) as pool:
        jsd = pool.map(calculate_drift, data.values(), 10)
    key_key = {k:i for k, i in zip(df.ID, df.index)}
    for j, k in zip(jsd, data.keys()):
        try:
            df.loc[key_key[k], 'drift_jsd'] = j
        except Exception as e:
            print(e, k)
    return df
        

def pitch_autocorrelation(tm, mel, t_max):
#   tm = tm[mel>0]
#   mel = mel[mel>0]
#   cents = np.log2(mel / np.mean(mel))
    dt = np.diff(tm[:2])
    N = int(t_max / dt)
    mel[mel<=0] = np.nan
    cents = np.log2(mel / np.mean(mel[np.isfinite(mel)]))
    return np.arange(N)*dt, utils.acf_2(cents, N)
#   return utils.autocorrelation_function_1d(tm, cents, bins)
        

def pitch_gradient_autocorrelation(tm, mel, t_max):
    dt = np.diff(tm[:2])
    N = int(t_max / dt)

    mel[mel<=0] = np.nan
    cents = np.log2(mel / np.mean(mel[np.isfinite(mel)]))
    c_diff = np.diff(cents)
    idx = np.isfinite(c_diff)
    c_diff = (c_diff - c_diff[idx].mean()) / c_diff[idx].std()
    return np.arange(N) * dt, utils.acf_2(c_diff, N)


def get_cents_from_file(f):
    fr, wav = audio_tools.load_audio(f)
    tm, mel = audio_tools.extract_pitch_from_wav(fr, wav)
    mel[mel<=0] = np.nan
    cents = np.log2(mel / np.nanmean(mel))
    c_diff = np.diff(cents)
    idx = np.isfinite(c_diff)
    c_diff = (c_diff - c_diff[idx].mean()) / c_diff[idx].std()
    return tm, mel, cents, c_diff


def fit_gmm(X, n):
    gm = GaussianMixture(n_components=n).fit(X)
    return gm


def get_gmm_scaling(cents):
    Y = cents[np.isfinite(cents)].reshape(np.isfinite(cents).sum(), 1)
    return [fit_gmm(Y, i).aic(Y) for i in range(1, 21)]


def time_vs_interval_2dhist(tm, cents, cbins, dt=0.05, tmax=2, norm=True, c_abs=False):
    c_dist = np.diff(np.meshgrid(cents, cents), axis=0)[0]
    if c_abs:
        c_dist = np.abs(c_dist)
    t_dist = np.diff(np.meshgrid(tm, tm), axis=0)[0]
    t_bin = t_dist // dt
    steps = int(tmax / dt)
    Xlbl = (np.arange(steps) + 1) * dt
    Ylbl = cbins[:-1] + np.diff(cbins[:2])*0.5
    hist = np.zeros((steps, len(cbins)-1), float)
    
    for i in range(steps):
        hist[i] = np.histogram(c_dist[(t_bin==i)&(np.isfinite(c_dist))], bins=cbins)[0]
        if norm:
            hist[i] = hist[i] / np.nansum(hist[i])

    return hist, Xlbl, Ylbl


def plot_TI_hist(cbins, tm, cents, fig='', ax='', nmax=2000, log=False, c_abs=False):
    idx = np.isfinite(cents)
    tm, cents = tm[idx][:nmax], cents[idx][:nmax]
    hist, Xlbl, Ylbl = time_vs_interval_2dhist(tm, cents, cbins, c_abs=c_abs)
    if isinstance(fig, str):
        fig, ax = plt.subplots()
    if log:
        hist = np.log10(hist)
    im = ax.imshow(hist.T)
    fig.colorbar(im, ax=ax)
    ax.invert_yaxis()
    ax.set_xticks(range(0, len(Xlbl), 10))
    ax.set_xticklabels(Xlbl[::10])
    ax.set_yticks(range(0, len(Ylbl), 10))
    ax.set_yticklabels(Ylbl[::10])


def plot_TI_hist_entropy(cbins, tm, cents, i):
    hist, Xlbl, Ylbl = time_vs_interval_2dhist(tm[i], cents[i], cbins)
    plt.plot(Xlbl, entropy(hist, axis=1), ':')
    

def plot_TI_hist_wander(cbins, tm, cents, i):
    hist, Xlbl, Ylbl = time_vs_interval_2dhist(tm[i], cents[i], cbins)
    plt.plot(Xlbl, [np.sum(np.abs(h * Ylbl)) for h in hist], ':')
    



    

