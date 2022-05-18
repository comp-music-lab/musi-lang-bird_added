"""

Figures for FMA (ISMIR) submission 2020

"""
from itertools import product
from pathlib import Path
import os
import sys


from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set2_8
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import spectrogram
from scipy.spatial import distance_matrix
from scipy.stats import kstest, ks_2samp, linregress, pearsonr
import statsmodels.nonparametric.api as smnp


import audio_tools
import bss_utils as utils
import pitch_analysis as PA


PATH_BASE = Path("C:/Users/yuto\Documents/PycharmProjects/FMA_BSS/")
PATH_DATA = PATH_BASE.joinpath("9_samples")
PATH_PETE = PATH_BASE.joinpath("peter_data")
PATH_RES = PATH_BASE.joinpath("Results")
#FIG_DIR = "/home/jmcbride/Projects/Shoichiro/Figs"
PATH_FIG  = PATH_BASE.joinpath('Figures')

FILES = {'human':["Yangguan Sandie.wav", "Ireland old style.wav", "p2.birthday1.wav"],
         'speech':["Sometimes behave so strangly.wav", "Vietnamese.wav", "English_short.wav"],
         'bird':["KAUAI.wav", "FIREB.wav", "CANYO.wav"]}



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



##########################################################################
### Visualising samples


def plot_spectrograms(each=False):
    fig, ax = plt.subplots(3,3)
    ax = ax.reshape(ax.size)
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            f = PATH_DATA.joinpath(f)
            fr, wav = audio_tools.read_wav(f)
            spec = spectrogram(wav, fr)[2]
            ax[i*3+j].imshow(np.log(spec), cmap='Greys')
            ax[i*3+j].invert_yaxis()
#           ax[i*3+j].imshow(spectrogram(wav), fr, cmap='grey')


def plot_extracted_audio(alg='pyin_smooth', out=False):
#   fig, ax = plt.subplots(3,3)
#   ax = ax.reshape(ax.size)
    out_list = []
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            fig, ax = plt.subplots()
            f = PATH_DATA.joinpath(f)
            fr, wav = audio_tools.read_wav(f)
            tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg=alg)
            ax.plot(tm, mel, 'o', c='k', alpha=0.3, ms=3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            if f.stem == "p2.birthday1":
                ax.set_ylim(140, 305)
            fig.savefig(f"pitch_{f.stem}.pdf")
#           ax[i*3+j].plot(tm, mel, 'o', c='k', alpha=0.3, ms=3)
#           ax[i*3+j].set_title(f.stem)
#           ax[i*3+j].set_xlabel('Time')
#           ax[i*3+j].set_ylabel('Frequency')
            out_list.append((tm, mel))
    if out:
        return out_list


def plot_pitch_tracking_segmented(case=0, alg='pyin_smooth'):
    fig, ax = plt.subplots(3,3, figsize=(16,12))
    fig.subplots_adjust(hspace=0.5)
    ax = ax.reshape(ax.size)
    if case:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':False},
                  {'spline':False, 'ycut':0.055},
                  {'spline':False, 'ycut':0.08},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False}]
    else:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':True},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False, 'ycut':0.02},
                  {'spline':True}]
    ft = 16
    ttls = ['Good', 'Okay', 'Okay', 'Discuss', 'Discuss', 'Okay', 'Good', 'Discuss', 'Discuss']
    ttls = ['1st', '3rd', '5th', '7th', '6th', '7th', '4th', '1st', '9th']
    idx = [0, 2, 4, 6, 5, 7, 3, 1, 8]
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            f = PATH_DATA.joinpath(f)
            fr, wav = audio_tools.read_wav(f)
            tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg=alg)
            PA.analyse_note_curvature(tm, mel, ax=ax[i*3+j], plot=True, **params[i*3+j])
            ax[i*3+j].set_title(ttls[i*3+j])
#   ax[0].annotate("Good", (0.1, 0.9), xycoords='axes fraction', fontsize=ft)

    fig.savefig(f"../segmentation_{case}.pdf", bbox_inches='tight')
    fig.savefig(f"../segmentation_{case}.png", bbox_inches='tight')



def plot_freq_histograms(data, log=False, piano=True):
    fig, ax = plt.subplots(3,3)
    ax = ax.reshape(ax.size)
    for i, (tm, freq) in enumerate(data):
        if log:
            if piano:
                sns.distplot(np.log(freq/440)/np.log(2), ax=ax[i], bins=50)
                ax[7].set_xlabel(r'$log_{2}(f/440)$, frequency relative to concert pitch (octaves)')
            else:
                sns.distplot(np.log(freq)/np.log(2), ax=ax[i], bins=50)
                ax[7].set_xlabel(r'$log_{2}$ frequency (octaves)')
        else:
            sns.distplot(freq, ax=ax[i])



def plot_pitch_histograms(data, redu=False, alg='mode', norm=False):
    fig, ax = plt.subplots(3,3)
    ax = ax.reshape(ax.size)
    for i, (tm, freq) in enumerate(data):
        if alg=='mode':
            denom = find_maximum_freq(freq)
        elif alg=='min':
            denom = freq.min()
        cents = np.log(freq/denom) / np.log(2) * 1200
        if redu:
            cents = np.array([(x+2450)%1200 for x in cents])
            dx = 5
            bins = np.arange(0, 1200+dx, dx)
            X = bins[:-1] + 0.5 * np.diff(bins[:2]) - 50
        else:
            bins = np.linspace(cents.min(), cents.max(), 50)
            print(f"Range = {cents.max() - cents.min()}")
            dx = np.diff(bins[:2])
            X = bins[:-1] + 0.5 * np.diff(bins[:2])
        hist = np.histogram(cents, bins=bins)[0]
        if norm:
            hist /= hist.sum()
        ax[i].bar(X, hist, dx)
#       ax[i].plot(X, hist)



##########################################################################
### Pitch discreteness


def plot_error_scaling_new(case=0):
    lbls = ['Human', 'Speech', 'Bird']
    cols = np.array(Paired_12.hex_colors)[[1,7,3]]
    cols2 = np.array(Paired_12.hex_colors)[[1,1,1,7,7,7,3,3,3]]

    if case:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':False},
                  {'spline':False, 'ycut':0.055},
                  {'spline':False, 'ycut':0.08},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False}]
    else:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':True},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False, 'ycut':0.02},
                  {'spline':True}]

#   fig, ax = plt.subplots(3,3, sharey=True)
    fig, ax = plt.subplots(3,3)
    fig2, ax2 = plt.subplots(3,1)
    fig3, ax3 = plt.subplots(2,1)
    lbls2 = [f"{a}_{i}" for a in lbls for i in range(1,4)]
    barY1 = []
    barY2 = []
    barY3 = []
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            f = PATH_DATA.joinpath(f)
            err = PA.get_error_file(f, kwargs=params[i*3+j])
            if not len(err):
                continue
            if not j:
#               ax[i*3+j].plot(range(err.size), err / err.min(), color=cols[i], label=lbls[i], alpha=.4)
                ax[i,j].plot(range(err.size), err, color=cols[i], label=lbls[i], alpha=.4)
            else:
#               ax[i*3+j].plot(range(err.size), err / err.min(), color=cols[i], alpha=.4)
                ax[i,j].plot(range(err.size), err, color=cols[i], alpha=.4)
            ax[i,j].set_ylim(0, err.max()*1.1)
#           ax2.plot(range(err.size), err, color=cols[i], label=f"{lbls[i]}_{j+1}", alpha=1-j*.3)
#           ax3[0].plot(range(err.size), err / err.max(), color=cols[i], alpha=.4)
#           ax3[1].plot(range(err.size), err.max() -err, color=cols[i], alpha=.4)
            ax3[0].plot(range(err.size-1), [hi/lo for lo, hi in zip(err[:-1], err[1:])], color=cols[i], alpha=.4)
            ax3[1].plot(range(err.size-1), [lo-hi for lo, hi in zip(err[:-1], err[1:])], color=cols[i], alpha=.4)
            barY1.append(err[0] / err[5])
            barY2.append(err[1] / err[0])
            barY3.append(err[0])
    ax2[0].bar(range(9), barY1, 0.5, color=cols2)
    ax2[1].bar(range(9), barY2, 0.5, color=cols2)
    ax2[2].bar(range(9), barY3, 0.5, color=cols2)
    for a in ax2:
        a.set_xticks(range(9))
        a.set_xticklabels(lbls2, rotation=90)
#   ax.legend(loc='upper right', frameon=False)
    for a in ax.ravel():
        a.set_xticks(range(10))
    ax[2,1].set_xlabel('Polynomial degree')
    ax[1,0].set_ylabel('Mean percentage error')
    ax2[0].set_ylabel("0th Error / 5th Error")
    ax2[1].set_ylabel("1st Error / 0th Error")
    ax2[2].set_ylabel("0th Error")

    fig.savefig(f"../error_scaling_{case}.pdf", bbox_inches='tight')
    fig2.savefig(f"../discreteness_{case}.pdf", bbox_inches='tight')



##########################################################################
### Paper Figures



def fig1(case=1, alg='pyin_smooth'):
#   fig, ax = plt.subplots(9,1, figsize=( 8,24))
    idx = [0, 2, 4, 6, 5, 7, 3, 1, 8]
    fig, ax = plt.subplots(3,3, figsize=(16,12))
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    fig3 = plt.figure(figsize=(16,12))
    gs = GridSpec(3,5, width_ratios=[1,1,1,.2,1])
    ax3 = [fig3.add_subplot(gs[i//3, i%3]) for i in idx] + [fig3.add_subplot(gs[:,4])]
    fig4 = plt.figure(figsize=(16,12))
    gs = GridSpec(4,3, height_ratios=[1,1,1,1])
    ax4 = [fig4.add_subplot(gs[i//3, i%3]) for i in idx] + [fig4.add_subplot(gs[3,:])]
    fig.subplots_adjust(hspace=0.4)
    fig3.subplots_adjust(hspace=0.4)
    fig4.subplots_adjust(hspace=0.4)
    idx2 = np.argsort(idx)
    ax = ax.reshape(ax.size)[idx]
    cols = np.array(Paired_12.hex_colors)[[1,7,3]]
    cols2 = np.array(Paired_12.hex_colors)[[1,1,1,7,7,7,3,3,3]][idx2]
    if case:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':False},
                  {'spline':False, 'ycut':0.055},
                  {'spline':False, 'ycut':0.08},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False}]
    else:
        params = [{'spline':False},
                  {'spline':True, 'mcut':0.1, 'dycut':10},
                  {'spline':True, 'mcut':0.05, 'dycut':10},
                  {'spline':True},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False},
                  {'spline':False, 'ycut':0.02},
                  {'spline':True}]
    prefix = [f"{a}{i} - " for a in 'HSB' for i in [1,2,3]]
    ft = 16
    names = ["Yangguan Sandie", "Ireland Old Style", "Happy Birthday",
            "'Sometimes behave so strangely'", "Vietnamese", "American English",
            "Kauai O' o", "Firecrest", "Canyon wren"]
    ttls = []
    Y = []
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            f = PATH_DATA.joinpath(f)
            fr, wav = audio_tools.read_wav(f)
            tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg=alg)
            tm, mel = tm[mel>0], mel[mel>0]
            err = PA.get_error_file(f, kwargs=params[i*3+j])
            Y.append(err[0]*100)
            ttls.append(prefix[i*3+j] + names[i*3+j].split()[0])
            for a in [ax[i*3+j], ax3[i*3+j], ax4[i*3+j]]:
                PA.analyse_note_curvature(tm, mel, ax=a, plot=True, **params[i*3+j])
                a.set_title(prefix[i*3+j] + names[i*3+j])
                a.set_xlabel('')
                a.set_ylabel('')
                if f.stem == "p2.birthday1":
                    a.set_ylim(140, 305)
    for a in [ax, ax3, ax4]:
        for i in [3,5,8]:
            a[i].set_xlabel('Time (s)')
        for i in [0,6,3]:
            a[i].set_ylabel('Frequency (Hz)')
    for a in [ax2, ax4[-1]]:
        a.bar(range(9), np.array(Y)[idx2], 0.5, color=cols2)
        a.set_xticks(range(9))
        a.set_xticklabels(np.array(ttls)[idx2], rotation=90)
        a.set_ylabel('Mean Percentage Error')
    for a in [ax3[-1]]:
        a.barh(range(9)[::-1], np.array(Y)[idx2], 0.5, color=cols2)
        a.set_yticks(range(9))
        a.set_yticklabels(np.array(ttls)[idx2][::-1], rotation=0)
        a.set_xlabel('Mean Percentage Error')
    
    return np.array(Y)

    

    fig.savefig(f"../figA.pdf", bbox_inches='tight')
    fig2.savefig(f"../figB.pdf", bbox_inches='tight')
    fig3.savefig(f"../figC.pdf", bbox_inches='tight')
    fig4.savefig(f"../figD.pdf", bbox_inches='tight')


def human_vs_alg(Y1, Y2):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    idx = np.argsort(Y1)
    prefix = [f"{a}{i}" for a in 'MSB' for i in [1,2,3]]
    width = 0.3
    cols = np.array(Paired_12.hex_colors)[[1,1,1,7,7,7,3,3,3]][idx]
    cols2 = np.array(Paired_12.hex_colors)[[0,0,0,6,6,6,2,2,2]][idx]

    print(idx)

    ax2.bar(np.arange(9)+width/2, np.array(Y1)[idx], width, color=cols)
    ax.bar(np.arange(9)-width/2, np.array(Y2)[idx], width, color=cols2)
    ax.set_xticks(range(9))
    ax.set_xticklabels(np.array(prefix)[idx], rotation=90)
    ax2.set_ylabel('Mean Percentage Error')
    ax.set_ylabel('Mean Pitch-dis. Rating')


def fig1_rev(alg='pyin_smooth'):
    fig = plt.figure(figsize=(16,12))
    gs = GridSpec(5,4, height_ratios=[1,1,1,.15,1], width_ratios=[1,1,.1,.9])
    ax = [a for i in range(3) for a in [fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 2:])]] + \
         [fig.add_subplot(gs[4,3]), fig.add_subplot(gs[4,:2])]
    ax2 = ax[-1].twinx()
    fig.subplots_adjust(hspace=0.4)
    cols = np.array(Paired_12.hex_colors)[[1,7,3]]
    fs = 16

    params = [{'spline':False},
              {'spline':True, 'mcut':0.1, 'dycut':10},
              {'spline':True, 'mcut':0.05, 'dycut':10},
              {'spline':False},
              {'spline':False, 'ycut':0.055},
              {'spline':False, 'ycut':0.08},
              {'spline':False},
              {'spline':False},
              {'spline':False}]


    prefix = [f"{a}{i}" for a in 'MSB' for i in [1,2,3]]
    ft = 16
    names = ["Yangguan Sandie", "Ireland Old Style", "Happy Birthday",
            "'Sometimes behave so strangely'", "Vietnamese", "American English",
            "Kauai O' o", "Firecrest", "Canyon wren"]
    ttls = []
    Y = []
    for i, (dtype, files) in enumerate(FILES.items()):
        for j, f in enumerate(files):
            f = PATH_DATA.joinpath(f)
            fr, wav = audio_tools.read_wav(f)
            tm, mel = audio_tools.extract_pitch_from_wav(fr, wav.astype(float), alg=alg)
            tm, mel = tm[mel>0], mel[mel>0]
            err = PA.get_error_file(f, kwargs=params[i*3+j])
            Y.append(err[0]*100)
            ttls.append(prefix[i*3+j] + names[i*3+j].split()[0])
            for a in [ax[i*3+j]]:
                PA.analyse_note_curvature(tm, mel, ax=a, plot=True, **params[i*3+j])
                a.set_title(prefix[i*3+j] + " - " + names[i*3+j])
                a.set_xlabel('')
                a.set_ylabel('')
                if f.stem == "p2.birthday1":
                    a.set_ylim(140, 305)
    for i in [6,7,8]:
        ax[i].set_xlabel('Time (s)')
    for i in [0,3,6]:
        ax[i].set_ylabel('Frequency (Hz)')

    Y2 = np.loadtxt(PATH_PETE.joinpath('peter_data.txt')).mean(axis=1)[[3,4,5,6,7,8,0,1,2]]
    pd.DataFrame(data={'name':names, 'mean_percentage_error':Y, 'peter_data':Y2}).to_csv(PATH_RES.joinpath("results.csv"), index=False)

    Y2 -= Y2.min()
    Y2 /= Y2.max()

    idx = np.argsort(Y)
    cols = np.array(Paired_12.hex_colors)[[1,1,1,7,7,7,3,3,3]][idx]
    cols2 = np.array(Paired_12.hex_colors)[[0,0,0,6,6,6,2,2,2]][idx]
    lbls = ['Mean Percentage Error', 'Pitch-discreteness\nRating']
    width = 0.3


    ax2.bar((np.arange(9)+width/2), np.array(Y)[idx], width, color=cols, edgecolor='k')
    ax[-1].bar((np.arange(9)-width/2), np.array(Y2)[idx], width, color=cols2, edgecolor='k', hatch='\\')
    for a, l in zip([ax2, ax[-1]], lbls):
        a.set_xticks(range(9))
        a.set_ylabel(l)
    ax[-1].set_xticklabels(np.array(prefix)[idx])

    ax[-1].set_ylim(0, 1.1)
    ax2.set_ylim(0, 6.1)
    cols3 = np.array(Paired_12.hex_colors)[[0,6,2]]
    cols4 = np.array(Paired_12.hex_colors)[[1,7,3]]

    leg_lbls = ['Human music', 'Human speech', 'Birdsong']
    handles = [Patch(facecolor=c, edgecolor='k', label=l, hatch='\\') for c, l in zip(cols3, leg_lbls)] 
#   ax[-1].legend(handles=handles, bbox_to_anchor=(0.36, 1.18), ncol=2, frameon=False)
    ax[-1].legend(loc='upper left', handles=handles, bbox_to_anchor=(0.03, 1.23), ncol=2, frameon=False)
    ax[-1].annotate("Pitch-discreteness rating", (0.07, 1.23), xycoords="axes fraction")

    handles = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(cols4, leg_lbls)] 
    ax2.legend(loc='upper right', handles=handles, bbox_to_anchor=(0.84, 1.23), ncol=2, frameon=False)
    ax2.annotate("Mean Percentage Error", (0.50, 1.23), xycoords="axes fraction")

    ax[-1].spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    cols = np.array(Paired_12.hex_colors)[[1,1,1,7,7,7,3,3,3]]
    ax[-2].scatter(Y, Y2, color=cols)
    ax[-2].set_xlabel("Mean Percentage Error")
    ax[-2].set_ylabel("Pitch-discreteness rating")

    fs = 18
    ax[0].annotate("A", ( 0.05, 0.93), xycoords="figure fraction", fontsize=fs)
    ax[-1].annotate("B", ( 0.05, 0.25), xycoords="figure fraction", fontsize=fs)
    ax[-2].annotate("C", ( 0.68, 0.25), xycoords="figure fraction", fontsize=fs)

    print(pearsonr(Y, Y2))
    plt.show()
    
    fig.savefig(PATH_FIG.joinpath("main_fig.pdf"), bbox_inches='tight')
    fig.savefig(PATH_FIG.joinpath("main_fig.png"), bbox_inches='tight')


if __name__ == "__main__":

    fig1_rev()


