#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from .units import EV_PER_HARTREE


def plot_f_vs_lambdas_withchar(freqs, chars, lambdas, ax=None):
    freqs = freqs * EV_PER_HARTREE * 1e3  # to meV
    base_cmap = mpl.colormaps["inferno"]
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "c_partial", base_cmap(np.linspace(0, 0.65, base_cmap.N))
    )
    a = 5
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.618 * a, a))
    else:
        fig = ax.get_figure()
    norm = plt.Normalize(chars.min(), chars.max())
    for fs, cs in zip(freqs.T, chars.T):
        points = np.array([lambdas, fs]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=new_cmap, norm=norm)
        lc.set_array(cs[:-1])
        ax.add_collection(lc)
    ax.set_ylim((np.min(freqs) - (pad := np.max(freqs) * 1e-2), np.max(freqs) + pad))
    ax.set_xlim((lambdas[0], lambdas[-1]))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\hbar\omega$ (meV)")
    fig.colorbar(mpl.cm.ScalarMappable(cmap=new_cmap), ax=ax, label="photon character")
    return plt, fig, ax


def get_lorrenztian(freq, peak, sigma=0.01, old=True, cutoff=None):
    if old:
        def l(x):
            return peak * (sigma / 2 * np.pi) / ((x - freq) ** 2 + (sigma / 2) ** 2)
    else:
        def l(x):
            return peak * (np.pi * sigma * (1 + ((x-freq)/sigma)**2))**-1
    return l


def get_spectra_func(f, c, broadening=0.1, old=True):
    broadened_funcs = (get_lorrenztian(freq, ir, broadening) for freq, ir in zip(f, c))
    return lambda x: sum((sm(x) for sm in broadened_funcs))


def plot_ir_basic(freqs, peaks, lambdas, broadening=0.1, freqs_range=None):
    freqs = freqs * EV_PER_HARTREE * 1e3  # to meV
    if len(lambdas) > 10:
        raise ValueError("I promise you don't want that many subplots")
    fig, axs = plt.subplots(nrows=len(lambdas), sharex=True, sharey=True)
    if freqs_range is None:
        freqs_range = np.linspace(0, freqs.max() * 1.3, 500)
    for a, fs, ps, l in zip(axs[::-1], freqs, np.abs(peaks), lambdas):
        spectra = get_spectra_func(fs, ps, broadening)
        a.plot(freqs_range, spectra(freqs_range))
        a.annotate(f"$\\lambda$={l:.3f}", (0.01, 0.8), xycoords="axes fraction")
        a.set_xlim(0, freqs_range.max())
    return plt, fig, axs


def plot_ir_2D(freqs, peaks, lambdas, broadening=0.01, freqs_range=None, ax=None):
    freqs = freqs * EV_PER_HARTREE * 1e3  # to meV
    spectra_funcs = (
        get_spectra_func(freq, ir, broadening) for freq, ir in zip(freqs, np.abs(peaks))
    )
    if freqs_range is None:
        freqs_range = np.linspace(0, freqs.max() * 1.1, 1000)
    irmat = np.zeros((len(freqs_range), len(lambdas)))
    for i, sf in enumerate(spectra_funcs):
        irmat[:, i] = sf(freqs_range)

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.618 * 5, 5))
    else:
        fig = ax.get_figure()
    im = ax.imshow(
        irmat,
        extent=[0, lambdas.max(), 0, freqs_range.max()],
        origin="lower",
        vmin=0,
        vmax=np.max(irmat),
        #aspect=lambdas.max() / freqs_range.max(),
        aspect="auto",
        # interpolation="none",
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\hbar \omega$ (meV / u. c.)")
    fig.colorbar(im, ax=ax, label="IR intensity")
    return plt, fig, ax
