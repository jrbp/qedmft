#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np
from .units import EV_PER_HARTREE


def plot_f_vs_lambdas_withchar(freqs, chars, lambdas):
    freqs = freqs * EV_PER_HARTREE * 1e3 # to meV
    base_cmap = mpl.colormaps["inferno"]
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('c_partial',
                                                            base_cmap(np.linspace(0, 0.65, base_cmap.N)))
    a=5
    fig, ax = plt.subplots(figsize=(1.618*a, a))
    norm = plt.Normalize(chars.min(), chars.max())
    for fs, cs in zip(freqs.T, chars.T):
        points = np.array([lambdas, fs]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=new_cmap, norm=norm)
        lc.set_array(cs[:-1])
        ax.add_collection(lc)
    ax.set_ylim((np.min(freqs)-(pad:=np.max(freqs)*1e-2), np.max(freqs)+pad))
    ax.set_xlim((lambdas[0], lambdas[-1]))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\hbar\omega$ (meV)")
    fig.colorbar(mpl.cm.ScalarMappable(cmap=new_cmap), ax=ax, label="photon character")
    return plt, fig, ax
