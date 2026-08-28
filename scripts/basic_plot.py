#!/usr/bin/env python3

import os
import numpy as np

from qedmft import adiabatic as qad
from qedmft.units import *
from qedmft.plotting import plot_f_vs_lambdas_withchar

# To use vaspout.h5 provided as cli arg:
# import sys
# from qedmft.io import adiabatic_dat_from_vasph5
# dat = adiabatic_dat_from_vasph5(sys.argv[1])
# dat.save_json("./bn_resp_example.json") # saves needed data to json file

# load the json example data
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bn_resp_example.json")
dat = qad.AdiabaticMatter.from_json(fn)

# setup cavity parameters
phon = qad.hmat_to_freqs(dat.cmat, dat.masses_flat) # get the bare phonon freqs
h0 = phon[-2] # fundamental cavity mode freq (here chosen from phonons to be in resonance)
nharm = 1 # number of harmonics to include
fs = np.array([h0 * i for i in range(1, nharm + 1)]) # set of mode frequencies
ds = np.zeros((nharm, 3))
ds[:, 0] = 1 # normalized lambda vectors for each mode

# photon modes with normalized lambdas
photons_norm = qad.PhotonModes(freqs=fs, lambdas=ds)
llambdas = np.linspace(0, 0.25, 1000)

# photon modes over the desired lambda range
phts = photons_norm.range_scale_lambda(llambdas) # set of photon configurations for all 

# solve for each lambda
freqs, displs, irvs, irns = qad.solve_all_qad(phts, dat)
chars = qad.pht_char(displs, phts)

# plot
plt, fig, ax = plot_f_vs_lambdas_withchar(freqs, chars, llambdas)
ax.annotate("BN", (0.01, 0.95), xycoords="axes fraction")
ax.set_ylim((0, 250))
# fig.savefig("./bn_example.png")
plt.show()
