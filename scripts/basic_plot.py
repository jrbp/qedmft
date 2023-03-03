#!/usr/bin/env python3

import os
import numpy as np
from qedmft.io import adiabatic_dat_from_p4vsp
from qedmft import adiabatic as qad
from qedmft.units import *
from qedmft.plotting import plot_f_vs_lambdas_withchar

base_dir = "/mnt/home/jbonini/ceph/photon_DMFT/linear_gamma/BN/norelax/"
dat = adiabatic_dat_from_p4vsp(os.path.join(base_dir, "chi_0/a0"))

phon = qad.hmat_to_freqs(dat.cmat, dat.masses_flat)
f = phon[-2]
nharm = 1
h0 = f
fs = np.array([h0 * i for i in range(1, nharm + 1)])
ds = np.zeros((nharm, 3))
ds[:, 0] = 1
photons_norm = qad.PhotonModes(freqs=fs, lambdas=ds)
llambdas = np.linspace(0, 0.25, 1000)

phts = photons_norm.range_scale_lambda(llambdas)
freqs, displs, irvs, irns = qad.solve_all_qad(phts, dat)
pht_part_all = displs[:, :, -photons_norm.nmodes :]
chars = qad.pht_char(displs, nmodes)

plt, fig, ax = plot_f_vs_lambdas_withchar(freqs, chars, llambdas)
ax.annotate("BN", (0.01, 0.95), xycoords="axes fraction")
ax.set_ylim((0, 250))
