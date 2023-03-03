#!/usr/bin/env python3
import numpy as np
from functools import singledispatch
from numpy import typing as npt
from dataclasses import dataclass
from typing import Sequence
from itertools import chain, repeat


def mass_flat(m_at, ndim=3):
    return np.array(list(chain.from_iterable(repeat(m, ndim) for m in m_at)))


@dataclass
class AdiabaticMatter:
    """Data from a finite difference or DFPT calculation"""

    calc_dir: str
    species: Sequence[str]
    cell: npt.NDArray[float]
    coords: npt.NDArray[float]
    masses: npt.NDArray[float]
    cmat: npt.NDArray[float]
    zs: npt.NDArray[float]
    eps_3D: npt.NDArray[float]
    chi0: npt.NDArray[float]

    @property
    def natoms(self) -> int:
        return len(self.species)

    @property
    def nmodes(self, ndim=3) -> int:
        return ndim * len(self.species)

    @property
    def masses_flat(self, ndim=3) -> npt.NDArray[float]:
        return mass_flat(self.masses, ndim)

    @property
    def zmat(self, ndim=3):
        return self.zs.reshape((-1, ndim))


@dataclass
class PhotonModes:
    freqs: npt.NDArray[float]
    lambdas: npt.NDArray[complex]

    @property
    def nmodes(self):
        return len(self.freqs)

    @property
    def masses_flat(self):
        return np.ones(self.nmodes)

    @property
    def omat(self):
        return np.diag(self.freqs)

    def range_scale_lambda(self, scales):
        return [self.__class__(self.freqs, s * self.lambdas) for s in scales]

#def get_coupled_hmat(mat, pht):
def get_coupled_hmat(pht, mat):
    nmodes = mat.nmodes + pht.nmodes
    X = pht.lambdas @ mat.chi0 @ pht.lambdas.T.conj()
    Z = pht.lambdas @ mat.zmat.T
    O = pht.omat
    C = np.block([[mat.cmat, np.zeros_like(Z.T)], [np.zeros_like(Z), np.zeros_like(O)]])
    OplusX = np.block([Z, O])
    elec_op = np.linalg.inv(np.eye(pht.nmodes) + X)
    return C + OplusX.T.conj() @ elec_op @ OplusX

def hmat_to_freqs(hmat_q, masses, freq_only=True):
    dynmat = np.copy(hmat_q)
    for i, m in enumerate(masses):
        dynmat[:, i] *= m**-0.5
        dynmat[i, :] *= m**-0.5
    eigvals, eigvecs = np.linalg.eigh(dynmat)
    freqs = np.sign(eigvals) * np.abs(eigvals) ** 0.5
    if freq_only:
        return freqs
    eigdispls = masses ** (-0.5) * eigvecs.T
    return freqs, eigdispls

# def solve_gen_eig(C, M, G=None):
#    """for (omega^2 M + omega G + C)x = 0 find set of omega,x solutions"""
#    if G is not None or np.linalg.det(M) < 1e-8:
#        raise NotImplementedError("still need to implement the G solver and massless dof (have it from other project though)")
#    # simple version

@singledispatch
def solve_all_qad(pht: PhotonModes, mat: AdiabaticMatter):
    masses = np.concatenate([mat.masses_flat, pht.masses_flat])
    hmat = get_coupled_hmat(pht, mat)
    return hmat_to_freqs(hmat, masses, freq_only=False)

@solve_all_qad.register(list)
def _(pht: Sequence[PhotonModes], mat: AdiabaticMatter):
    freqs_all = []
    displs_all = []
    for p in pht:
        this_fs, this_ds = solve_all_qad(p, mat)
        freqs_all.append(this_fs)
        displs_all.append(this_ds)
    return np.array(freqs_all), np.array(displs_all)

def pht_char(displs, nphtmodes):
    pht_part_all = displs[:,:,-nphtmodes:]
    return np.sqrt((pht_part_all**2).sum(-1))
