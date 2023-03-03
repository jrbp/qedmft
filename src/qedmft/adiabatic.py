#!/usr/bin/env python3
import numpy as np
from numpy import typing as npt
from dataclasses import dataclass
from typing import Sequence
from itertools import chain, repeat


def mass_flat(m_at, ndim=3):
    return list(chain.from_iterable(repeat(m, ndim) for m in m_at))

def hmat_to_freqs(hmat_q, masses=None):
    dynmat = np.copy(hmat_q)
    if masses is not None:
        for i, m in enumerate(masses):
            dynmat[:, i] *= m**-0.5
            dynmat[i, :] *= m**-0.5
    eigvals = np.linalg.eigvalsh(dynmat)
    freqs = np.sign(eigvals) * np.abs(eigvals) ** 0.5
    return freqs

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

def get_coupled_hmat(mat, pht):
    nmodes = mat.nmodes + pht.nmodes
    X =  pht.lambdas @ mat.chi0 @ pht.lambdas.T.conj()
    Z = pht.lambdas @ mat.zmat.T
    O = pht.omat
    C = np.block([[mat.cmat, np.zeros_like(Z.T)],
                  [np.zeros_like(Z), np.zeros_like(O)]])
    OplusX = np.block([Z, O])
    elec_op = np.linalg.inv(np.eye(pht.nmodes) + X)
    return C + OplusX.T.conj() @ elec_op @ OplusX
