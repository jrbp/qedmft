#!/usr/bin/env python3
from typing import Union
from warnings import warn
import numpy as np
import numpy.typing as npt
from json import JSONEncoder
from h5py import File

from .units import *
from .adiabatic import AdiabaticMatter
from .data import atomic_masses, atomic_numbers

def adiabatic_dat_from_vasph5(path, enforce_asr=False):
    with File(path, 'r') as f:
        poscar = f['input']['poscar']
        response = f['results']['linear_response']
        # HACK: assuming direct coords, even this check is a pretty ugly type conversion
        assert abs(float(np.array(poscar['scale'])) - 1.0) < 1e-8
        assert bool(np.array(poscar['direct_coordinates']))

        cmat = -np.array(response['force_constants']) * (ANG_PER_BOHR**2 / EV_PER_HARTREE)
        eps3d = np.array(response['electron_dielectric_tensor'])
        cell = np.array(poscar['lattice_vectors']) / ANG_PER_BOHR
        vol = np.dot(np.cross(cell[0], cell[1]), cell[2])
        species = list(map(lambda s: s.decode(), poscar['ion_types']))

        res = AdiabaticMatter(calc_dir = path,
                              species = species,
                              cell = cell,
                              coords = np.array(poscar['position_ions']),
                              #masses = MASS_AMU_FACT * np.array(list(map(lambda s: atomic_masses[atomic_numbers[s]], species))),
                              masses = MASS_AMU_FACT * np.array([atomic_masses[atomic_numbers[s]] for s in species]),
                              cmat = apply_asr_correction(cmat) if enforce_asr else cmat,
                              zs = np.array(response['born_charges']),
                              eps_3D = eps3d,
                              chi0 = vol * (eps3d - np.eye(3)) / (4 * np.pi),)
    return res

def apply_asr_correction(cmat:npt.NDArray[Union[np.float64, np.complex128]], ndim=3):
    warn("asr correction needs testing")
    ndim = 3
    nat = int(len(cmat)/ndim)
    at_dir_basis = np.reshape(symmetrize_mat(cmat), (nat, nat, ndim, ndim))
    at_ident = np.zeros_like(at_dir_basis)
    for i in range(nat):
        at_ident[i,i] = np.eye(3)
    at_diag_part = at_ident @ at_dir_basis
    at_dir_basis_new = at_dir_basis - at_diag_part
    new_diags = -at_dir_basis_new.sum(axis=0)
    for i in range(nat):
        at_dir_basis_new[i,i] = new_diags[i]
    return symmetrize_mat(np.reshape(at_dir_basis_new, cmat.shape))

def symmetrize_mat(cmat:npt.NDArray[Union[np.float64, np.complex128]]):
    # only that cmat should be a hermitian matrix
    return 0.5 * (cmat.T.conj() + cmat)

class NumpyArrayEncoder(JSONEncoder):
    # will switch to hdf5 for freq dependence
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
