#!/usr/bin/env python3
from typing import Union
from warnings import warn
import numpy as np
import numpy.typing as npt
from json import JSONEncoder
from py4vasp import Calculation

from .units import *
from .adiabatic import AdiabaticMatter

def adiabatic_dat_from_p4vsp(path, enforce_asr=False):
    """
    TODO: should switch to just parsing the xml/hdf5 myself
          or using ase or pymatgen
          as py4vasp has annoying dependencies and wants to
          always open jupyter notebooks

    """
    vcalc = Calculation.from_path(path)
    res = {}
    ase_struct = vcalc.structure.to_ase()
    vol_au = ase_struct.cell.volume / ANG_PER_BOHR**3
    res["calc_dir"] = path
    res["species"] = ase_struct.get_chemical_symbols()
    res["cell"] = ase_struct.cell / ANG_PER_BOHR
    res["coords"] = ase_struct.get_scaled_positions()
    res["masses"] = ase_struct.get_masses() * MASS_AMU_FACT
    # is vasp sign convention for forces or second energy derivs?
    cmat_raw = -vcalc.force_constant.to_dict()["force_constants"] * (ANG_PER_BOHR**2 / EV_PER_HARTREE)
    if enforce_asr:
        res["cmat"] = apply_asr_correction(cmat_raw)
    else:
        res["cmat"] = cmat_raw
    res["zs"] = vcalc.born_effective_charge.to_dict()["charge_tensors"]
    res["eps_3D"] = vcalc.dielectric_tensor.to_dict()["clamped_ion"]
    res["chi0"] = vol_au * (vcalc.dielectric_tensor.to_dict()["clamped_ion"] - np.eye(3)) / (4 * np.pi)
    return AdiabaticMatter(**res)

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
