#!/usr/bin/env python3
import numpy as np
from json import JSONEncoder
from py4vasp import Calculation

from .units import *
from .adiabatic import AdiabaticMatter

def adiabatic_dat_from_p4vsp(path):
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
    res["cmat"] = -vcalc.force_constant.to_dict()["force_constants"] * (ANG_PER_BOHR**2 / EV_PER_HARTREE)
    res["zs"] = vcalc.born_effective_charge.to_dict()["charge_tensors"]
    res["eps_3D"] = vcalc.dielectric_tensor.to_dict()["clamped_ion"]
    res["chi0"] = vol_au * (vcalc.dielectric_tensor.to_dict()["clamped_ion"] - np.eye(3)) / (4 * np.pi**2)
    return AdiabaticMatter(**res)


class NumpyArrayEncoder(JSONEncoder):
    # will switch to hdf5 for freq dependence
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
