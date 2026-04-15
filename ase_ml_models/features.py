# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase import Atoms
from ase.db.core import Database

# -------------------------------------------------------------------------------------
# GET FEATURES CONST
# -------------------------------------------------------------------------------------

def get_features_const(
    atoms: Atoms,
    features_dict: dict = None,
    features_names: list = ["Eaff", "Eneg", "Ipot", "Drad"],
):
    """
    Get constant features.
    """
    from mendeleev import element
    # Get chemical symbols.
    symbol_list = sorted(set(atoms.get_chemical_symbols()))
    # Get features dict.
    if features_dict is None:
        features_dict = {}
    for symbol in [symbol for symbol in symbol_list if symbol not in features_dict]:
        elem = element(symbol)
        features = []
        if "Eaff" in features_names:
            features.append(elem.electron_affinity)
        if "Eneg" in features_names:
            features.append(elem.en_pauling)
        if "Ipot" in features_names:
            features.append(elem.ionenergies[1] if 1 in elem.ionenergies else np.nan)
        if "Drad" in features_names:
            features.append(
                elem.metallic_radius * 0.01 if elem.metallic_radius else np.nan
            )
        features_dict[symbol] = features
    # Get constant features.
    features_const = np.zeros((len(atoms), len(features_names)))
    for ii, aa in enumerate(atoms):
        features_const[ii] = features_dict[aa.symbol]
    # Return constant features.
    return features_const

# -------------------------------------------------------------------------------------
# GET FEATURES SOAP
# -------------------------------------------------------------------------------------

def get_features_soap(
    atoms: Atoms,
    periodic: bool = True,
    r_cut: int = 3,
    n_max: int = 4,
    l_max: int = 6,
    sigma: float = 0.35,
    sparse: bool = False,
):
    """
    Get SOAP features.
    """
    from inspect import signature
    from dscribe.descriptors import SOAP
    atoms_copy = atoms.copy()
    atoms_copy.symbols = ["X" for _ in atoms_copy]
    # Create SOAP descriptor.
    if "r_cut" in signature(SOAP.__init__).parameters:
        soap_desc = SOAP(
            species=["X"],
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            sparse=sparse,
        )
    else:
        soap_desc = SOAP(
            species=["X"],
            periodic=periodic,
            rcut=r_cut,
            nmax=n_max,
            lmax=l_max,
            sigma=sigma,
            sparse=sparse,
        )
    # Return SOAP features.
    return soap_desc.create(atoms_copy)

# -------------------------------------------------------------------------------------
# GET FEATURES BANDS
# -------------------------------------------------------------------------------------

def get_features_bands(
    energy: list,
    pdos_list: list,
    delta_e: float = 0.1,
):
    """
    Get the features of bands.
    """
    i_zero = np.argmin(np.abs(energy))
    i_minus = np.argmin(np.abs(energy+delta_e))
    i_plus = np.argmin(np.abs(energy-delta_e))
    features_bands = np.zeros((len(pdos_list), 8))
    for ii, pdos_dict in enumerate(pdos_list):
        for orbital in pdos_dict:
            if len(pdos_dict[orbital].shape) > 1:
                pdos_dict[orbital] = np.sum(pdos_dict[orbital], axis=1)
        pdos_sp = pdos_dict["s"]
        pdos_sp += pdos_dict["p"]
        sp_filling = np.trapz(y=pdos_sp[:i_zero], x=energy[:i_zero])
        sp_density = np.sum(pdos_sp[i_minus:i_plus])/len(pdos_sp[i_minus:i_plus])
        if "d" in pdos_dict:
            pdos_d = pdos_dict["d"]
            denom = np.trapz(y=pdos_d, x=energy)
            d_filling = np.trapz(y=pdos_d[:i_zero], x=energy[:i_zero])
            d_density = np.sum(pdos_d[i_minus:i_plus]) / len(pdos_d[i_minus:i_plus])
            d_centre = np.trapz(y=pdos_d*energy, x=energy)/denom
            d_mom_2 = np.trapz(y=pdos_d*np.power(energy-d_centre, 2), x=energy)/denom
            d_width = np.sqrt(d_mom_2)
            d_mom_3 = np.trapz(y=pdos_d*np.power(energy-d_centre, 3), x=energy)/denom
            d_skewness = d_mom_3/np.power(d_width, 3)
            d_mom_4 = np.trapz(y=pdos_d*np.power(energy-d_centre, 4), x=energy)/denom
            d_kurtosis = d_mom_4/np.power(d_width, 4)
        else:
            d_filling = np.nan
            d_density = np.nan
            d_centre = np.nan
            d_width = np.nan
            d_skewness = np.nan
            d_kurtosis = np.nan
        features_bands[ii, 0] = d_filling
        features_bands[ii, 1] = d_centre
        features_bands[ii, 2] = d_width
        features_bands[ii, 3] = d_skewness
        features_bands[ii, 4] = d_kurtosis
        features_bands[ii, 5] = sp_filling
        features_bands[ii, 6] = d_density
        features_bands[ii, 7] = sp_density
    return features_bands

# -------------------------------------------------------------------------------------
# PRINT FEATURES
# -------------------------------------------------------------------------------------

def write_features(
    atoms_list: list,
    filename: str = "features.txt",
):
    """
    Write features to file.
    """
    with open(filename, "w") as fileobj:
        for atoms in atoms_list:
            features = atoms.info["features"]
            features_names = atoms.info["features_names"]
            print(f'{"symbol":7s}', end="", file=fileobj)
            for ii in range(features.shape[1]):
                print(f"  {features_names[ii]:11s}", end="", file=fileobj)
            print("", file=fileobj)
            for ii in range(features.shape[0]):
                print(f"{atoms[ii].symbol:7s}", end="", file=fileobj)
                for feature in features[ii, :]:
                    print(f"{feature:+13.4e}", end="", file=fileobj)
                print("", file=fileobj)
            print("", file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------