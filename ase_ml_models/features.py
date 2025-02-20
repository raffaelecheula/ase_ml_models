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
):
    """Get constant features."""
    from mendeleev import element
    features_const = np.zeros((len(atoms), 4))
    for ii, atom in enumerate(atoms):
        elem = element(atom.symbol)
        features_const[ii, 0] = elem.electron_affinity
        features_const[ii, 1] = elem.en_pauling
        features_const[ii, 2] = elem.ionenergies[1]
        if elem.metallic_radius:
            features_const[ii, 3] = elem.metallic_radius * 0.01
        else:
            features_const[ii, 3] = np.nan
    return features_const

# -------------------------------------------------------------------------------------
# GET FEATURES SOAP
# -------------------------------------------------------------------------------------

def get_features_soap(
    atoms: Atoms,
    periodic: bool = True,
    rcut: int = 3,
    nmax: int = 4,
    lmax: int = 6,
    sigma: float = 0.35,
    sparse: bool = False,
):
    """Get SOAP features."""
    from dscribe.descriptors import SOAP
    atoms_copy = atoms.copy()
    atoms_copy.symbols = ["X" for _ in atoms_copy]
    soap_desc = SOAP(
        species=["X"],
        periodic=periodic,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        sigma=sigma,
        sparse=sparse,
    )
    return soap_desc.create(atoms_copy)

# -------------------------------------------------------------------------------------
# PRINT FEATURES
# -------------------------------------------------------------------------------------

def write_features(
    atoms_list: list,
    filename: str = "features.txt",
):
    """Write features to file."""
    with open(filename, 'w') as fileobj:
        for atoms in atoms_list:
            features = atoms.info["features"]
            features_names = atoms.info["features_names"]
            print(f'{"symbol":7s}', end='', file=fileobj)
            for ii in range(features.shape[1]):
                print(f'  {features_names[ii]:11s}', end='', file=fileobj)
            print('', file=fileobj)
            for ii in range(features.shape[0]):
                print(f'{atoms[ii].symbol:7s}', end='', file=fileobj)
                for feature in features[ii, :]:
                    print(f'{feature:+13.4e}', end='', file=fileobj)
                print('', file=fileobj)
            print('', file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------