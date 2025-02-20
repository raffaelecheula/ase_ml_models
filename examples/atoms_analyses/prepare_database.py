# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.io import read
from ase.db import connect

from ase_ml_models.databases import write_atoms_list_to_db
from ase_ml_models.utilities import get_connectivity, plot_connectivity
from ase_ml_models.features import (
    get_features_const,
    get_features_soap,
    write_features,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    filename_list = ["atoms.traj"] # TODO: list of filenames.
    db_ase_name = "db_ase.db"
    filename_features = "features.txt"

    # Features names.
    features_names = [
        "e_affinity", # Electron affinity
        "en_pauling", # Pauling electronegativity
        "ion_pot", # Ionization potential
        "d_radius", # radius of d orbitals
        "work_func", # work function
        "d_filling", # d-band filling
        "d_center", # d-band center
        "d_width", # d-band width
        "d_skewness", # d-band skewness
        "d_kurtosis", # d-band kurtosis
        "sp_filling", # sp-band filling
        "d_density", # density of d states at Fermi
        "sp_density", # density of sp states at Fermi
    ]
    features_names += [f'SOAP_{ii+1:02d}' for ii in range(70)]

    # Read atoms and get features.
    atoms_list = []
    for filename in filename_list:
        # Read atoms.
        atoms = read(filename=filename)
        atoms.info["name"] = "name of structure" # TODO: name of the structure.
        atoms.info["indices_ads"] = [36, 37] # TODO: indices of the adsorbate.
        # Get connectivity.
        atoms.info["connectivity"] = get_connectivity(atoms=atoms, method="ase")
        # Get features const.
        features_const = get_features_const(atoms=atoms)
        # Get clean atoms.
        atoms_clean = read(filename="atoms_clean.traj") # TODO: get the corresponding clean atoms.
        workfunction = 10. # TODO: get the workfunction of the clean atoms.
        features_bands = np.ones((len(atoms_clean), 8)) # TODO: get the features of the clean atoms.
        # Get features soap.
        features_soap = get_features_soap(atoms=atoms_clean)
        # Assemble features matrix.
        features = np.ones((len(atoms), len(features_names))) * np.nan
        features[:, :4] = features_const
        features[:, 4] = workfunction
        features[:len(atoms_clean), 5:13] = features_bands
        features[:len(atoms_clean), -70:] = features_soap
        # Store features in info.
        atoms.info["features"] = features
        atoms.info["features_names"] = features_names
        atoms_list.append(atoms)
    # Write atoms to database.
    if db_ase_name is not None:
        db_ase = connect(name=db_ase_name, append=False)
        write_atoms_list_to_db(
            atoms_list=atoms_list,
            db_ase=db_ase,
            keys_store=["name"],
            keys_match=["name"],
        )
    # Write features to file.
    if filename_features is not None:
        write_features(atoms_list=atoms_list, filename=filename_features)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------