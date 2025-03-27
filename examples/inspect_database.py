# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect
from ase.gui.gui import GUI

from ase_ml_models.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    species_type = "reactions" # adsorbates | reactions
    kwargs = {"species": "H2O*→OH*+H*"}
    
    # Read Ase database.
    db_ase_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_ase = connect(db_ase_name)
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    
    for ii, atoms in enumerate(atoms_list):
        species = atoms.info["species"]
        surface = atoms.info["surface"]
        E_act = atoms.info["E_act"]
        ΔE_react = atoms.info["ΔE_react"]
        print(f"{ii+1:2d} {species:15s} {surface:15s} {E_act:6.2f} {ΔE_react:6.2f}")

    # GUI.
    gui = GUI(atoms_list)
    gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------