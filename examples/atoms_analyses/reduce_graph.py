# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

from ase_ml_models.utilities import (
    get_connectivity,
    plot_connectivity,
    get_reduced_graph_atoms,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    from ase.build import molecule, fcc111
    from ase.constraints import FixAtoms
    # Surface.
    atoms = fcc111("Pt", (3, 3, 4), vacuum=12.0)
    atoms.set_constraint(FixAtoms(mask=[atom.tag > 2 for atom in atoms]))
    # Adsorbate.
    ads = molecule("CO")
    ads.translate(atoms.positions[28]-ads.positions[1]+[0., 0., 1.80])
    #ads.translate(atoms.positions[31]-ads.positions[1]+[0., 0., 1.80])
    atoms += ads
    indices_ads = [36, 37]
    # Parameters.
    method = "ase"
    bond_cutoff = 2
    # Get reduced graph atoms.
    atoms = get_reduced_graph_atoms(
        atoms=atoms,
        indices_ads=indices_ads,
        method=method,
        bond_cutoff=bond_cutoff,
    )
    ## Plot connectivity.
    plot_connectivity(
        atoms=atoms,
        connectivity=atoms.info["connectivity"],
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------