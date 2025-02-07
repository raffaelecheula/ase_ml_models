# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

from ase_ml_models.utilities import get_connectivity, plot_connectivity

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
    ads.translate(atoms.positions[31]-ads.positions[1]+[0., 0., 1.80])
    atoms += ads
    indices_ads = [36, 37]
    method = "ase"
    # Connectivity.
    connectivity = get_connectivity(
        atoms=atoms,
        method=method,
    )
    # Plot connectivity.
    plot_connectivity(
        atoms=atoms,
        connectivity=connectivity,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------