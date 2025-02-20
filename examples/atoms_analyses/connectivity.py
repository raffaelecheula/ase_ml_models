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
    atoms_clean = fcc111("Pt", (3, 3, 4), vacuum=12.0)
    atoms_clean.set_constraint(FixAtoms(mask=[aa.tag > 2 for aa in atoms_clean]))
    atoms_clean.write("atoms_clean.traj")
    # Adsorbate.
    ads = molecule("CO")
    ads.translate(atoms_clean.positions[31]-ads.positions[1]+[0., 0., 1.80])
    atoms = atoms_clean+ads
    atoms.info["indices_ads"] = [36, 37]
    atoms.write("atoms.traj")
    # Connectivity.
    connectivity = get_connectivity(
        atoms=atoms,
        method="ase",
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