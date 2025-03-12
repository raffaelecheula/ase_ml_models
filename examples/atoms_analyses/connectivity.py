# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.io import read
from matplotlib.pyplot import savefig

from ase_ml_models.utilities import (
    get_connectivity,
    get_connectivity_from_list,
    plot_connectivity,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main(struct="TS"):

    # Parameters.
    rot_x = +20
    rot_y = -125

    # Adsorbate.
    atoms = read(f"atoms_{struct}.traj")
    atoms.cell[2,2] += 5.
    if struct != "gas":
        atoms[31].symbol = "Pt"
        atoms.translate([+0.4, +0.4, +0.0])
    # Connectivity.
    if struct == "TS":
        atoms.info["connectivity"] = get_connectivity_from_list(
            atoms_list=[read(f"atoms_{ss}.traj") for ss in ["IS", "FS"]],
            method="ase",
            ensure_bonding=True,
        )
    else:
        atoms.info["connectivity"] = get_connectivity(
            atoms=atoms,
            method="ase",
            ensure_bonding=True,
        )
    # Write graph.
    ax = plot_connectivity(atoms=atoms, show_plot=False, alpha=1.0)
    ax.elev = rot_x
    ax.azim = -90-rot_y
    savefig(f"graph_{struct}.png", dpi=300, transparent=True)
    # Write struture.
    atoms.write(
        filename=f"image_{struct}.png",
        scale=300,
        maxwidth=500,
        radii=0.95,
        rotation=f"-90x,{rot_y}y,{rot_x}x",
        show_unit_cell=False,
    )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    for struct in ["clean", "gas", "IS", "TS", "FS"]:
        main(struct=struct)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------