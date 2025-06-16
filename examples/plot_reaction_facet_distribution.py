# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from ase.build import bulk
from wulffpack import SingleCrystal

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # From DFT calculations.
    surface_energies_metals = {
        "Rh": {(1,1,1): 0.133, (1,0,0): 0.155},
        "Pd": {(1,1,1): 0.103, (1,0,0): 0.108},
        "Co": {(1,1,1): 0.050, (1,0,0): 0.061},
        "Ni": {(1,1,1): 0.102, (1,0,0): 0.122},
        "Cu": {(1,1,1): 0.073, (1,0,0): 0.084},
        "Au": {(1,1,1): 0.060, (1,0,0): 0.066},
        "Pt": {(1,1,1): 0.089, (1,0,0): 0.110},
        "Ru": {(1,1,1): 0.166, (1,0,0): 0.194},
        "Ag": {(1,1,1): 0.071, (1,0,0): 0.072},
        "Os": {(1,1,1): 0.189, (1,0,0): 0.238},
        "Fe": {(1,1,1): 0.356, (1,0,0): 0.312},
        "Ir": {(1,1,1): 0.132, (1,0,0): 0.176},
    }
    
    colors = ["mediumslateblue", "darkorange"]
    
    metals = []
    fractions = {"111": [], "100": []}
    results_all = {}
    for metal, surface_energies in surface_energies_metals.items():
        # Get particle.
        particle = SingleCrystal(
            surface_energies=surface_energies,
            primitive_structure=bulk("Rh", cubic=True),
            natoms=1000,
        )
        metals.append(metal)
        fractions["111"].append(particle.facet_fractions[(1,1,1)] * 100)
        fractions["100"].append(particle.facet_fractions[(1,0,0)] * 100)
        results_all[metal] = {
            "111": float(particle.facet_fractions[(1,1,1)]),
            "100": float(particle.facet_fractions[(1,0,0)]),
        }

    # Write facet fractions.
    yaml_results = "facets_fractions.yaml"
    # Custom YAML representer for floats.
    def float_representer(dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:float', f"{value:7.4E}")
    yaml.add_representer(float, float_representer)
    with open(yaml_results, "w") as fileobj:
        yaml.dump(
            data=results_all,
            stream=fileobj,
            default_flow_style=False,
            sort_keys=False,
        )

    # Facet fractions plot.
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    for ii, fraction in enumerate(fractions.values()):
        ax.bar(
            x=metals,
            height=fraction,
            bottom=None if ii == 0 else fractions["111"],
            color=colors[ii],
            label=f"({list(fractions.keys())[ii]})",
        )
    ax.set_xlim(-0.5, len(metals) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Facet fraction [%]")
    ax.legend(
        edgecolor="black",
        framealpha=1.,
        loc="lower left",
        bbox_to_anchor=(0.001, 0.001)
    )
    # Save plot.
    dirname = f"images/wulff_construction"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/facet_distribution.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------