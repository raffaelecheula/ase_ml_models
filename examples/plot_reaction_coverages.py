# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    reaction = "RWGS" # WGS | RWGS
    task = "database" # database | extrapol
    half = None # 1 | 2
    models = {
        "database": "DFT+DFT",
        "extrapol": "WWLGPR+WWLGPR",
    }

    # Read microkinetics results.
    yaml_results = f"results_{task}_{reaction}.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results = results_all[models[task]]
    
    coverages_dict = {
        "*": "silver",
        "CO2**": "black",
        "CO*": "gold",
        "O*": "orangered",
        "COOH**": "black",
        "HCOO**": "black",
        "H*": "darkturquoise",
        "OH*": "black",
        "H2O*": "black",
        "HCO**": "black",
        "COH*": "black",
    }
    
    # DFT data.
    coverages = [[] for _ in coverages_dict]
    surfaces = []
    for surface in results:
        for ii, path in enumerate(coverages_dict):
            coverages[ii].append(results[surface][None]["coverages"][path] * 100)
        surfaces.append(modify_name(surface))
    coverages = np.array(coverages)
    
    # Get half data for extrapolation task.
    if task == "extrapol":
        half_num = len(surfaces) // 2
        if half == 1:
            surfaces = surfaces[:half_num]
            coverages = coverages[:, :half_num]
        elif half == 2:
            surfaces = surfaces[half_num:]
            coverages = coverages[:, half_num:]
    
    # Coverages plot.
    fig, ax = plt.subplots(figsize=(13, 4), dpi=200)
    for ii, path in enumerate(coverages_dict):
        ax.bar(
            x=surfaces,
            height=coverages[ii],
            color=coverages_dict[path],
            label=path,
            bottom=None if ii == 0 else np.sum(coverages[:ii], axis=0),
        )
    ax.set_xlim(-0.5, len(surfaces)-0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Site coverages [%]")
    ax.tick_params(axis="x", rotation=90)
    # Customize legend.
    handles, labels = ax.get_legend_handles_labels()
    selected = [0, 2, 3, 6]
    if task == "extrapol":
        ax.legend(
            handles=[handles[ii] for ii in selected],
            labels=[labels[ii] for ii in selected],
            edgecolor="black",
            framealpha=1.,
            loc="center right",
            bbox_to_anchor=(1.15, 0.5)
        )
        plt.subplots_adjust(bottom=0.30, top=0.95, left=0.08, right=0.80)
    else:
        plt.subplots_adjust(bottom=0.30, top=0.95, left=0.08, right=0.89)
    # Save figure.
    dirname = f"images/reaction_{reaction}_{task}"
    os.makedirs(dirname, exist_ok=True)
    num = f"_{half}" if task == "extrapol" and half is not None else ""
    plt.savefig(f"{dirname}/reaction_coverages_{models[task]}{num}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------