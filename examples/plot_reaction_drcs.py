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
    
    drcs_dict = {
        "CO2+2*<=>CO2**": "black",
        "CO+*<=>CO*": "black",
        "H2O+*<=>H2O*": "black",
        "H2+2*<=>H*+H*": "mediumslateblue",
        "CO2**<=>CO*+O*": "mediumslateblue",
        "COOH**<=>CO*+OH*": "darkorange",
        "COOH**+*<=>CO2**+H*": "orangered",
        "HCOO**+*<=>CO2**+H*": "mediumseagreen",
        "HCOO**+*<=>HCO**+O*": "darkgreen",
        "H2O*+*<=>OH*+H*": "violet",
        "OH*+*<=>O*+H*": "darkorchid",
        "COOH**<=>COH*+O*": "black",
        "COH*+*<=>CO*+H*": "black",
        "HCO**<=>CO*+H*": "black",
    }
    
    # DFT data.
    drcs = [[] for _ in drcs_dict]
    surfaces = []
    for surface in results:
        for ii, path in enumerate(drcs_dict):
            drcs[ii].append(results[surface][None]["DRC"][path])
        surfaces.append(modify_name(surface))
    
    drcs = np.array(drcs)
    for ii in range(drcs.shape[1]):
        if abs(np.sum(drcs[:, ii])) > 1.5:
            drcs[:, ii] = 0.
        else:
            drcs[:, ii] = drcs[:, ii] / np.sum(drcs[:, ii]) * 100
    
    # Get half data for extrapolation task.
    if task == "extrapol":
        half_num = len(surfaces) // 2
        if half == 1:
            surfaces = surfaces[:half_num]
            drcs = drcs[:, :half_num]
        elif half == 2:
            surfaces = surfaces[half_num:]
            drcs = drcs[:, half_num:]
    
    # Degrees of rate control plot.
    fig, ax = plt.subplots(figsize=(13, 4), dpi=200)
    for ii, path in enumerate(drcs_dict):
        ax.bar(
            x=surfaces,
            height=drcs[ii],
            color=drcs_dict[path],
            label=modify_name(path, replace_dict={"<=>": " ⇌ "}),
            bottom=None if ii == 0 else np.sum(drcs[:ii], axis=0),
        )
    ax.set_xlim(-0.5, len(surfaces)-0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Degree of rate control [%]")
    ax.tick_params(axis="x", rotation=90)
    # Customize legend.
    handles, labels = ax.get_legend_handles_labels()
    selected = [4, 5, 6, 7, 8, 9, 10]
    if task == "extrapol":
        ax.legend(
            handles=[handles[ii] for ii in selected],
            labels=[labels[ii] for ii in selected],
            edgecolor="black",
            framealpha=1.,
            loc="center right",
            bbox_to_anchor=(1.26, 0.5)
        )
        plt.subplots_adjust(bottom=0.30, top=0.95, left=0.08, right=0.80)
    else:
        plt.subplots_adjust(bottom=0.30, top=0.95, left=0.08, right=0.89)
    # Save figure.
    dirname = f"images/reaction_{reaction}_{task}"
    os.makedirs(dirname, exist_ok=True)
    num = f"_{half}" if task == "extrapol" and half is not None else ""
    plt.savefig(f"{dirname}/reaction_drcs_{models[task]}{num}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------