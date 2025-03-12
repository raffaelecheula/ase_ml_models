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

    # Read microkinetics results.
    yaml_results = "results.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_DFT = results_all["DFT+DFT"]
    
    path_dict = {
        "CO-O path": "mediumslateblue",
        "COO-H path": "darkorange",
        "H-COO path": "mediumseagreen",
    }
    
    # DFT data.
    paths_DFT = [[] for _ in path_dict]
    surfaces_DFT = []
    for surface in results_DFT:
        for ii, path in enumerate(path_dict):
            paths_DFT[ii].append(results_DFT[surface][None]["paths"][path] * 100)
        surfaces_DFT.append(modify_name(surface, replace_dict={}))
    
    # Turnover frequencies plot.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for ii, path in enumerate(path_dict):
        ax.bar(
            x=surfaces_DFT,
            height=paths_DFT[ii],
            color=path_dict[path],
            label=path,
            bottom=None if ii == 0 else np.sum(paths_DFT[:ii], axis=0),
        )
    ax.set_xlim(-0.5, len(surfaces_DFT)-0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("R$_{path}$ [%]")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(
        edgecolor="black",
        framealpha=1.,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.03),
    )
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.10, right=0.95)
    os.makedirs("images/TOF", exist_ok=True)
    plt.savefig(f"images/TOF/materials_RPA.png", dpi=300)
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------