# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Model.
    model = "SKLearn+SKLearn"

    # Read microkinetics results.
    yaml_results = "results.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_DFT = results_all["DFT+DFT"]
    
    # Read microkinetics results extrapolation.
    yaml_results = "results_extra.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_extra = results_all[model]
    
    facet_fracs = {
        "100": 0.34,
        "111": 0.66,
    }
    
    # DFT data.
    tofs_DFT = {}
    materials_DFT = []
    for surface in results_DFT:
        # Material name.
        split = re.split(r"[()]", surface)
        material = split[0]+split[2]
        facet = split[1]
        # TOF.
        tof = results_DFT[surface][None]["CO TOF"]*facet_fracs[facet]
        tofs_DFT[material] = tofs_DFT.get(material, 0.) + tof
    ltofs_DFT = []
    for material in tofs_DFT:
        ltofs_DFT.append(np.log10(tofs_DFT[material]))
        materials_DFT.append(modify_name(material, replace_dict={}))
    
    # Extrapolated data.
    tofs_list_extra = {}
    materials_extra = []
    for surface in results_extra:
        # Material name.
        split = re.split(r"[()]", surface)
        material = split[0]+split[2]
        facet = split[1]
        # TOF list.
        tof_list = [
            results_extra[surface][ii]["CO TOF"]*facet_fracs[facet] for ii in range(5)
        ]
        if material in tofs_list_extra:
            tof_list = list(np.sum([tof_list, tofs_list_extra[material]], axis=0))
        tofs_list_extra[material] = tof_list
    ltofs_extra = []
    lstds_extra = []
    for material, tof_list in tofs_list_extra.items():
        ltof_list = [np.log10(tof) for tof in tof_list]
        ltofs_extra.append(np.mean(ltof_list))
        lstds_extra.append(np.std(ltof_list))
        materials_extra.append(modify_name(material, replace_dict={}))
    
    # Turnover frequencies plot.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(
        x=materials_DFT,
        height=np.array(ltofs_DFT)+14,
        color="orange",
        label="DFT calculations",
        bottom=-14,
    )
    ax.bar(
        x=materials_extra,
        height=np.array(ltofs_extra)+14,
        color="crimson",
        alpha=0.7,
        label="extrapolation",
        bottom=-14,
    )
    ax.errorbar(
        x=materials_extra,
        y=ltofs_extra,
        yerr=np.array(lstds_extra).T * 2,
        fmt=" ",
        color="black",
        capsize=5,
    )
    ax.set_xlim(-0.5, len(materials_DFT+materials_extra)-0.5)
    ax.set_ylabel("log$_{10}$(TOF [1/s])")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(edgecolor="black", loc="lower right", framealpha=1.)
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.10, right=0.95)
    os.makedirs("images/TOF", exist_ok=True)
    plt.savefig(f"images/TOF/materials_TOF_extra_{model}.png", dpi=300)
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------