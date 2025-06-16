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

    # Parameters.
    reaction = "RWGS" # WGS | RWGS
    model = "WWLGPR+WWLGPR"

    # Read microkinetics results.
    yaml_results = f"results_database_{reaction}.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_DFT = results_all["DFT+DFT"]
    
    # Read microkinetics results extrapolation.
    yaml_results = f"results_extrapol_{reaction}.yaml"
    with open(yaml_results, 'r') as fileobj:
        results_all = yaml.safe_load(fileobj)
    results_extra = results_all[model]
    
    # Read facets fractions.
    yaml_facet_fracs = "facets_fractions.yaml"
    with open(yaml_facet_fracs, 'r') as fileobj:
        facet_fracs_dict = yaml.safe_load(fileobj)
    
    # DFT data.
    tofs_DFT = {}
    materials_DFT = []
    tof_fractions = {}
    for surface in results_DFT:
        # Material name.
        split = re.split(r"[()]", surface)
        material = split[0]+split[2]
        facet = split[1]
        matrix = material.split("+")[0]
        fract = facet_fracs_dict[matrix][facet]
        # TOF.
        tof = results_DFT[surface][None]["TOF"] * fract
        tofs_DFT[material] = tofs_DFT.get(material, 0.) + tof
        # TOF fractions.
        if material not in tof_fractions:
            tof_fractions[material] = {}
        tof_fractions[material][facet] = tof
    ltofs_DFT = []
    for material in tofs_DFT:
        ltofs_DFT.append(np.log10(tofs_DFT[material]))
        materials_DFT.append(modify_name(material))
    
    # Extrapolated data.
    tofs_list_extra = {}
    materials_extra = []
    for surface in results_extra:
        # Material name.
        split = re.split(r"[()]", surface)
        material = split[0]+split[2]
        facet = split[1]
        matrix = material.split("+")[0]
        fract = facet_fracs_dict[matrix][facet]
        # TOF list.
        tof_list = [
            results_extra[surface][ii]["TOF"] * fract for ii in range(5)
        ]
        if material in tofs_list_extra:
            tof_list = list(np.sum([tof_list, tofs_list_extra[material]], axis=0))
        tofs_list_extra[material] = tof_list
        # TOF fractions.
        if material not in tof_fractions:
            tof_fractions[material] = {}
        tof_fractions[material][facet] = np.mean(tof_list)
    ltofs_extra = []
    lstds_extra = []
    for material, tof_list in tofs_list_extra.items():
        ltof_list = [np.log10(tof) for tof in tof_list]
        ltofs_extra.append(np.mean(ltof_list))
        lstds_extra.append(np.std(ltof_list))
        materials_extra.append(modify_name(material))
    
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
        yerr=np.array(lstds_extra).T,
        color="crimson",
        alpha=0.7,
        label="extrapolation",
        bottom=-14,
        capsize=5,
    )
    ax.set_xlim(-0.5, len(materials_DFT+materials_extra)-0.5)
    ax.set_ylim(-14, 2)
    ax.set_ylabel("log$_{10}$(TOF [1/s])")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(edgecolor="black", loc="lower right", framealpha=1.)
    plt.subplots_adjust(bottom=0.25, top=0.95, left=0.10, right=0.95)
    dirname = f"images/reaction_{reaction}_extrapol"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/materials_TOF_extra_{model}.png", dpi=300)
    
    # TOF fractions.
    facet_list = ["111", "100"]
    colors = ["mediumslateblue", "darkorange"]
    materials = [modify_name(material) for material in tof_fractions.keys()]
    fractions = {}
    for facet in facet_list:
        fractions[facet] = []
        for material in tof_fractions:
            denom = sum([tof_fractions[material][facet] for facet in facet_list])
            fractions[facet].append(tof_fractions[material][facet] / denom * 100)
    # TOF fractions plot.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for ii, fraction in enumerate(fractions.values()):
        ax.bar(
            x=materials,
            height=fraction,
            bottom=None if ii == 0 else fractions["111"],
            color=colors[ii],
            label=f"({list(fractions.keys())[ii]})",
        )
    ax.set_xlim(-0.5, len(materials) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Facet TOF contribution [%]")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(
        edgecolor="black",
        framealpha=1.,
        loc="upper left",
    )
    # Save plot.
    plt.tight_layout()
    plt.savefig(f"{dirname}/facets_TOF_contributions_{model}.png", dpi=300)
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------