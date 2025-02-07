# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase.db import connect
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
from sklearn.linear_model import LinearRegression

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.linear import (
    get_correlation_heatmap,
    lsr_prepare,
    get_lsr_data_dict,
    get_lsr_models_dict,
)
from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database.
    db_ase_name = "atoms_adsorbates_DFT.db"
    most_stable = True
    material_labels = True
    time_lim = 100
    get_heatmap = False
    
    # Read Ase database.
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)

    # Colors for the different classes.
    colors_dict = {
        "100 metal": "crimson",
        "111 metal": "orange",
        "100 sa-alloy": "darkcyan",
        "111 sa-alloy": "lightgreen",
    }

    # Dictionary for Linear Scaling Relations.
    species_LSR = ["CO*", "H*", "O*"]
    fixed_LSR = {
        "CO2*": ["CO*"],
        "COH*": ["CO*"],
        "cCOOH*": ["CO*"],
        "H2O*": ["CO*"],
        "HCO*": ["CO*"],
        "HCOO*": ["O*"],
        "OH*": ["O*"],
    }

    # Prepare the data for Linear Scaling Relations.
    lsr_prepare(atoms_list=atoms_list, species_LSR=species_LSR, fixed_LSR=fixed_LSR)
    # Get the heatmap.
    os.makedirs("images/LSR", exist_ok=True)
    if get_heatmap is True:
        ax = get_correlation_heatmap(atoms_list=atoms_list)
        plt.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.20)
        plt.savefig(f"images/LSR/correlation_heatmap.png", dpi=300)
    # Get the data for the LSR relations.
    lsr_data_all_dict = get_lsr_data_dict(atoms_train=atoms_list)
    models_all_dict = get_lsr_models_dict(lsr_data_dict=lsr_data_all_dict)
    # Plot the data.
    for species in fixed_LSR:
        # Get atoms objects for the species.
        atoms_spec = [
            atoms for atoms in atoms_list if atoms.info["species"] == species
        ]
        # Get the data for the LSR.
        lsr_data_dict = get_lsr_data_dict(
            atoms_train=atoms_spec,
            keys_LSR=["miller_index", "material_type"],
        )
        # Get the LSR model.
        models_dict = get_lsr_models_dict(lsr_data_dict=lsr_data_dict)
        # Prepare the plot.
        fig, ax = plt.subplots(figsize=(8, 8))
        title = modify_name(species, replace_dict={})
        ax.set_title(title, fontdict={"fontsize": 20})
        species_x = modify_name(fixed_LSR[species][0], replace_dict={})
        species_y = modify_name(species, replace_dict={})
        ax.set_xlabel("E$_{form}$ ("+species_x+") [eV]", fontdict={"fontsize": 16})
        ax.set_ylabel("E$_{form}$ ("+species_y+") [eV]", fontdict={"fontsize": 16})
        ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # Set plot limits.
        e_lsr_all_list = [ee[0] for ee in lsr_data_all_dict[species]["E_LSR"]]
        e_form_all_list = lsr_data_all_dict[species]["E_form"]
        surface_all_list = lsr_data_all_dict[species]["surface"]
        ax.set_xlim(min(e_lsr_all_list)-0.5, max(e_lsr_all_list)+0.5)
        ax.set_ylim(min(e_form_all_list)-0.5, max(e_form_all_list)+0.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        # Plot the results.
        texts = []
        for ii, key in enumerate(models_dict):
            e_form_list = lsr_data_dict[key]["E_form"]
            e_lsr_list = [ee[0] for ee in lsr_data_dict[key]["E_LSR"]]
            surface_list = lsr_data_dict[key]["surface"]
            points = ax.scatter(
                x=e_lsr_list,
                y=e_form_list,
                s=100,
                edgecolors='black',
                label=key,
                color=colors_dict[key],
                zorder=2,
            )
        intercept = models_all_dict[species].intercept_
        slope = models_all_dict[species].coef_[0]
        line = plt.plot(
            [-20, +20],
            [intercept-20*slope, intercept+20*slope],
            linewidth=4,
            alpha=0.5,
            color="grey",
            zorder=0,
        )
        if material_labels is True:
            # Add text.
            for xx, yy, name in zip(e_lsr_all_list, e_form_all_list, surface_all_list):
                text = ax.text(
                    x=xx,
                    y=yy,
                    s=modify_name(name, replace_dict={}),
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=3,
                )
                texts.append(text)
            # Adjust text.
            texts, patches = adjust_text(
                texts=texts,
                expand=(1.3, 1.9),
                arrowprops={"arrowstyle": '-', "color": 'grey', "alpha": 0.5},
                prevent_crossings=False,
                zorder=1,
                time_lim=time_lim,
            )
        # Save the plot.
        name = species.replace("*", "")
        plt.savefig(f"images/LSR/{name}.png", dpi=300)
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------