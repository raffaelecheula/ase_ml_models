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
from ase_ml_models.linear import get_bep_data_dict, get_bep_models_dict
from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database.
    db_ase_name = "atoms_reactions_DFT.db"
    most_stable = True
    material_labels = False
    time_lim = 100
    
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

    # List of species to plot.
    species_list = [
        'CO2*→CO*+O*',
        'COH*→CO*+H*',
        'H2O*→OH*+H*',
        'H2*→H*+H*',
        'HCOO*→CO2*+H*',
        'HCOO*→HCO*+O*',
        'HCO*→CO*+H*',
        'OH*→O*+H*',
        'cCOOH*→CO*+OH*',
        'cCOOH*→COH*+O*',
        'tCOOH*→CO2*+H*'
    ]
    
    # Get the data for the BEP relations.
    bep_data_all_dict = get_bep_data_dict(atoms_train=atoms_list)
    models_all_dict = get_bep_models_dict(bep_data_dict=bep_data_all_dict)
    # Plot the data.
    os.makedirs("images/BEP", exist_ok=True)
    for species in species_list:
        # Get atoms objects for the species.
        atoms_spec = [
            atoms for atoms in atoms_list if atoms.info["species"] == species
        ]
        # Get the data for the BEP.
        bep_data_dict = get_bep_data_dict(
            atoms_train=atoms_spec,
            keys_BEP=["miller_index", "material_type"],
        )
        # Get BEP model.
        models_dict = get_bep_models_dict(bep_data_dict=bep_data_dict)
        # Prepare the plot.
        if material_labels is True:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            plt.subplots_adjust(left=0.20, right=0.90, bottom=0.20, top=0.90)
        title = modify_name(species, replace_dict={})
        ax.set_title(title, fontdict={"fontsize": 20})
        ax.set_xlabel("ΔE$_{react}$ [eV]", fontdict={"fontsize": 16})
        ax.set_ylabel("E$_{act}$ [eV]", fontdict={"fontsize": 16})
        ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
        # Set plot limits.
        deltae_all_list = bep_data_all_dict[species]["ΔE_react"]
        e_act_all_list = bep_data_all_dict[species]["E_act"]
        surface_all_list = bep_data_all_dict[species]["surface"]
        ax.set_xlim(min(deltae_all_list)-0.5, max(deltae_all_list)+0.5)
        ax.set_ylim(min(e_act_all_list)-0.5, max(e_act_all_list)+0.5)
        #base = 1.0 if material_labels is False and species == "CO2*→CO*+O*" else 0.5
        #ax.xaxis.set_major_locator(MultipleLocator(base=base))
        #ax.yaxis.set_major_locator(MultipleLocator(base=base))
        # Plot the results.
        texts = []
        for ii, key in enumerate(models_dict):
            # Plot the BEP data.
            deltae_list = bep_data_dict[key]["ΔE_react"]
            e_act_list = bep_data_dict[key]["E_act"]
            points = ax.scatter(
                x=deltae_list,
                y=e_act_list,
                s=100,
                edgecolors='black',
                label=key,
                color=colors_dict[key],
                zorder=2,
            )
            # BEP regression for metals.
            if "metal" in key:
                intercept = models_dict[key].intercept_
                slope = models_dict[key].coef_[0]
                line = plt.plot(
                    [-20, +20],
                    [intercept-20*slope, intercept+20*slope],
                    linewidth=4,
                    alpha=0.5,
                    color=colors_dict[key],
                    zorder=0,
                )
        if material_labels is True:
            # Add text.
            for xx, yy, name in zip(deltae_all_list, e_act_all_list, surface_all_list):
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
        plt.savefig(f"images/BEP/{name}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------