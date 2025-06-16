# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
from sklearn.linear_model import LinearRegression

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.linear import get_bep_data_dict, get_bep_models_dict
from ase_ml_models.workflow import update_ts_atoms
from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database.
    db_ase_name = "databases/atoms_reactions_DFT_database.db"
    most_stable = True
    material_labels = False
    time_lim = 100
    legend = False
    update_features = False
    
    # Read Ase database.
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)

    # Update features from an Ase database.
    if update_features:
        db_ads_name = "databases/atoms_adsorbates_DFT_database.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_list, db_ads=db_ads)

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
        #'HCOOH*→HCOO*+H*',
        #'HCOOH*→cCOOH*+H*',
    ]
    
    # Get the data for the BEP relations.
    bep_data_all_dict = get_bep_data_dict(atoms_train=atoms_list)
    models_all_dict = get_bep_models_dict(bep_data_dict=bep_data_all_dict)
    # Plot the data.
    label_key = "label" if material_labels is True else "nolabel"
    os.makedirs(f"images/linear_relations/BEP_{label_key}", exist_ok=True)
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
            fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
        else:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            plt.subplots_adjust(left=0.20, right=0.90, bottom=0.20, top=0.90)
        title = modify_name(species)
        ax.set_title(title, fontdict={"fontsize": 20})
        ax.set_xlabel("ΔE$_{react}$ [eV]", fontdict={"fontsize": 16})
        ax.set_ylabel("E$_{act}$ [eV]", fontdict={"fontsize": 16})
        ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
        # Set plot limits.
        deltae_all_list = bep_data_all_dict[species]["ΔE_react"]
        e_act_all_list = bep_data_all_dict[species]["E_act"]
        surface_all_list = bep_data_all_dict[species]["surface"]
        xlims = [min(deltae_all_list)-0.5, max(deltae_all_list)+0.5]
        ylims = [min(e_act_all_list)-0.5, max(e_act_all_list)+0.5]
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        base = 0.5 if max(deltae_all_list)-min(deltae_all_list) < 3.0 else 1.0
        ax.xaxis.set_major_locator(MultipleLocator(base=base))
        ax.yaxis.set_major_locator(MultipleLocator(base=base))
        # Plot the results.
        texts = []
        points = []
        lines = []
        for ii, key in enumerate(colors_dict):
            # Label.
            facet, material = key.split(" ")
            material_2 = "SAA" if "sa-alloy" in material else "PM"
            label = f"{material_2} ({facet})"
            # Plot the BEP data.
            deltae_list = bep_data_dict[key]["ΔE_react"]
            e_act_list = bep_data_dict[key]["E_act"]
            point = ax.scatter(
                x=deltae_list,
                y=e_act_list,
                s=100,
                edgecolors='black',
                label=label,
                color=colors_dict[key],
                zorder=2,
            )
            points.append(point)
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
                    label=f"{label} BEP",
                )
                lines.append(line)
                text = "$E_{act} = "+f"{intercept:+5.3f}{slope:+5.3f}"+"ΔE_{react}$"
                ax.text(
                    x=xlims[0]+(xlims[1]-xlims[0])*0.05,
                    y=ylims[1]-(ylims[1]-ylims[0])*0.05*(ii+1.5),
                    s=text,
                    fontsize=14 if material_labels is True else 9,
                    color=colors_dict[key],
                    ha='left',
                    va='bottom',
                    zorder=2,
                )
        # Plot Eact = 0 line.
        #ax.plot([-10, +10], [-10, +10], color="grey", linestyle="--", zorder=0)
        if material_labels is True:
            # Add text.
            for xx, yy, name in zip(deltae_all_list, e_act_all_list, surface_all_list):
                text = ax.text(
                    x=xx,
                    y=yy,
                    s=modify_name(name),
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=3,
                )
                texts.append(text)
            # Adjust text.
            texts, patches = adjust_text(
                texts=texts,
                expand=(1.8, 2.0),
                arrowprops={"arrowstyle": '-', "color": 'grey', "alpha": 0.5},
                prevent_crossings=False,
                zorder=1,
                time_lim=time_lim,
            )
        # Save the plot.
        name = species.replace("*", "")
        if legend is True:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0,2,4,5,1,3]
            ax.legend(
                [handles[ii] for ii in order],
                [labels[ii] for ii in order],
                fontsize=12 if material_labels is True else 9,
                loc="lower right",
                edgecolor="black",
            )
        plt.savefig(f"images/linear_relations/BEP_{label_key}/{name}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------