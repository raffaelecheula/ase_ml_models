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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db, get_atoms_most_stable
from ase_ml_models.linear import (
    get_correlation_heatmap,
    tsr_prepare,
    get_tsr_data_dict,
    get_tsr_models_dict,
    tsr_predict,
)
from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database.
    db_ase_name = "atoms_adsorbates_DFT.db"
    most_stable = True
    material_labels = False
    time_lim = 100
    get_heatmap = False
    
    # Read Ase database.
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    
    ## Get the most stable structures.
    #keys_most_stable = ["species", "surface", "site"]
    #atoms_list = get_atoms_most_stable(
    #    atoms_list=atoms_list,
    #    keys_most_stable=keys_most_stable,
    #)
    
    # Colors for the different classes.
    colors_dict = {
        "100 metal": "crimson",
        "111 metal": "orange",
        "100 sa-alloy": "darkcyan",
        "111 sa-alloy": "lightgreen",
    }

    # Dictionary for Thermochemical Scaling Relations.
    species_TSR = ["CO*", "H*", "O*"]
    fixed_TSR = {
        "CO2*": ["CO*"],
        "COH*": ["CO*"],
        "cCOOH*": ["CO*"],
        "H2O*": ["CO*"],
        "HCO*": ["CO*"],
        "HCOO*": ["O*"],
        "OH*": ["O*"],
    }

    # Prepare the data for Thermochemical Scaling Relations.
    tsr_prepare(atoms_list=atoms_list, species_TSR=species_TSR, fixed_TSR=fixed_TSR)
    # Get the heatmap.
    os.makedirs("images/TSR", exist_ok=True)
    if get_heatmap is True:
        ax = get_correlation_heatmap(atoms_list=atoms_list)
        plt.subplots_adjust(left=0.15, right=0.90, bottom=0.20, top=0.90)
        plt.savefig(f"images/TSR/correlation_heatmap.png", dpi=300)
    # Get the data for the TSR relations.
    tsr_data_all_dict = get_tsr_data_dict(atoms_train=atoms_list)
    models_all_dict = get_tsr_models_dict(tsr_data_dict=tsr_data_all_dict)
    # Plot the data.
    for species in fixed_TSR:
        # Get atoms objects for the species.
        atoms_spec = [
            atoms for atoms in atoms_list if atoms.info["species"] == species
        ]
        # Get the data for the TSR.
        tsr_data_dict = get_tsr_data_dict(
            atoms_train=atoms_spec,
            keys_TSR=["miller_index", "material_type"],
        )
        # Get the TSR model.
        models_dict = get_tsr_models_dict(tsr_data_dict=tsr_data_dict)
        # Prepare the plot.
        if material_labels is True:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
            plt.subplots_adjust(left=0.20, right=0.90, bottom=0.20, top=0.90)
        title = modify_name(species, replace_dict={})
        ax.set_title(title, fontdict={"fontsize": 20})
        species_x = modify_name(fixed_TSR[species][0], replace_dict={})
        species_y = modify_name(species, replace_dict={})
        ax.set_xlabel("E$_{form}$ ("+species_x+") [eV]", fontdict={"fontsize": 16})
        ax.set_ylabel("E$_{form}$ ("+species_y+") [eV]", fontdict={"fontsize": 16})
        ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        # Set plot limits.
        e_tsr_all_list = [ee[0] for ee in tsr_data_all_dict[species]["E_TSR"]]
        e_form_all_list = tsr_data_all_dict[species]["E_form"]
        surface_all_list = tsr_data_all_dict[species]["surface"]
        ax.set_xlim(min(e_tsr_all_list)-0.5, max(e_tsr_all_list)+0.5)
        ax.set_ylim(min(e_form_all_list)-0.5, max(e_form_all_list)+0.5)
        #base = 0.5 if material_labels is True else 1.0
        #ax.xaxis.set_major_locator(MultipleLocator(base=base))
        #ax.yaxis.set_major_locator(MultipleLocator(base=base))
        # Plot the results.
        texts = []
        for ii, key in enumerate(models_dict):
            e_form_list = tsr_data_dict[key]["E_form"]
            e_tsr_list = [ee[0] for ee in tsr_data_dict[key]["E_TSR"]]
            surface_list = tsr_data_dict[key]["surface"]
            points = ax.scatter(
                x=e_tsr_list,
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
        y_test = [atoms.info["E_form"] for atoms in atoms_spec]
        y_pred = tsr_predict(
            atoms_test=atoms_spec,
            models_dict=models_all_dict,
        )
        # Calculate the MAE and the RMSE.
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Species {species}")
        print(f"TOT MAE:   {mae:7.4f} [eV]")
        print(f"TOT RMSE:  {rmse:7.4f} [eV]")
        
        if material_labels is True:
            # Add text.
            for xx, yy, name in zip(e_tsr_all_list, e_form_all_list, surface_all_list):
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
        plt.savefig(f"images/TSR/{name}.png", dpi=300)
    
# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------