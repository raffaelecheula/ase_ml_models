# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.utilities import modify_name

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Parameters.
    task = "database" # database | extrapol
    models = {
        "database": "DFT+DFT",
        "extrapol": "WWLGPR+WWLGPR",
    }

    # Ase database names.
    model_ads, model_ts = models[task].split("+")
    db_ads_name = f"databases/atoms_adsorbates_{model_ads}_{task}.db"
    db_ts_name = f"databases/atoms_reactions_{model_ts}_{task}.db"
    
    # Get materials and Miller indices.
    with open("materials.yaml", 'r') as fileobj:
        data = yaml.safe_load(fileobj)
    miller_index_list = data["miller_indices"]
    material_list = data[f"materials_{task}"]
    
    # Reaction mechanism parameters.
    yaml_file = 'mechanism.yaml'
    
    # Read the reaction mechanism.
    with open(yaml_file, 'r') as fileobj:
        mechanism = yaml.safe_load(fileobj)

    adsorbates_list = [
        species["name"] for species in mechanism["species-adsorbates"]
        if species["name"] != "(Rh)"
    ]
    reactions_list = [
        species["name"] for species in mechanism["species-reactions"]
        if species["name"] != "(Rh)"
        if not species["thermo"].get("sticking", False)
    ]

    # Read Ase databases.
    db_ads = connect(db_ads_name)
    db_ts = connect(db_ts_name)
    atoms_ads_list = get_atoms_list_from_db(db_ase=db_ads)
    atoms_ts_list = get_atoms_list_from_db(db_ase=db_ts)
    
    # Colors.
    color_dict = {"100": "lightgreen", "111": "orange"}
    replace_dict = {"(Rh)": "*", "(Rh,Rh)": "**", "<=>": "→"}
    
    # Get formation energies.
    energy_ads_dict = {}
    energy_ts_dict = {}
    for material in material_list:
        for miller_index in miller_index_list:
            # Adsorbates.
            for species_cantera in adsorbates_list:
                atoms_list = [
                    atoms for atoms in atoms_ads_list
                    if atoms.info["material"] == material
                    if atoms.info["miller_index"] == miller_index
                    if atoms.info["species_cantera"] == species_cantera
                ]
                if len(atoms_list) == 0:
                    print("Not found:", material, miller_index, species_cantera)
                    continue
                atoms = sorted(atoms_list, key=lambda atoms: atoms.info["E_form"])[0]
                if species_cantera not in energy_ads_dict:
                    energy_ads_dict[species_cantera] = {}
                if miller_index not in energy_ads_dict[species_cantera]:
                    energy_ads_dict[species_cantera][miller_index] = {}
                energy_ads_dict[species_cantera][miller_index][material] = (
                    atoms.info["E_form"]
                )
            # Reactions.
            for species_cantera in reactions_list:
                atoms_list = [
                    atoms for atoms in atoms_ts_list
                    if atoms.info["material"] == material
                    if atoms.info["miller_index"] == miller_index
                    if atoms.info["species_cantera"] == species_cantera
                ]
                if len(atoms_list) == 0:
                    print("Not found:", material, miller_index, species_cantera)
                    continue
                atoms = sorted(atoms_list, key=lambda atoms: atoms.info["E_form"])[0]
                if species_cantera not in energy_ts_dict:
                    energy_ts_dict[species_cantera] = {}
                if miller_index not in energy_ts_dict[species_cantera]:
                    energy_ts_dict[species_cantera][miller_index] = {}
                energy_ts_dict[species_cantera][miller_index][material] = (
                    atoms.info["E_form"]
                )

    # Formation energies plot.
    figsize = (len(material_list)+2, 18)
    for ii, energy_spec_dict in enumerate([energy_ads_dict, energy_ts_dict]):
        fig, axes = plt.subplots(ncols=4, nrows=3, figsize=figsize, dpi=150)
        axes = axes.flatten()
        for jj, (species, energy_dict_par) in enumerate(energy_spec_dict.items()):
            ax = axes[jj]
            y_text = {}
            for miller_index, energy_dict in energy_dict_par.items():
                y_vect = [
                    energy_dict[material] if material in energy_dict else None
                    for material in material_list
                ]
                ax.scatter(
                    x=material_list,
                    y=y_vect,
                    facecolors=color_dict[miller_index],
                    edgecolors='black',
                    s=50,
                    label=miller_index,
                )
                for material, yy in zip(material_list, y_vect):
                    if yy is None:
                        continue
                    if material not in y_text:
                        y_text[material] = []
                    y_text[material].append(yy)
            for material, y_list in y_text.items():
                if max(y_list) > 3.:
                    yy = min(y_list) - 0.30
                    va = "top"
                else:
                    yy = max(y_list) + 0.30
                    va = "bottom"
                ax.text(
                    x=material,
                    y=yy,
                    s=modify_name(material),
                    ha="center",
                    va=va,
                    rotation=90,
                    fontsize=10,
                )
            ax.set_title(modify_name(species, replace_dict=replace_dict))
            ax.set_xlim(-0.5, len(material_list)-0.5)
            ax.set_ylim(-3.0, +5.0)
            ax.set_ylabel("Formation energy [eV]")
            ax.get_xaxis().set_visible(False)
            #ax.legend(
            #    edgecolor="black",
            #    framealpha=1.,
            #    loc="lower left",
            #    bbox_to_anchor=(0.01, 0.03),
            #)
            ax.set_position(np.array(ax.get_position().bounds)*[1.0, 1.0, 0.95, 0.95])
        for ax in axes[jj+1:]:
            ax.axis("off")
        os.makedirs("images/mechanism_energies", exist_ok=True)
        species = "adsorbates" if ii == 0 else "reactions"
        model = model_ads if ii == 0 else model_ts
        filename = f"energies_{species}_{model}_{task}"
        plt.savefig(f"images/mechanism_energies/{filename}.png", dpi=300)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------