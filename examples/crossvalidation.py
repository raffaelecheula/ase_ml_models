# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.workflow import (
    update_ts_atoms,
    get_atoms_ref,
    get_crossval,
    crossvalidation,
    ensemble_crossvalidation,
    parity_plot,
    violin_plot,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    species_type = "adsorbates" # adsorbates | reactions
    crossval_name = "StratifiedGroupKFold" # StratifiedKFold | StratifiedGroupKFold
    key_groups = "surface" # surface | bulk_elements
    key_stratify = "species"
    n_splits = 6
    random_state = 42
    most_stable = False
    ensemble = False
    store_data = False
    add_ref_atoms = True
    exclude_add = True
    # Model selection.
    model_name = "WWLGPR" # Linear | SKLearn | WWLGPR
    model_sklearn = "LightGBM" # RandomForest | XGBoost | LightGBM
    update_features = True
    model_name_ref = model_name[:]
    # Model parameters.
    target = "E_act" if species_type == "reactions" else "E_form"
    if model_name == "Linear":
        model_name = "TSR" if species_type == "adsorbates" else "BEP"
    if model_name_ref == "Linear":
        model_name_ref = "TSR" if species_type == "adsorbates" else "BEP"
    model_params_dict = {
        "TSR": {"keys_TSR": ["species"] if most_stable else ["species", "site"]},
        "BEP": {"keys_BEP": ["species", "miller_index"]},
        "SKLearn": {"target": target, "model": None, "hyperparams": None},
        "WWLGPR": {"target": target, "hyperparams": None},
    }
    species_ref = ["CO*", "H*", "O*"]
    fixed_TSR = {
        "CO2*": ["CO*"],
        "COH*": ["CO*"],
        "cCOOH*": ["CO*"],
        "H2O*": ["CO*"],
        "HCO*": ["CO*"],
        "HCOO*": ["O*"],
        "OH*": ["O*"],
    }
    color_dict = {
        "TSR": "darkcyan",
        "BEP": "darkcyan",
        "SKLearn": "orchid",
        "WWLGPR": "crimson",
    }
    # Model hyperparameters.
    model_params = model_params_dict[model_name]
    # Get optimized hyperparameters.
    from optimized_hyperparams import get_optimized_hyperparams
    model_params = get_optimized_hyperparams(
        model_params=model_params,
        model_name=model_name,
        model_sklearn=model_sklearn,
        species_type=species_type,
    )
    
    # Read Ase database.
    db_ase_name = f"atoms_{species_type}_DFT_bulk.db"
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    for atoms in atoms_list:
        atoms.info["E_form_gas"] = atoms.info["E_form"]-atoms.info["E_bind"]
    # Update features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"atoms_adsorbates_{model_name_ref}.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_list, db_ads=db_ads)
    # Initialize cross-validation.
    crossval = get_crossval(
        crossval_name=crossval_name,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        atoms_add = get_atoms_ref(atoms_list=atoms_list, species_ref=species_ref)
    else:
        atoms_add = []
    # Preprocess the data.
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_prepare
        tsr_prepare(
            atoms_list=atoms_list+atoms_add,
            species_TSR=species_ref,
            fixed_TSR=fixed_TSR,
        )
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list+atoms_add)
    # Prepare Ase database.
    db_model_name = f"atoms_{species_type}_{model_name}.db"
    db_model = connect(db_model_name, append=False) if store_data else None
    # Cross-validation.
    if ensemble is True:
        results = ensemble_crossvalidation(
            atoms_list=atoms_list,
            model_name=model_name,
            crossval=crossval,
            key_groups=key_groups,
            key_stratify=key_stratify,
            atoms_add=atoms_add,
            exclude_add=exclude_add,
            db_model=db_model,
            model_params=model_params,
        )
    else:
        results = crossvalidation(
            atoms_list=atoms_list,
            model_name=model_name,
            crossval=crossval,
            key_groups=key_groups,
            key_stratify=key_stratify,
            atoms_add=atoms_add,
            exclude_add=exclude_add,
            db_model=db_model,
            model_params=model_params,
        )
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    # Calculate the MAE and the RMSE.
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Average results:")
    print(f"TOT MAE:  {mae:7.3f} [eV]")
    print(f"TOT RMSE: {rmse:7.3f} [eV]")
    # Parity plot.
    model = model_sklearn if model_name == "SKLearn" else model_name
    model_ref = model_sklearn if model_name_ref == "SKLearn" else model_name_ref
    if species_type == "reactions" and update_features is True:
        model = f"{model}_from_{model_ref}"
    task = "stable" if most_stable is True else "all"
    os.makedirs("images/plots", exist_ok=True)
    lims = [-2.2, +2.8] if species_type == "adsorbates" else [-1.4, +5.2]
    ax = parity_plot(results=results, lims=lims, color=color_dict[model_name])
    plt.savefig(f"images/plots/parity_{species_type}_{model}_{task}.png")
    # Violin plot.
    ax = violin_plot(results=results, color=color_dict[model_name])
    plt.savefig(f"images/plots/violin_{species_type}_{model}_{task}.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------