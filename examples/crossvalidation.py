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
    species_type = "reactions" # adsorbates | reactions
    crossval_name = "StratifiedKFold" # StratifiedKFold | StratifiedGroupKFold
    key_groups = "material"
    key_stratify = "species"
    n_splits = 5
    random_state = 42
    most_stable = False
    ensemble = True
    # Model parameters.
    model_name = "SKLearn" # LSR | BEP | SKLearn | WWLGPR
    update_features = True
    model_name_ref = "SKLearn"
    target = "E_act" if species_type == "reactions" else "E_form"
    model_params_dict = {
        "LSR": {"keys_LSR": ["species"]},
        "BEP": {"keys_BEP": ["species"]},
        "SKLearn": {"target": target, "model": None, "hyperparams": None},
        "WWLGPR": {"target": target, "hyperparams": None},
    }
    add_ref_atoms = True
    species_ref = ["CO*", "H*", "O*"]
    
    from lightgbm import LGBMRegressor
    model_params_dict["SKLearn"] = {
        "model": LGBMRegressor(),
        "hyperparams": {
            "n_estimators": 169,
            "num_leaves": 8,
            "min_child_samples": 3,
            "learning_rate": 0.09237454406367815,
            "max_bin": int(np.exp(10)),
            "colsample_bytree": 0.6560437528345557,
            "reg_alpha": 0.026210047424520093,
            "reg_lambda": 0.12165458798539781,
            "verbose": -1,
        },
    }
    
    # Read Ase database.
    db_ase_name = f"atoms_{species_type}_DFT.db"
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
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
    if model_name == "LSR":
        from ase_ml_models.linear import lsr_prepare
        lsr_prepare(atoms_list=atoms_list+atoms_add, species_LSR=species_ref)
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list+atoms_add)
    # Prepare Ase database.
    db_model = connect(f"atoms_{species_type}_{model_name}.db", append=False)
    # Cross-validation.
    if ensemble is True:
        results = ensemble_crossvalidation(
            atoms_list=atoms_list,
            model_name=model_name,
            crossval=crossval,
            key_groups=key_groups,
            key_stratify=key_stratify,
            atoms_add=atoms_add,
            db_model=db_model,
            model_params=model_params_dict[model_name],
        )
    else:
        results = crossvalidation(
            atoms_list=atoms_list,
            model_name=model_name,
            crossval=crossval,
            key_groups=key_groups,
            key_stratify=key_stratify,
            atoms_add=atoms_add,
            db_model=db_model,
            model_params=model_params_dict[model_name],
        )
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    # Calculate the MAE and the RMSE.
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Average results:")
    print(f"TOT MAE:   {mae:7.4f} [eV]")
    print(f"TOT RMSE:  {rmse:7.4f} [eV]")
    # Parity plot.
    os.makedirs("images/plots", exist_ok=True)
    lims = [-2.2, +3.2] if species_type == "adsorbates" else [-1.4, +5.2]
    ax = parity_plot(results=results, lims=lims)
    plt.savefig(f"images/plots/parity_{species_type}_{model_name}.png")
    # Violin plot.
    ax = violin_plot(results=results)
    plt.savefig(f"images/plots/violin_{species_type}_{model_name}.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------