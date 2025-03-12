# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db, write_atoms_to_db
from ase_ml_models.workflow import (
    update_ts_atoms,
    get_crossval,
    train_model_and_predict,
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
    ensemble = True
    add_ref_atoms = True
    # Model selection.
    model_name = "SKLearn" # Linear | SKLearn | WWLGPR
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
    
    # Read Ase train database.
    db_train_name = f"atoms_{species_type}_DFT_bulk.db"
    db_train = connect(db_train_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_train, **kwargs)
    # Read Ase extra database.
    db_extra_name = f"atoms_{species_type}_DFT_extra_empty.db"
    db_extra = connect(db_extra_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_extra = get_atoms_list_from_db(db_ase=db_extra, **kwargs)
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        db_add_name = f"atoms_{species_type}_DFT_extra_add.db"
        db_add = connect(db_add_name)
        kwargs = {"most_stable": True} if most_stable is True else {}
        atoms_add = get_atoms_list_from_db(db_ase=db_add, **kwargs)
    else:
        atoms_add = []
    
    # Update features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"atoms_adsorbates_{model_name_ref}_extra.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_extra, db_ads=db_ads)
    # Initialize cross-validation.
    crossval = get_crossval(
        crossval_name=crossval_name,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Preprocess the data.
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_prepare
        tsr_prepare(
            atoms_list=atoms_list+atoms_add+atoms_extra,
            species_TSR=species_ref,
            fixed_TSR=fixed_TSR,
        )
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list+atoms_add+atoms_extra)

    # Print number of atoms.
    print(f"n train: {len(atoms_list)}")
    print(f"n extra: {len(atoms_extra)}")
    print(f"n added: {len(atoms_add)}")
    # Prepare Ase database.
    db_model = connect(f"atoms_{species_type}_{model_name}_extra.db", append=False)
    # Extrapolation.
    if ensemble is True:
        indices = list(range(len(atoms_list)))
        groups = [atoms.info[key_groups] for atoms in atoms_list]
        stratify = [atoms.info[key_stratify] for atoms in atoms_list]
        y_pred_list = []
        for jj, (indices_train, indices_test) in enumerate(
            crossval.split(X=indices, y=stratify, groups=groups)
        ):
            # Split the data.
            atoms_train = list(np.array(atoms_list, dtype=object)[indices_train])
            # Add reference atoms to the train sets.
            atoms_train += atoms_add
            # Train the model and do predictions.
            results = train_model_and_predict(
                model_name=model_name,
                atoms_train=atoms_train,
                atoms_test=atoms_extra,
                model_params=model_params,
            )
            y_pred_list.append(results["y_pred"])
        y_pred = list(np.mean(y_pred_list, axis=0))
        y_std = list(np.std(y_pred_list, axis=0))
        # Store the results in the ase database.
        y_pred_array = np.array(y_pred_list)
        add_indices = []
        for kk, atoms in enumerate(atoms_extra):
            e_form = y_pred[kk]
            e_form_list = y_pred_array[:, kk]
            atoms_copy = atoms.copy()
            atoms_copy.info = atoms.info.copy()
            atoms_copy.info["E_form"] = e_form
            atoms_copy.info["E_form_list"] = e_form_list
            write_atoms_to_db(atoms=atoms_copy, db_ase=db_model)
    else:
        atoms_list += atoms_add
        results = train_model_and_predict(
            model_name=model_name,
            atoms_train=atoms_list,
            atoms_test=atoms_extra,
            model_params=model_params,
        )
        y_pred = results["y_pred"]
        # Store the results in the ase database.
        add_indices = []
        for kk, atoms in enumerate(atoms_extra):
            e_form = y_pred[kk]
            atoms_copy = atoms.copy()
            atoms_copy.info = atoms.info.copy()
            atoms_copy.info["E_form"] = float(e_form)
            write_atoms_to_db(atoms=atoms_copy, db_ase=db_model)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------