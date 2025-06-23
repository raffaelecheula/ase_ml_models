# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db, write_atoms_to_db
from ase_ml_models.workflow import (
    update_ts_atoms,
    get_crossvalidator,
    train_model_and_predict,
)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    species_type = "adsorbates" # adsorbates | reactions # Type of species.
    stratified = True # Stratified cross-validation.
    group = False # Group cross-validation.
    key_groups = "surface" # surface | elements # Key for grouping the data.
    key_stratify = "species" # Key for stratification.
    n_splits = 5 # Number of splits for cross-validation.
    random_state = 42 # Random state for reproducibility.
    most_stable = False # Use only the most stable structures.
    ensemble = True # Use the cross-validator to get an ensemble of models.
    add_ref_atoms = False # Add reference atoms to the training set.

    # Model selection.
    model_name = "Graph" # TSR | BEP | SKLearn | WWLGPR | Graph | PyG
    model_sklearn = "LightGBM" # RandomForest | XGBoost | LightGBM
    update_features = True # Update features of TS atoms from an Ase database.
    model_name_ref = model_name[:] if model_name != "BEP" else "TSR"
    
    # Model parameters.
    species_ref = ["CO*", "H*", "O*"]
    # Get model parameters from a separate file.
    from models_parameters import get_model_parameters
    model_params = get_model_parameters(
        model_name=model_name,
        model_sklearn=model_sklearn,
        species_type=species_type,
        most_stable=most_stable,
    )
    
    # Read Ase train database.
    db_train_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_train = connect(db_train_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_train, **kwargs)
    # Read Ase extra database.
    db_extra_name = f"databases/atoms_{species_type}_DFT_extrapol_empty.db"
    db_extra = connect(db_extra_name)
    atoms_extra = get_atoms_list_from_db(db_ase=db_extra)
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        db_add_name = f"databases/atoms_{species_type}_DFT_extrapol_add.db"
        db_add = connect(db_add_name)
        kwargs = {"most_stable": True} if most_stable is True else {}
        atoms_add = get_atoms_list_from_db(db_ase=db_add, **kwargs)
    else:
        atoms_add = []
    
    # Update TS features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"databases/atoms_adsorbates_{model_name_ref}_extrapol.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_extra, db_ads=db_ads)
   
    # Preprocess the data.
    atoms_all = atoms_list + atoms_add + atoms_extra
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_prepare
        fixed_TSR = model_params.pop("fixed_TSR")
        tsr_prepare(atoms_all, species_TSR=species_ref, fixed_TSR=fixed_TSR)
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_all)
    elif model_name == "Graph":
        from ase_ml_models.graph import graph_preprocess, precompute_distances
        node_weight_dict = {"A0": 1.00, "S1": 0.80, "S2": 0.20}
        edge_weight_dict = {"AA": 0.50, "AS": 1.00, "SS": 0.50}
        graph_preprocess(
            atoms_list=atoms_all,
            node_weight_dict=node_weight_dict,
            edge_weight_dict=edge_weight_dict,
        )
        filename = "distances_extrapol.npy"
        distances = precompute_distances(
            atoms_X=atoms_list+atoms_add,
            atoms_Y=atoms_extra,
            filename=filename,
        )
        model_params.update({"distances": distances})

    # Print number of atoms.
    print(f"n train: {len(atoms_list)}")
    print(f"n extra: {len(atoms_extra)}")
    print(f"n added: {len(atoms_add)}")
    
    # Initialize cross-validation.
    crossval = get_crossvalidator(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Prepare Ase database.
    db_model_name = f"databases/atoms_{species_type}_{model_name}_extrapol.db"
    db_model = connect(db_model_name, append=False)
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