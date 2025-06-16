# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ase_ml_models.databases import get_atoms_list_from_db
from ase_ml_models.yaml import customize_yaml, convert_numpy_to_python
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
    key_groups = "surface" # surface | elements
    key_stratify = "species"
    n_splits = 3
    random_state = 42
    most_stable = False
    add_ref_atoms = True
    exclude_add = True
    fraction_data = 0.15
    # Model selection.
    model_name = "WWLGPR" # Linear | SKLearn | WWLGPR
    model_sklearn = "LightGBM" # RandomForest | XGBoost | LightGBM
    update_features = False
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
    
    # Read Ase database.
    db_ase_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    # Fraction of the data.
    random.Random(random_state).shuffle(atoms_list)
    atoms_list = atoms_list[:int(fraction_data*len(atoms_list))]
    # Update features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"databases/atoms_adsorbates_{model_name_ref}_database.db"
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
    if model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list+atoms_add)
    
    # Hyperparameters optimization.
    num_samples = 100
    #np.random.seed(random_state)
    
    if model_name == "SKLearn" and model_sklearn == "LightGBM":
        from lightgbm import LGBMRegressor
        model_params_dict["SKLearn"]["model"] = LGBMRegressor()
        hyperparams_fixed = {
            "max_bin": int(np.exp(10)),
            "verbose": -1,
        }
        hyperparams_scan = {
            "n_estimators": np.random.randint(100, 1000, num_samples),
            "num_leaves": np.random.randint(3, 20, num_samples),
            "min_child_samples": np.random.randint(1, 10, num_samples),
            "learning_rate": np.random.uniform(0.0001, 0.01, num_samples),
            "colsample_bytree": np.random.uniform(0.01, 1., num_samples),
            "reg_alpha": np.random.uniform(0.0001, 0.1, num_samples),
            "reg_lambda": np.random.uniform(0.0001, 0.1, num_samples),
        }

    if model_name == "WWLGPR":
        hyperparams_fixed = {
            "cutoff": 2,
            "inner_cutoff": 1,
            "gpr_sigma": 1,
        }
        hyperparams_scan = {
            "inner_weight": np.random.uniform(0.5, 1.0, num_samples),
            "outer_weight": np.random.uniform(0.0, 0.5, num_samples),
            "gpr_reg": np.random.uniform(0.001, 0.010, num_samples),
            "gpr_len": np.random.uniform(1., 100., num_samples),
            "edge_s_s": np.random.uniform(0.0, 1.0, num_samples),
            "edge_s_a": np.random.uniform(0.5, 1.0, num_samples),
            "edge_a_a": np.random.uniform(0.0, 1.0, num_samples),
        }
    
    # Yaml file.
    os.makedirs("hyperparams", exist_ok=True)
    task = f"groupval_{key_groups}" if "Group" in crossval_name else "crossval"
    model = model_name if model_name != "SKLearn" else model_sklearn
    yaml_results = f"hyperparams/hyperparams_{task}.yaml"
    # Customize YAML writer.
    customize_yaml(float_format="{:10.5E}")
    
    # Model hyperparameters.
    model_params = model_params_dict[model_name]
    hyperparams_dict = {}
    for ii in range(num_samples):
        hyperparams = {key: hyperparams_scan[key][ii] for key in hyperparams_scan}
        hyperparams.update(hyperparams_fixed)
        model_params["hyperparams"] = hyperparams
        # Cross-validation.
        results = crossvalidation(
            atoms_list=atoms_list,
            model_name=model_name,
            crossval=crossval,
            key_groups=key_groups,
            key_stratify=key_stratify,
            atoms_add=atoms_add,
            exclude_add=exclude_add,
            db_model=None,
            model_params=model_params,
            print_error_thr=np.inf,
        )
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        # Calculate the MAE and the RMSE.
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print("Average results:")
        print(f"TOT MAE:   {mae:7.4f} [eV]")
        print(f"TOT RMSE:  {rmse:7.4f} [eV]")
        # Load the results.
        results_all = {}
        if os.path.isfile(yaml_results):
            with open(yaml_results, 'r') as fileobj:
                results_all = yaml.safe_load(fileobj)
        # Store the results.
        if model not in results_all or results_all[model]["MAE"] > mae:
            results_params = {
                "hyperparams": hyperparams,
                "MAE": mae,
                "RMSE": rmse,
            }
            results_all[model] = convert_numpy_to_python(results_params)
            with open(yaml_results, 'w') as fileobj:
                yaml.dump(
                    data=results_all,
                    stream=fileobj,
                    default_flow_style=None,
                    width=1000,
                    sort_keys=False,
                )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------