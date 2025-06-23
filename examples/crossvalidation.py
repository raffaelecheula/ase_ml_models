# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
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
    calibrate_uncertainty,
    parity_plot,
    violin_plot,
    groups_errors_plot,
    uncertainty_plot,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Cross-validation parameters.
    species_type = "adsorbates" # adsorbates | reactions
    stratified = True
    group = True
    key_groups = "surface" # surface | elements
    key_stratify = "species"
    n_splits = 5
    random_state = 42
    most_stable = False
    ensemble = True
    store_data = False
    add_ref_atoms = True
    exclude_add = True
    # Model selection.
    model_name = "Graph" # TSR | BEP | SKLearn | WWLGPR | Graph | PyG
    model_sklearn = "LightGBM" # RandomForest | XGBoost | LightGBM
    update_features = False
    model_name_ref = model_name[:] if model_name != "BEP" else "TSR"
    
    # Model parameters.
    species_ref = ["CO*", "H*", "O*"]
    target = "E_act" if species_type == "reactions" else "E_form"
    fixed_TSR = {spec: ["CO*"] for spec in ["CO2*", "COH*", "cCOOH*", "HCO*"]}
    fixed_TSR.update({spec: ["O*"] for spec in ["HCOO*", "OH*", "H2O*"]})
    fixed_TSR.update({spec: ["H*"] for spec in ["H2*"]})
    # Model hyperparameters.
    model_params_dict = {
        "TSR": {"keys_TSR": ["species"] if most_stable else ["species", "site"]},
        "BEP": {"keys_BEP": ["species", "miller_index"]},
        "SKLearn": {"target": target},
        "WWLGPR": {"target": target},
        "Graph": {"target": target},
        "PyG": {"target": target},
    }
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
    db_ase_name = f"databases/atoms_{species_type}_DFT_database.db"
    db_ase = connect(db_ase_name)
    kwargs = {"most_stable": True} if most_stable is True else {}
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, **kwargs)
    # Reference atoms to add to the train sets.
    if add_ref_atoms is True and species_type == "adsorbates":
        atoms_add = get_atoms_ref(atoms_list=atoms_list, species_ref=species_ref)
    else:
        atoms_add = []
    
    # Update features from an Ase database.
    if update_features is True and species_type == "reactions":
        db_ads_name = f"databases/atoms_adsorbates_{model_name_ref}_database.db"
        db_ads = connect(db_ads_name)
        update_ts_atoms(atoms_list=atoms_list, db_ads=db_ads)
    # Preprocess the data.
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_prepare
        tsr_prepare(atoms_list, species_TSR=species_ref, fixed_TSR=fixed_TSR)
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_preprocess
        sklearn_preprocess(atoms_list=atoms_list)
    elif model_name == "Graph":
        from ase_ml_models.graph import graph_preprocess, precompute_distances
        node_weight_dict = {"A0": 1.00, "S1": 0.80, "S2": 0.20}
        edge_weight_dict = {"AA": 0.50, "AS": 1.00, "SS": 0.50}
        graph_preprocess(
            atoms_list=atoms_list,
            node_weight_dict=node_weight_dict,
            edge_weight_dict=edge_weight_dict,
        )
        filename = "distances.npy"
        distances = precompute_distances(atoms_X=atoms_list, filename=filename)
        model_params.update({"distances": distances})
    
    # Print number of atoms.
    print(f"n atoms: {len(atoms_list)}")
    print(f"n added: {len(atoms_add)}")
    
    # Initialize cross-validation.
    crossval = get_crossval(
        stratified=stratified,
        group=group,
        n_splits=n_splits,
        random_state=random_state,
    )
    # Prepare Ase database.
    db_model_name = f"databases/atoms_{species_type}_{model_name}_database.db"
    db_model = connect(db_model_name, append=False) if store_data else None
    # Cross-validation.
    crossvalidation_fun = ensemble_crossvalidation if ensemble else crossvalidation
    results = crossvalidation_fun(
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
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    # Calculate the MAE and the RMSE.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print("\nAverage results:")
    print(f"TOT MAE:  {mae:7.4f} [eV]")
    print(f"TOT RMSE: {rmse:7.4f} [eV]")
    
    # Plots parameters.
    plot_parity = True
    plot_species = False
    plot_materials = False
    plot_uncertainty = False
    # Colors and names for plots.
    color_dict = {
        "TSR": "darkcyan",
        "BEP": "darkcyan",
        "SKLearn": "orchid",
        "WWLGPR": "crimson",
        "Graph": "crimson",
        "PyG": "crimson",
    }
    color = color_dict[model_name]
    task = f"groupval_{key_groups}" if group is True else "crossval"
    task += "_stable" if most_stable is True else "_all"
    dirname = f"images/crossvalidation/{task}"
    os.makedirs(dirname, exist_ok=True)
    model = model_sklearn if model_name == "SKLearn" else model_name
    model_ref = model_sklearn if model_name_ref == "SKLearn" else model_name_ref
    if species_type == "reactions" and update_features is True:
        model = f"{model}_from_{model_ref}"
    # Parity plot.
    if plot_parity is True:
        lims = [-2.2, +2.8] if species_type == "adsorbates" else [-1.4, +5.2]
        ax = parity_plot(results=results, lims=lims, color=color)
        plt.savefig(f"{dirname}/parity_{species_type}_{model}.png")
    # Species error plot.
    if plot_species is True:
        ax = groups_errors_plot(results, atoms_list, key="species", color=color)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
        plt.savefig(f"{dirname}/species_{species_type}_{model}.png")
    # Material error plot.
    if plot_materials is True:
        ax = groups_errors_plot(results, atoms_list, key="material", color=color)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.35)
        plt.savefig(f"{dirname}/material_{species_type}_{model}.png")
    # Uncertainty quantification.
    if plot_uncertainty is True and ensemble is True:
        results = calibrate_uncertainty(results=results, fit_intercept=False)
        ax = uncertainty_plot(results=results, color=color)
        plt.savefig(f"{dirname}/uncertainty_{species_type}_{model}.png")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    main()
    print(f"Execution time: {timeit.default_timer() - start:.2f} s")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------