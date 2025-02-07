# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db.core import Database

# -------------------------------------------------------------------------------------
# TRAIN MODEL AND PREDICT
# -------------------------------------------------------------------------------------

def train_model_and_predict(
    model_name: str,
    atoms_train: list,
    atoms_test: list,
    model_params: dict = {},
) -> dict:
    """Train the model and predict the test data."""
    # Linear scaling models.
    if model_name == "LSR":
        from ase_ml_models.linear import lsr_train, lsr_predict
        models_dict = lsr_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = lsr_predict(
            atoms_test=atoms_test,
            models_dict=models_dict,
            **model_params,
        )
        results = {"y_pred": y_pred, "models_dict": models_dict}
    # BEP models.
    elif model_name == "BEP":
        # Train the BEP model.
        from ase_ml_models.linear import bep_train, bep_predict
        models_dict = bep_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = bep_predict(
            atoms_test=atoms_test,
            models_dict=models_dict,
            **model_params,
        )
        results = {"y_pred": y_pred, "models_dict": models_dict}
    # Scikit-learn model.
    elif model_name == "SKLearn":
        from ase_ml_models.sklearn import sklearn_train, sklearn_predict
        # Train the scikit-learn model.
        model = sklearn_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = sklearn_predict(
            atoms_test=atoms_test,
            model=model,
            target=model_params.get("target", "E_form"),
        )
        results = {"y_pred": y_pred, "model": model}
    # WWL-GPR model.
    elif model_name == "WWLGPR":
        from ase_ml_models.wwlgpr import wwlgpr_train, wwlgpr_predict
        # Train the WWL-GPR model.
        model = wwlgpr_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = wwlgpr_predict(
            atoms_test=atoms_test,
            model=model,
            target=model_params.get("target", "E_form"),
        )
        results = {"y_pred": y_pred, "model": model}
    # Return the results.
    return results

# -------------------------------------------------------------------------------------
# CROSSVALIDATION
# -------------------------------------------------------------------------------------

def crossvalidation(
    atoms_list: list,
    model_name: str,
    crossval: object,
    key_groups: str = "material",
    key_stratify: str = "species",
    atoms_add: list = [],
    db_model: Database = None,
    print_error_thr: float = 0.5, # [eV]
    model_params: dict = {},
) -> dict:
    """Cross-validation test."""
    from ase_ml_models.databases import write_atoms_to_db
    # Get groups and stratify for splits.
    groups = [atoms.info[key_groups] for atoms in atoms_list]
    stratify = [atoms.info[key_stratify] for atoms in atoms_list]
    # Initialize cross-validation.
    indices = list(range(len(atoms_list)))
    y_test_all = []
    y_pred_all = []
    y_err_all = []
    for ii, (indices_train, indices_test) in enumerate(
        crossval.split(X=indices, y=stratify, groups=groups)
    ):
        # Split the data.
        atoms_train = list(np.array(atoms_list, dtype=object)[indices_train])
        atoms_test = list(np.array(atoms_list, dtype=object)[indices_test])
        y_test = [atoms.info["E_form"] for atoms in atoms_test]
        # Add reference atoms to the train sets.
        atoms_train += atoms_add
        # Train the model and do predictions.
        results = train_model_and_predict(
            model_name=model_name,
            atoms_train=atoms_train,
            atoms_test=atoms_test,
            model_params=model_params,
        )
        y_pred = results["y_pred"]
        # Store the results in the ase database.
        for atoms, e_form, e_form_dft in zip(atoms_test, y_pred, y_test):
            if db_model is not None:
                atoms_copy = atoms.copy()
                atoms_copy.info = atoms.info.copy()
                atoms_copy.info["E_form"] = e_form
                atoms_copy.info["E_form_dft"] = e_form_dft
                write_atoms_to_db(atoms=atoms_copy, db_ase=db_model)
            # Print high-error structures.
            if np.abs(e_form-e_form_dft) > print_error_thr:
                print(f"{atoms.info['name']:70s} {e_form:+7.3f} {e_form_dft:+7.3f}")
        # Store the results in lists.
        y_test_all += y_test
        y_pred_all += y_pred
        y_err_all += list(np.abs(np.array(y_pred)-np.array(y_test)))
    # Return the results.
    results = {
        "y_test": y_test_all,
        "y_pred": y_pred_all,
        "y_err": y_err_all,
    }
    return results

# -------------------------------------------------------------------------------------
# ENSEMBLE CROSSVALIDATION
# -------------------------------------------------------------------------------------

def ensemble_crossvalidation(
    atoms_list: list,
    model_name: str,
    crossval: object,
    key_groups: str = "material",
    key_stratify: str = "species",
    atoms_add: list = [],
    db_model: Database = None,
    print_error_thr: float = 0.5, # [eV]
    model_params: dict = {},
) -> dict:
    """Cross-validation test with uncertainty prediction."""
    from ase_ml_models.databases import write_atoms_to_db
    # Get groups and stratify for splits.
    groups = [atoms.info[key_groups] for atoms in atoms_list]
    stratify = [atoms.info[key_stratify] for atoms in atoms_list]
    # Initialize cross-validation.
    indices = list(range(len(atoms_list)))
    y_test_all = []
    y_pred_all = []
    y_err_all = []
    y_std_all = []
    for ii, (indices_train, indices_test) in enumerate(
        crossval.split(X=indices, y=stratify, groups=groups)
    ):
        # Split the data.
        atoms_train = list(np.array(atoms_list, dtype=object)[indices_train])
        atoms_test = list(np.array(atoms_list, dtype=object)[indices_test])
        # Prepare results.
        y_test = [atoms.info["E_form"] for atoms in atoms_test]
        y_pred_list = []
        # Split the test data.
        stratify_ii = [atoms.info[key_stratify] for atoms in atoms_train]
        groups_ii = [atoms.info[key_groups] for atoms in atoms_train]
        for jj, (indices_train_jj, indices_test_jj) in enumerate(
            crossval.split(X=indices_train, y=stratify_ii, groups=groups_ii)
        ):
            atoms_train_jj = list(
                np.array(atoms_train, dtype=object)[indices_train_jj]
            )
            # Add reference atoms to the train sets.
            atoms_train += atoms_add
            # Train the model and do predictions.
            results = train_model_and_predict(
                model_name=model_name,
                atoms_train=atoms_train_jj,
                atoms_test=atoms_test,
                model_params=model_params,
            )
            y_pred_list.append(results["y_pred"])
        y_pred = list(np.mean(y_pred_list, axis=0))
        y_std = list(np.std(y_pred_list, axis=0))
        # Store the results in the ase database.
        y_pred_array = np.array(y_pred_list)
        for ii, atoms in enumerate(atoms_test):
            e_form = y_pred[ii]
            e_form_list = y_pred_array[:, ii]
            e_form_dft = y_test[ii]
            if db_model is not None:
                atoms_copy = atoms.copy()
                atoms_copy.info = atoms.info.copy()
                atoms_copy.info["E_form"] = e_form
                atoms_copy.info["E_form_list"] = e_form_list
                atoms_copy.info["E_form_dft"] = e_form_dft
                write_atoms_to_db(atoms=atoms_copy, db_ase=db_model)
            # Print high-error structures.
            if np.abs(e_form-e_form_dft) > print_error_thr:
                print(f"{atoms.info['name']:70s} {e_form:+7.3f} {e_form_dft:+7.3f}")
        # Store the results in lists.
        y_test_all += y_test
        y_pred_all += y_pred
        y_err_all += list(np.abs(np.array(y_pred)-np.array(y_test)))
        y_std_all += y_std
    # Return the results.
    results = {
        "y_test": y_test_all,
        "y_pred": y_pred_all,
        "y_err": y_err_all,
        "y_std": y_std_all,
    }
    return results

# -------------------------------------------------------------------------------------
# GET CROSSVAL
# -------------------------------------------------------------------------------------

def get_crossval(
    crossval_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> object:
    """Get cross-validator."""
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        GroupKFold,
        StratifiedGroupKFold,
    )
    crossval_dict = {
        "KFold": KFold,
        "StratifiedKFold": StratifiedKFold,
        "GroupKFold": GroupKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
    }
    crossval = crossval_dict[crossval_name](
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return crossval

# -------------------------------------------------------------------------------------
# GET ATOMS REF
# -------------------------------------------------------------------------------------

def get_atoms_ref(
    atoms_list: list,
    species_ref: list,
    most_stable: bool = False,
) -> list:
    """Get most stable atoms for reference species."""
    atoms_ref = [
        atoms for atoms in atoms_list if atoms.info["species"] in species_ref
    ]
    if most_stable is True:
        atoms_ref = [
            atoms for atoms in atoms_ref if atoms.info["most_stable"] is True
        ]
    return atoms_ref

# -------------------------------------------------------------------------------------
# UPDATE TS ATOMS
# -------------------------------------------------------------------------------------

def update_ts_atoms(
    atoms_list: list,
    db_ads: Database,
    most_stable: bool = True,
):
    """Update transition state atoms with adsorbate data."""
    from ase_ml_models.databases import get_atoms_list_from_db
    for atoms in atoms_list:
        # Get names of reactants and products.
        reactants, products = atoms.info["species"].split("→")
        reactants = reactants.split("+")
        products = products.split("+")
        # Prepare kwargs for the database.
        kwargs = {"surface": atoms.info['surface']}
        # Calculate energy of the first image.
        e_first = 0.
        for species in reactants:
            kwargs.update({"species": species})
            atoms_ads_list = get_atoms_list_from_db(db_ase=db_ads, **kwargs)
            if len(atoms_ads_list) > 0:
                atoms_ads = sorted(atoms_ads_list, key=lambda x: x.info["E_form"])[0]
                e_first += atoms_ads.info["E_form"]
            else:
                print(species)
        # Calculate energy of the last image.
        e_last = 0.
        for species in products:
            kwargs.update({"species": species})
            atoms_ads_list = get_atoms_list_from_db(db_ase=db_ads, **kwargs)
            if len(atoms_ads_list) > 0:
                atoms_ads = sorted(atoms_ads_list, key=lambda x: x.info["E_form"])[0]
                e_last += atoms_ads.info["E_form"]
            else:
                print(species)
        # Update the atoms info.
        atoms.info["E_first"] = e_first
        atoms.info["E_last"] = e_last
        atoms.info["ΔE_react"] = e_last - e_first
        # Update features.
        for key in ["E_first", "E_last", "ΔE_react"]:
            index = atoms.info["features_names"].index(key)
            features = np.array(atoms.info["features"])
            features[:, index] = [atoms.info[key]] * len(atoms)
            atoms.info["features"] = features
        # Update average features.
        for key in ["E_first", "E_last", "ΔE_react"]:
            index = atoms.info["features_ave_names"].index(key)
            atoms.info["features_ave"][index] = atoms.info[key]

# -------------------------------------------------------------------------------------
# PARITY PLOT
# -------------------------------------------------------------------------------------

def parity_plot(
    results: dict,
    ax: object = None,
    lims: list = [-3, +5],
    alpha: float = 0.3,
    color: str = "crimson",
) -> object:
    """Parity plot of the results."""
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lims, lims, "k--")
    if "y_std" not in results:
        ax.scatter(
            x=results["y_test"],
            y=results["y_pred"],
            marker="o",
            alpha=alpha,
            color=color,
        )
    else:
        ax.errorbar(
            x=results["y_test"],
            y=results["y_pred"],
            yerr=results["y_std"],
            fmt="o",
            alpha=alpha,
            color=color,
            capsize=5,
        )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("E$_{DFT}$ [eV]", fontdict={"fontsize": 16})
    ax.set_ylabel("E$_{model}$ [eV]", fontdict={"fontsize": 16})
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return ax

# -------------------------------------------------------------------------------------
# VIOLIN PLOT
# -------------------------------------------------------------------------------------

def violin_plot(
    results: dict,
    ax: object = None,
    ylim: list = [0., +1.5],
    alpha: float = 0.8,
    color: str = "crimson",
) -> object:
    """Violin plot of the errors."""
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
    violin = ax.violinplot(
        dataset=[results["y_err"]],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )["bodies"][0]
    violin.set_facecolor(color)
    violin.set_alpha(alpha)
    violin.set_edgecolor("k")
    ax.set_ylabel("Errors [eV]", fontdict={"fontsize": 16})
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return ax

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------