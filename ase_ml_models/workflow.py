# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.db.core import Database
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------------------------------------------
# TRAIN MODEL AND PREDICT
# -------------------------------------------------------------------------------------

def train_model_and_predict(
    model_name: str,
    atoms_train: list,
    atoms_test: list,
    model_params: dict = {},
) -> dict:
    """
    Train the model and predict the test data.
    """
    # Linear scaling models.
    if model_name == "TSR":
        from ase_ml_models.linear import tsr_train, tsr_predict
        models_dict = tsr_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = tsr_predict(
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
            **model_params,
        )
        results = {"y_pred": y_pred, "model": model}
    # Graph model.
    elif model_name == "Graph":
        from ase_ml_models.graph import graph_train, graph_predict
        # Train the Graph model.
        model = graph_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = graph_predict(
            atoms_test=atoms_test,
            model=model,
            **model_params,
        )
        results = {"y_pred": y_pred, "model": model}
    # Pytorch Geometric model.
    elif model_name == "PyG":
        from ase_ml_models.pyg import pyg_train, pyg_predict
        # Train the Pytorch Geometric model.
        model = pyg_train(
            atoms_train=atoms_train,
            **model_params,
        )
        # Predict test data.
        y_pred = pyg_predict(
            atoms_test=atoms_test,
            model=model,
            **model_params,
        )
        results = {"y_pred": y_pred, "model": model}
    # Return the results.
    return results

# -------------------------------------------------------------------------------------
# IDENTITY 1 FOLD
# -------------------------------------------------------------------------------------

class Identity1Fold:
    """
    Cross-validator that does not split the data, but returns a single fold where 
    train indices = test indices = all indices (for calculating training errors).
    """
    def __init__(self, n_splits: int = 1, **kwargs: dict):
        assert n_splits == 1, "Identity1Fold only supports n_splits=1."
    
    def get_n_splits(self, X: list = None, y: list = None, groups: list = None):
        return 1
    
    def split(self, X: list, y: list = None, groups: list = None):
        yield range(len(X)), range(len(X))

# -------------------------------------------------------------------------------------
# GET CROSSVALIDATOR
# -------------------------------------------------------------------------------------

def get_crossvalidator(
    crossval_name: str = None,
    stratified: bool = None,
    group: bool = None,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> object:
    """
    Get cross-validator.
    """
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        GroupKFold,
        StratifiedGroupKFold,
    )
    # Prepare the cross-validation parameters.
    kwargs = {"shuffle": shuffle, "random_state": random_state}
    # Check the cross-validation type and create the object.
    if n_splits == 1:
        crossval = Identity1Fold(n_splits=n_splits)
    elif crossval_name == "KFold" or [group, stratified] == [False] * 2:
        crossval = KFold(n_splits=n_splits, **kwargs)
    elif crossval_name == "StratifiedKFold" or [group, stratified] == [False, True]:
        crossval = StratifiedKFold(n_splits=n_splits, **kwargs)
    elif crossval_name == "GroupKFold" or [group, stratified] == [True, False]:
        crossval = GroupKFold(n_splits=n_splits)
    elif crossval_name == "StratifiedGroupKFold" or [group, stratified] == [True] * 2:
        crossval = StratifiedGroupKFold(n_splits=n_splits, **kwargs)
    else:
        raise ValueError("Wrong cross-validation parameters.")
    # Return the cross-validation object.
    return crossval

# -------------------------------------------------------------------------------------
# CROSSVALIDATION
# -------------------------------------------------------------------------------------

def crossvalidation(
    atoms_list: list,
    model_name: str,
    crossval: object,
    key_groups: str = "material",
    key_stratify: str = "species",
    key_name: str = "name",
    atoms_add: list = [],
    exclude_add: bool = True,
    db_model: Database = None,
    print_mean_errors: bool = True,
    print_error_thr: float = None, # [eV]
    model_params: dict = {},
    ensemble: bool = False,
    resampling: bool = False,
    n_splits_ensemble: int = None,
    n_resamples: int = 100,
    db_kwargs: dict = {},
) -> dict:
    """
    Cross-validation test with uncertainty prediction.
    """
    from ase_ml_models.databases import write_atoms_to_db
    from sklearn.utils import resample
    # Get groups and stratify for splits.
    groups = [str(atoms.info[key_groups]) for atoms in atoms_list]
    stratify = [str(atoms.info[key_stratify]) for atoms in atoms_list]
    # Names of atoms added.
    atoms_add_names = [atoms.info[key_name] for atoms in atoms_add]
    # Threshold for prediction errors.
    print_error_thr = np.inf if print_error_thr is None else print_error_thr
    # Initialize cross-validation.
    y_true_all = []
    y_pred_all = []
    y_std_all = []
    indices_all = []
    indices_list = list(range(len(atoms_list)))
    for ii, (indices_train, indices_test) in enumerate(
        crossval.split(X=indices_list, y=stratify, groups=groups)
    ):
        # Split the data into train and test lists of atoms objects.
        atoms_train = list(np.array(atoms_list, dtype=object)[indices_train])
        atoms_test = list(np.array(atoms_list, dtype=object)[indices_test])
        # Prepare a list of indices to produce the train sets.
        indices_ii = list(range(len(atoms_train)))
        # Prepare data for training an ensemble of models.
        if ensemble is True:
            # Set the number of splits.
            if ii == 0:
                if n_splits_ensemble is None:
                    n_splits_ensemble = crossval.n_splits - 1
                crossval.n_splits = n_splits_ensemble
            # Get a list of indices that are subgroups of indices_ii, obtained
            # with the same cross-validator used to split train and test data.
            stratify_ii = [str(atoms.info[key_stratify]) for atoms in atoms_train]
            groups_ii = [str(atoms.info[key_groups]) for atoms in atoms_train]
            indices_jj_list = [indices for indices, _ in (
                crossval.split(X=indices_ii, y=stratify_ii, groups=groups_ii)
            )]
        else:
            # For standard cross-validation (no ensemble of models), use the
            # whole train set.
            indices_jj_list = [indices_ii]
        # Apply resampling to the list of indices_jj, to train an ensemble of
        # models via bootstrapping (random resampling with replacement).
        if resampling is True:
            indices_jj_list_copy = indices_jj_list.copy()
            indices_jj_list = []
            for indices in indices_jj_list_copy:
                for jj in range(n_resamples):
                    indices_jj_list.append(resample(indices, random_state=jj))
        # Prepare the results.
        y_true = [atoms.info["E_form"] for atoms in atoms_test]
        y_pred_list = []
        # Loop over the list of indices.
        for indices_jj in indices_jj_list:
            # Get the atoms for the current train set.
            atoms_train_jj = list(np.array(atoms_train, dtype=object)[indices_jj])
            # Add reference atoms to the train sets.
            atoms_train_jj += atoms_add
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
        excluded = []
        for kk, atoms in enumerate(atoms_test):
            e_form = y_pred[kk]
            e_form_list = y_pred_array[:, kk]
            e_form_dft = y_true[kk]
            if atoms.info[key_name] in atoms_add_names and exclude_add is True:
                excluded.append(kk)
            if db_model is not None:
                atoms_copy = atoms.copy()
                atoms_copy.info = atoms.info.copy()
                atoms_copy.info["E_form"] = e_form
                atoms_copy.info["E_form_list"] = e_form_list
                atoms_copy.info["E_form_DFT"] = e_form_dft
                write_atoms_to_db(atoms=atoms_copy, db_ase=db_model, **db_kwargs)
            # Print high-error structures.
            if np.abs(e_form - e_form_dft) > print_error_thr:
                print(f"{atoms.info[key_name]:70s} {e_form:+7.3f} {e_form_dft:+7.3f}")
        # Discard excluded atoms from results.
        y_true_ok = [yy for kk, yy in enumerate(y_true) if kk not in excluded]
        y_pred_ok = [yy for kk, yy in enumerate(y_pred) if kk not in excluded]
        y_std_ok = [yy for kk, yy in enumerate(y_std) if kk not in excluded]
        indices_ok = [ii for kk, ii in enumerate(indices_test) if kk not in excluded]
        # Print MAE and RMSE of the split.
        if print_mean_errors is True:
            mae = mean_absolute_error(y_true_ok, y_pred_ok)
            rmse = mean_squared_error(y_true_ok, y_pred_ok, squared=False)
            print(f"---- Split {ii+1} ----")
            print(f"MAE:  {mae:6.4f} eV")
            print(f"RMSE: {rmse:6.4f} eV")
        # Store the results in lists.
        y_true_all += y_true_ok
        y_pred_all += y_pred_ok
        y_std_all += y_std_ok
        indices_all += indices_ok
    # Print MAE and RMSE of all the splits.
    if print_mean_errors is True:
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = mean_squared_error(y_true_all, y_pred_all, squared=False)
        print("----  Total  ----")
        print(f"MAE:  {mae:6.4f} eV")
        print(f"RMSE: {rmse:6.4f} eV")
    # Return the results.
    results = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "indices": [int(ii) for ii in indices_all],
    }
    if ensemble is True or resampling is True:
        results["y_std"] = y_std_all
    return results

# -------------------------------------------------------------------------------------
# CHANGE TARGET ENERGY
# -------------------------------------------------------------------------------------

def change_target_energy(
    y_pred: list,
    atoms_test: list,
    target: str = "E_form",
):
    """
    Change the target energy to formation energy.
    """
    if target == "E_bind":
        y_pred = [
            yy + atoms.info["E_form_gas"] for yy, atoms in zip(y_pred, atoms_test)
        ]
    elif target == "E_act":
        y_pred = [
            yy + atoms.info["E_first"] for yy, atoms in zip(y_pred, atoms_test)
        ]
    return [float(yy) for yy in y_pred]

# -------------------------------------------------------------------------------------
# GET ATOMS REF
# -------------------------------------------------------------------------------------

def get_atoms_ref(
    atoms_list: list,
    species_ref: list,
    most_stable: bool = False,
) -> list:
    """
    Get most stable atoms for reference species.
    """
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
    e_form_dict: dict = {},
    reactants_products_dict: dict = None,
    key_arrow: str = "→",
    key_plus: str = "+",
    key_or: str = " / ",
    features_key_dict: dict = None,
):
    """
    Update transition state atoms with adsorbate data.
    """
    from ase_ml_models.databases import get_atoms_list_from_db
    # Get dictionary of reactants and products for each species.
    if reactants_products_dict is None:
        reactants_products_dict = {}
        for atoms in atoms_list:
            species = atoms.info["species"]
            reactants, products = atoms.info["species"].split(key_arrow)
            reactants = reactants.split(key_plus)
            products = products.split(key_plus)
            reactants_products_dict[species] = [reactants, products]
    # 
    for atoms in atoms_list:
        # Get names of reactants and products.
        reactants, products = reactants_products_dict[atoms.info["species"]]
        # Prepare kwargs for the database.
        kwargs = {"surface": atoms.info["surface"]}
        # Calculate energy of the first image.
        e_first = 0.
        for species in reactants:
            species_list = species.split(key_or) if key_or in species else [species]
            atoms_ads_list = []
            for species_ii in species_list:
                kwargs.update({"species": species_ii})
                atoms_ads_list += get_atoms_list_from_db(db_ase=db_ads, **kwargs)
            if len(atoms_ads_list) > 0:
                atoms_ads = sorted(atoms_ads_list, key=lambda x: x.info["E_form"])[0]
                e_first += atoms_ads.info["E_form"]
            elif species in e_form_dict:
                e_first += e_form_dict[species]
            else:
                print(f"Warning: {species} not found!")
        # Calculate energy of the last image.
        e_last = 0.
        for species in products:
            species_list = species.split(key_or) if key_or in species else [species]
            atoms_ads_list = []
            for species_ii in species_list:
                kwargs.update({"species": species_ii})
                atoms_ads_list += get_atoms_list_from_db(db_ase=db_ads, **kwargs)
            if len(atoms_ads_list) > 0:
                atoms_ads = sorted(atoms_ads_list, key=lambda x: x.info["E_form"])[0]
                e_last += atoms_ads.info["E_form"]
            elif species in e_form_dict:
                e_last += e_form_dict[species]
            else:
                print(f"Warning: {species} not found!")
        # Update the atoms info.
        atoms.info["E_first"] = e_first
        atoms.info["E_last"] = e_last
        atoms.info["ΔE_react"] = e_last - e_first
        # Update features.
        for key in ["E_first", "E_last", "ΔE_react"]:
            feat = features_key_dict[key] if features_key_dict is not None else key
            index = atoms.info["features_names"].index(feat)
            features = np.array(atoms.info["features"])
            features[:, index] = [atoms.info[key]] * len(atoms)
            atoms.info["features"] = features
        # Update average features.
        #for key in ["E_first", "E_last", "ΔE_react"]:
        #    feat = features_key_dict[key] if features_key_dict is not None else key
        #    index = atoms.info["features_ave_names"].index(feat)
        #    atoms.info["features_ave"][index] = atoms.info[key]

# -------------------------------------------------------------------------------------
# GET PREDICTION ERRORS
# -------------------------------------------------------------------------------------

def get_prediction_errors(
    results: dict,
) -> np.ndarray:
    """
    Get prediction errors from the results.
    """
    return np.abs(np.array(results["y_pred"]) - np.array(results["y_true"]))

# -------------------------------------------------------------------------------------
# GET MEANS AND STD BINS
# -------------------------------------------------------------------------------------

def get_means_and_std_bins(
    x_vect: list,
    y_vect: list,
    n_bins: int = 8,
) -> list:
    """
    Get mean value and standard deviation of bins.
    """
    x_vect = np.array(x_vect)
    y_vect = np.array(y_vect)
    indices = np.argsort(x_vect)
    x_bins = np.array_split(x_vect[indices], n_bins)
    y_bins = np.array_split(y_vect[indices], n_bins)
    x_means = [np.mean(xx) for xx in x_bins]
    y_means = [np.mean(yy) for yy in y_bins]
    y_stds = [np.std(yy) for yy in y_bins]
    return x_means, y_means, y_stds

# -------------------------------------------------------------------------------------
# CALIBRATE UNCERTAINTY
# -------------------------------------------------------------------------------------

def calibrate_uncertainty(
    results: dict,
    n_bins: int = 8,
    fit_intercept: bool = False,
) -> dict:
    """
    Calibrate the uncertainty.
    """
    if "y_std" not in results:
        return results
    # Get bins.
    x_means, y_means, y_stds = get_means_and_std_bins(
        x_vect=get_prediction_errors(results=results),
        y_vect=results["y_std"],
        n_bins=n_bins,
    )
    # Correct the uncertainty.
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression(fit_intercept=fit_intercept)
    regr.fit(np.array(x_means).reshape(-1, 1), np.array(y_means))
    m_line, a_line = regr.coef_[0], regr.intercept_
    y_means = [(yy - a_line) / m_line for yy in y_means]
    y_stds = [yy / m_line for yy in y_stds]
    results["y_std"] = [(yy - a_line) / m_line for yy in results["y_std"]]
    if fit_intercept is True:
        results["y_std"] = [yy if yy > 0. else 0. for yy in results["y_std"]]
    # Return the calibrate uncertainty.
    results["m_calib"] = m_line
    results["a_calib"] = a_line
    return results

# -------------------------------------------------------------------------------------
# PARITY PLOT
# -------------------------------------------------------------------------------------

def parity_plot(
    results: dict,
    ax: object = None,
    lims: list = [-3, +5],
    alpha: float = 0.50,
    color: str = "crimson",
    show_errors: bool = True,
    add_violin_plot: bool = True,
) -> object:
    """
    Parity plot of the results.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    ax.plot(lims, lims, "k--")
    ax.errorbar(
        x=results["y_true"],
        y=results["y_pred"],
        yerr=results["y_std"] if "y_std" in results else None,
        fmt="none",
        capsize=3,
        color="black",
    )
    ax.scatter(
        x=results["y_true"],
        y=results["y_pred"],
        s=50,
        c=color,
        alpha=alpha,
        edgecolors="k",
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(r"$E_{\mathregular{DFT}}$ [eV]", fontsize=16)
    ax.set_ylabel(r"$E_{\mathregular{model}}$ [eV]", fontsize=16)
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Calculate the MAE and the RMSE.
    if show_errors is True:
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        ax.text(
            x=lims[0] + (lims[1] - lims[0]) * 0.26,
            y=lims[0] + (lims[1] - lims[0]) * 0.92,
            s=f"MAE = {mae:6.3f} eV\nRMSE = {rmse:6.3f} eV",
            fontsize=13,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "edgecolor": "black",
                "facecolor": "white",
                "linewidth": 1.5,
            },
        )
    # Add violin plot.
    if add_violin_plot is True:
        inset_ax = fig.add_axes([0.76, 0.15, 0.17, 0.25])
        violin_plot(
            results=results,
            ax=inset_ax,
            ylim=[0., +1.5],
            alpha=0.8,
            color=color if isinstance(color, str) else "gray",
            show_errors=False,
        )
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
    show_errors: bool = True,
) -> object:
    """
    Violin plot of the errors.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    violin = ax.violinplot(
        dataset=[get_prediction_errors(results=results)],
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
    if show_errors is True:
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        ax.text(
            x=0.85,
            y=0.92*ylim[1],
            s=f"MAE = {mae:6.3f} eV\nRMSE = {rmse:6.3f} eV",
            fontsize=13,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "edgecolor": "black",
                "facecolor": "white",
                "linewidth": 1.5,
            },
        )
    return ax

# -------------------------------------------------------------------------------------
# UNCERTAINTY PLOT
# -------------------------------------------------------------------------------------

def uncertainty_plot(
    results: dict,
    n_bins: int = 8,
    ax: object = None,
    lims: list = [0.0, 1.0],
    alpha: float = 0.2,
    color: str = "crimson",
) -> object:
    """
    Uncertainty vs errors plot.
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    # Get means and std of bins.
    x_means, y_means, y_stds = get_means_and_std_bins(
        x_vect=get_prediction_errors(results=results),
        y_vect=results["y_std"],
        n_bins=n_bins,
    )
    # Plot the data.
    ax.plot(lims, lims, "k--")
    ax.errorbar(
        x=x_means,
        y=y_means,
        yerr=y_stds,
        ms=5,
        fmt="o",
        color="black",
        capsize=3,
    )
    ax.set_xlabel("Errors [eV]", fontdict={"fontsize": 16})
    ax.set_ylabel("Uncertainty [eV]", fontdict={"fontsize": 16})
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return ax

# -------------------------------------------------------------------------------------
# SPECIES ERRORS PLOT
# -------------------------------------------------------------------------------------

def groups_errors_plot(
    results: dict,
    atoms_list: list,
    group_list: list = None,
    key: str = "species",
    ax: object = None,
    percentile: bool = True,
    ylim: list = [0.0, 1.2],
    alpha: float = 0.8,
    kwargs_xticks: dict = {"rotation": 90., "ha": "center"},
    color: str = "crimson",
    modify_groups: bool = True,
    kwargs_name: dict = {},
    violin_plot: bool = True,
) -> object:
    """
    Groups errors plot.
    """
    if group_list is None:
        group_list = [atoms_list[ii].info[key] for ii in results["indices"]]
    group_dict = {}
    for group, y_err in zip(group_list, get_prediction_errors(results=results)):
        if group not in group_dict:
            group_dict[group] = []
        group_dict[group].append(y_err)
    group_dict = {key: group_dict[key] for key in sorted(group_dict.keys())}
    # Get the color for each group.
    if isinstance(color, dict):
        color = [color[key] for key in group_dict.keys()]
    # Modify group names if needed.
    if modify_groups is True:
        from ase_ml_models.utilities import modify_name
        group_names = [modify_name(name, **kwargs_name) for name in group_dict.keys()]
    # Plot the data.
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(len(group_dict) * 0.3 + 3, 5), dpi=300)
    if violin_plot is True:
        violins = ax.violinplot(
            dataset=group_dict.values(),
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )["bodies"]
        for ii, violin in enumerate(violins):
            violin.set_facecolor(color[ii] if isinstance(color, list) else color)
            violin.set_alpha(alpha)
            violin.set_edgecolor("k")
        if percentile is True:
            values = list(group_dict.values())
            yy = [np.median(vv) for vv in values]
            yerr = [
                [np.median(vv) - np.percentile(vv, 16) for vv in values],
                [np.percentile(vv, 84) - np.median(vv) for vv in values]
            ]
        else:
            yy = [np.mean(ii) for ii in group_dict.values()]
            yerr = [np.std(ii) for ii in group_dict.values()]
        ax.errorbar(
            x=range(1, len(group_dict) + 1),
            y=yy,
            yerr=yerr,
            ms=5,
            fmt="o",
            color="black",
            capsize=2,
        )
    else:
        ax.bar(
            x=range(1, len(group_dict) + 1),
            height=[np.mean(ii) for ii in group_dict.values()],
            yerr=[np.std(ii) for ii in group_dict.values()],
            color=color,
            alpha=alpha,
            capsize=5,
        )
    ax.set_xticks(list(range(1, len(group_dict) + 1)))
    ax.set_xticklabels(group_names, **kwargs_xticks)
    ax.set_ylabel("Errors [eV]", fontdict={"fontsize": 16})
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=13, width=1.5, length=6, direction="inout")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return ax

# -------------------------------------------------------------------------------------
# PRINT RESULTS YAML
# -------------------------------------------------------------------------------------

def print_results_yaml(
    results: dict,
    keys: list = ["y_true", "y_pred"],
    filename: str = "results.yaml",
) -> None:
    """
    Print results dictionary to a yaml file.
    """
    import yaml
    from ase_ml_models.yaml import convert_numpy_to_python, customize_yaml
    # Prepare results for yaml.
    results_yaml = {key: results[key] for key in keys}
    # Write to yaml file.
    customize_yaml(float_format="{:+0.8f}")
    with open(filename, "w", encoding="utf-8") as fileobj:
        yaml.dump(convert_numpy_to_python(results_yaml), fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------