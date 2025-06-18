# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------------
# TSR PREPARE
# -------------------------------------------------------------------------------------

def tsr_prepare(
    atoms_list: list,
    species_TSR: list,
    fixed_TSR: dict = {},
):
    """Prepare the data for Linear Scaling Relations."""
    # Get reference energies for each surface.
    energies_ref_dict = {}
    for atoms in [
        atoms for atoms in atoms_list if atoms.info["species"] in species_TSR
    ]:
        surface = atoms.info["surface"]
        species = atoms.info["species"]
        if surface not in energies_ref_dict:
            energies_ref_dict[surface] = {}
        if atoms.info["E_form"] is None:
            continue
        if atoms.info["E_form"] < energies_ref_dict[surface].get(species, np.inf):
            energies_ref_dict[surface][species] = atoms.info["E_form"]
    # Add the reference energies to the atoms objects.
    for atoms in atoms_list:
        surface = atoms.info["surface"]
        species = atoms.info["species"]
        species_TSR_ii = fixed_TSR[species] if species in fixed_TSR else species_TSR
        atoms.info["species_TSR"] = species_TSR_ii
        atoms.info["E_TSR"] = [
            energies_ref_dict[surface][species] for species in species_TSR_ii
        ]

# -------------------------------------------------------------------------------------
# GET TSR DATA DICT
# -------------------------------------------------------------------------------------

def get_tsr_data_dict(
    atoms_train: list,
    keys_TSR: list = ["species"],
):
    """Get the dictionary of data for Linear Scaling Relations."""
    # Prepare the dictionary.
    tsr_data_dict = {
        " ".join([atoms.info[key] for key in keys_TSR]):
            {"E_form": [], "E_TSR": [], "surface": []}
        for atoms in atoms_train
    }
    # Collect the data for TSR relations.
    for atoms in atoms_train:
        key = " ".join([atoms.info[key] for key in keys_TSR])
        tsr_data_dict[key]["E_form"].append(atoms.info["E_form"])
        tsr_data_dict[key]["E_TSR"].append(atoms.info["E_TSR"])
        if "surface" in atoms.info:
            tsr_data_dict[key]["surface"].append(atoms.info["surface"])
    return tsr_data_dict

# -------------------------------------------------------------------------------------
# GET TSR MODELS DICT
# -------------------------------------------------------------------------------------

def get_tsr_models_dict(
    tsr_data_dict: dict,
):
    """Train the Linear Scaling Relation models."""
    # Train the models.
    models_dict = {}
    for key in tsr_data_dict:
        e_form_list = tsr_data_dict[key]["E_form"]
        e_lrs_list = tsr_data_dict[key]["E_TSR"]
        y_dep = np.array(e_form_list)
        X_indep = np.array(e_lrs_list)
        regr = LinearRegression()
        regr.fit(X_indep, y_dep)
        models_dict[key] = regr
    return models_dict

# -------------------------------------------------------------------------------------
# TSR TRAIN
# -------------------------------------------------------------------------------------

def tsr_train(
    atoms_train: list,
    keys_TSR: list = ["species"],
    **kwargs: dict,
):
    """Get the data and train the Linear Scaling Relation models."""
    tsr_data_dict = get_tsr_data_dict(atoms_train=atoms_train, keys_TSR=keys_TSR)
    models_dict = get_tsr_models_dict(tsr_data_dict=tsr_data_dict)
    return models_dict

# -------------------------------------------------------------------------------------
# TSR PREDICT
# -------------------------------------------------------------------------------------

def tsr_predict(
    atoms_test: list,
    models_dict: dict,
    keys_TSR: list = ["species"],
    **kwargs: dict,
):
    """Predict energies from Linear Scaling Relation models."""
    y_pred = []
    for atoms in atoms_test:
        key = " ".join([atoms.info[key] for key in keys_TSR])
        e_form = models_dict[key].predict([atoms.info["E_TSR"]])[0]
        y_pred.append(e_form)
    return y_pred

# -------------------------------------------------------------------------------------
# GET CORRELATION HEATMAP
# -------------------------------------------------------------------------------------

def get_correlation_heatmap(
    atoms_list: list,
):
    """Get the correlation heatmap of the energies."""
    from pandas import DataFrame
    from seaborn import heatmap
    from ase_ml_models.utilities import modify_name
    # Prepare the dictionary of energies.
    energies_dict = {}
    for atoms in atoms_list:
        surface = atoms.info["surface"]
        species = modify_name(atoms.info["species"])
        if species not in energies_dict:
            energies_dict[species] = {}
        energies_dict[species][surface] = atoms.info["E_form"]
    # Prepare the DataFrame and calculate the correlation.
    df = DataFrame(energies_dict)
    data_corr = df.corr(method='spearman')
    # Plot the heatmap.
    ax = heatmap(data=data_corr, vmin=-1.0, vmax=+1.0)
    ax.set_title("Spearman correlation")
    return ax

# -------------------------------------------------------------------------------------
# GET BEP DATA DICT
# -------------------------------------------------------------------------------------

def get_bep_data_dict(
    atoms_train: list,
    keys_BEP: list = ["species"],
):
    """Get the dictionary of data for Brønsted-Evans-Polanyi relations."""
    # Prepare the dictionary.
    bep_data_dict = {
        " ".join([atoms.info[key] for key in keys_BEP]): {
            "E_act": [], "ΔE_react": [], "surface": [],
        }
        for atoms in atoms_train
    }
    # Collect the data for BEP relations.
    for atoms in atoms_train:
        # Calculate activation energy and reaction energy.
        if "E_act" in atoms.info:
            e_act = atoms.info["E_act"]
        else:
            e_act = atoms.info["E_form"] - atoms.info["E_first"]
        if "ΔE_react" in atoms.info:
            deltae = atoms.info["ΔE_react"]
        else:
            deltae = atoms.info["E_last"] - atoms.info["E_first"]
        # Store the data in the dictionary.
        key = " ".join([atoms.info[key] for key in keys_BEP])
        bep_data_dict[key]["E_act"].append(e_act)
        bep_data_dict[key]["ΔE_react"].append(deltae)
        if "surface" in atoms.info:
            bep_data_dict[key]["surface"].append(atoms.info["surface"])
    return bep_data_dict

# -------------------------------------------------------------------------------------
# GET MODELS DICT
# -------------------------------------------------------------------------------------

def get_bep_models_dict(
    bep_data_dict: dict,
):
    """Train the Brønsted-Evans-Polanyi models."""
    # Train the models.
    models_dict = {}
    for key in bep_data_dict:
        e_act_list = bep_data_dict[key]["E_act"]
        deltae_list = bep_data_dict[key]["ΔE_react"]
        y_dep = np.array(e_act_list)
        X_indep = np.array(deltae_list).reshape(-1, 1)
        regr = LinearRegression()
        regr.fit(X_indep, y_dep)
        models_dict[key] = regr
    return models_dict

# -------------------------------------------------------------------------------------
# BEP TRAIN
# -------------------------------------------------------------------------------------

def bep_train(
    atoms_train: list,
    keys_BEP: list = ["species"],
    **kwargs: dict,
):
    """Get the data and train the Brønsted-Evans-Polanyi models."""
    bep_data_dict = get_bep_data_dict(atoms_train=atoms_train, keys_BEP=keys_BEP)
    models_dict = get_bep_models_dict(bep_data_dict=bep_data_dict)
    return models_dict

# -------------------------------------------------------------------------------------
# BEP PREDICT
# -------------------------------------------------------------------------------------

def bep_predict(
    atoms_test: list,
    models_dict: dict,
    keys_BEP: list = ["species"],
    **kwargs: dict,
):
    """Predict energies from Brønsted-Evans-Polanyi models."""
    y_pred = []
    for atoms in atoms_test:
        key = " ".join([atoms.info[key] for key in keys_BEP])
        deltae = atoms.info["ΔE_react"]
        e_act = models_dict[key].predict(np.array(deltae).reshape(-1, 1))[0]
        e_form = atoms.info["E_first"] + e_act
        y_pred.append(e_form)
    return y_pred
        
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------