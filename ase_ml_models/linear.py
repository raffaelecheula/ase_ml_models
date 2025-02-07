# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------------
# LSR PREPARE
# -------------------------------------------------------------------------------------

def lsr_prepare(
    atoms_list: list,
    species_LSR: list,
    fixed_LSR: dict = {},
):
    """Prepare the data for Linear Scaling Relations."""
    # Get reference energies for each surface.
    energies_ref_dict = {}
    for atoms in [
        atoms for atoms in atoms_list if atoms.info["species"] in species_LSR
    ]:
        surface = atoms.info["surface"]
        species = atoms.info["species"]
        if surface not in energies_ref_dict:
            energies_ref_dict[surface] = {}
        if atoms.info["E_form"] < energies_ref_dict[surface].get(species, np.inf):
            energies_ref_dict[surface][species] = atoms.info["E_form"]
    # Add the reference energies to the atoms objects.
    for atoms in atoms_list:
        surface = atoms.info["surface"]
        species = atoms.info["species"]
        species_LSR_ii = fixed_LSR[species] if species in fixed_LSR else species_LSR
        atoms.info["species_LSR"] = species_LSR_ii
        atoms.info["E_LSR"] = [
            energies_ref_dict[surface][species] for species in species_LSR_ii
        ]

# -------------------------------------------------------------------------------------
# GET LSR DATA DICT
# -------------------------------------------------------------------------------------

def get_lsr_data_dict(
    atoms_train: list,
    keys_LSR: list = ["species"],
):
    """Get the dictionary of data for Linear Scaling Relations."""
    # Prepare the dictionary.
    lsr_data_dict = {
        " ".join([atoms.info[key] for key in keys_LSR]):
            {"E_form": [], "E_LSR": [], "surface": []}
        for atoms in atoms_train
    }
    # Collect the data for LSR relations.
    for atoms in atoms_train:
        key = " ".join([atoms.info[key] for key in keys_LSR])
        lsr_data_dict[key]["E_form"].append(atoms.info["E_form"])
        lsr_data_dict[key]["E_LSR"].append(atoms.info["E_LSR"])
        if "surface" in atoms.info:
            lsr_data_dict[key]["surface"].append(atoms.info["surface"])
    return lsr_data_dict

# -------------------------------------------------------------------------------------
# GET LSR MODELS DICT
# -------------------------------------------------------------------------------------

def get_lsr_models_dict(
    lsr_data_dict: dict,
):
    """Train the Linear Scaling Relation models."""
    # Train the models.
    models_dict = {}
    for key in lsr_data_dict:
        e_form_list = lsr_data_dict[key]["E_form"]
        e_lrs_list = lsr_data_dict[key]["E_LSR"]
        y_dep = np.array(e_form_list)
        X_indep = np.array(e_lrs_list)
        regr = LinearRegression()
        regr.fit(X_indep, y_dep)
        models_dict[key] = regr
    return models_dict

# -------------------------------------------------------------------------------------
# LSR TRAIN
# -------------------------------------------------------------------------------------

def lsr_train(
    atoms_train: list,
    keys_LSR: list = ["species"],
):
    """Get the data and train the Linear Scaling Relation models."""
    lsr_data_dict = get_lsr_data_dict(atoms_train=atoms_train, keys_LSR=keys_LSR)
    models_dict = get_lsr_models_dict(lsr_data_dict=lsr_data_dict)
    return models_dict

# -------------------------------------------------------------------------------------
# LSR PREDICT
# -------------------------------------------------------------------------------------

def lsr_predict(
    atoms_test: list,
    models_dict: dict,
    keys_LSR: list = ["species"],
):
    """Predict energies from Linear Scaling Relation models."""
    y_pred = []
    for atoms in atoms_test:
        key = " ".join([atoms.info[key] for key in keys_LSR])
        e_form = models_dict[key].predict([atoms.info["E_LSR"]])[0]
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