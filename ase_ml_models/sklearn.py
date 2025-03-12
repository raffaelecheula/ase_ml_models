# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

# -------------------------------------------------------------------------------------
# SKLEARN TRAIN
# -------------------------------------------------------------------------------------

def sklearn_preprocess(
    atoms_list: list,
    pipeline: list = [SimpleImputer(), RobustScaler()],
    encode_categorical: bool = True,
):
    # Get the features of the train set.
    features_names = atoms_list[0].info["features_ave_names"]
    features_proc_names = [
        name for feature, name 
        in zip(atoms_list[0].info["features_ave"], features_names)
        if not isinstance(feature, str)
    ]
    # Get the features and categorical features.
    features = []
    features_categ = []
    for atoms in atoms_list:
        if features_names != atoms.info["features_ave_names"]:
            print("Warning: Feature names are not the same.")
        features.append([
            feature for ii, feature in enumerate(atoms.info["features_ave"])
            if not isinstance(feature, str)
        ])
        features_categ.append([
            feature for ii, feature in enumerate(atoms.info["features_ave"])
            if isinstance(feature, str)
        ])
    # Encode categorical features.
    if encode_categorical:
        enc = OneHotEncoder()
        features_categ = enc.fit_transform(features_categ)
        features = np.hstack([features, features_categ.toarray()])
        features_proc_names += [f"is_{ii}" for ii in np.hstack(enc.categories_)]
    # Set pipeline.
    model = make_pipeline(*pipeline)
    # Train the model.
    features = model.fit_transform(X=features)
    # Store the features in the atoms objects.
    for ii, atoms in enumerate(atoms_list):
        atoms.info["features_mod"] = features[ii]
        atoms.info["features_mod_names"] = features[ii]

# -------------------------------------------------------------------------------------
# SKLEARN TRAIN
# -------------------------------------------------------------------------------------

def sklearn_train(
    atoms_train: list,
    model: object = None,
    hyperparams: dict = None,
    target: str = "E_form",
):
    """Train a scikit-learn model."""
    # Default model.
    if model is None:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        if hyperparams is None:
            hyperparams = {"n_estimators": 100, "max_depth": 10}
    # Get the features and target values of the train set.
    X_train = np.array([atoms.info["features_mod"] for atoms in atoms_train])
    y_train = np.array([atoms.info[target] for atoms in atoms_train])
    # Set model parameters.
    if hyperparams is not None:
        model.set_params(**hyperparams)
    model.fit(X=X_train, y=y_train)
    return model

# -------------------------------------------------------------------------------------
# SKLEARN PREDICT
# -------------------------------------------------------------------------------------

def sklearn_predict(
    atoms_test: list,
    model: object,
    target: str = "E_form",
):
    # Get the features of the test set.
    X_test = np.array([atoms.info["features_mod"] for atoms in atoms_test])
    # Predict the target values.
    y_pred = model.predict(X=X_test)
    if target == "E_bind":
        y_pred = [yy+atoms.info["E_form_gas"] for yy, atoms in zip(y_pred, atoms_test)]
    elif target == "E_act":
        y_pred = [yy+atoms.info["E_first"] for yy, atoms in zip(y_pred, atoms_test)]
    return [float(yy) for yy in y_pred]
        
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------