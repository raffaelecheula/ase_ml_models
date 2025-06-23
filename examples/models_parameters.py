# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------------------------
# GET MODEL PARAMETERS
# -------------------------------------------------------------------------------------

def get_model_parameters(
    model_name: str,
    model_sklearn: str,
    species_type: str,
    most_stable: bool,
):
    """
    Get model parameters based on the model name and species type. 
    """
    
    # Set target energy based on species type.
    target = "E_act" if species_type == "reactions" else "E_form"

    # TSR models.
    if model_name == "TSR":
        # TSR model parameters.
        fixed_TSR = {spec: ["CO*"] for spec in ["CO2*", "COH*", "cCOOH*", "HCO*"]}
        fixed_TSR.update({spec: ["O*"] for spec in ["HCOO*", "OH*", "H2O*"]})
        fixed_TSR.update({spec: ["H*"] for spec in ["H2*"]})
        model_params = {
            "keys_TSR": ["species"] if most_stable else ["species", "site"],
            "fixed_TSR": fixed_TSR,
        }

    # BEP models.
    elif model_name == "BEP":
        # BEP model parameters.
        model_params = {
            "keys_BEP": ["species", "miller_index"],
        }

    # SKLearn model.
    elif model_name == "SKLearn":
        if model_sklearn == "RandomForest":
            # Random Forest model.
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="auto",
                bootstrap=True,
                verbose=0,
            )
        elif model_sklearn == "XGBoost":
            # XGBoost model.
            from xgboost import XGBRegressor
            model = XGBRegressor(
                booster='gbtree',
                n_estimators=7,
                max_leaves=8,
                min_child_weight=2.685,
                learning_rate=0.4743,
                subsample=0.8852,
                colsample_bylevel=1.0,
                colsample_bytree=0.8929,
                reg_alpha=0.0471,
                reg_lambda=1.4083,
            )
        elif model_sklearn == "LightGBM":
            # LightGBM model.
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=169,
                num_leaves=8,
                min_child_samples=3,
                learning_rate=0.0924,
                colsample_bytree=0.656,
                reg_alpha=0.0262,
                reg_lambda=0.1216,
                verbose=-1,
            )
        elif model_sklearn == "GPR":
            # Gaussian Process Regression model.
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                RBF,
                ConstantKernel,
                WhiteKernel,
            )
            kernel = (
                ConstantKernel(constant_value_bounds=[1e-1, 1e+1]) *
                RBF(length_scale_bounds=[1e-2, 1e+2]) +
                WhiteKernel(noise_level_bounds=[1e-1, 1e+1])
            )
            model = GaussianProcessRegressor(kernel=kernel)
        # Sklearn model parameters.
        model_params = {
            "model": model,
            "target": target,
        }
    
    # WWLGPR model.
    if model_name == "WWLGPR":
        if species_type == "adsorbates":
            hyperparams = {
                "inner_weight": 0.64,
                "outer_weight": 0.17,
                "gpr_reg": 0.008,
                "gpr_len": 70.,
                "edge_s_s": 0.90,
                "edge_s_a": 0.90,
                "edge_a_a": 0.90,
            }
        elif species_type == "reactions":
            model_params["hyperparams"] = {
                "inner_weight": 0.60,
                "outer_weight": 0.02,
                "gpr_reg": 0.0300,
                "gpr_len": 22.0,
                "edge_s_s": 0.50,
                "edge_s_a": 1.00,
                "edge_a_a": 0.70,
            }
        # WWLGPR model parameters.
        model_params = {
            "target": target,
            "hyperparams": hyperparams,
        }
    
    # Graph model.
    if model_name == "Graph":
        # Graph model parameters.
        model_params = {
            "target": target,
            "model_name": "KRR",
            "kwargs_kernel": {"length_scale": 30},
            "kwargs_model": {"alpha": 1e-4},
        }
    
    return model_params
    
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------