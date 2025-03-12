# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np

# -------------------------------------------------------------------------------------
# GET OPTIMIZED HYPERPARAMETERS
# -------------------------------------------------------------------------------------

def get_optimized_hyperparams(
    model_params: dict,
    model_name: str,
    model_sklearn: str,
    species_type: str,
):

    if model_name == "SKLearn" and model_sklearn == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        model_params["model"] = RandomForestRegressor()
        model_params["hyperparams"] = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "bootstrap": True,
            "verbose": 0,
        }
    
    if model_name == "SKLearn" and model_sklearn == "XGBoost":
        from xgboost import XGBRegressor
        model_params["model"] = XGBRegressor()
        model_params["hyperparams"] = {
            "booster": 'gbtree',
            "n_estimators": 7,
            "max_leaves": 8,
            "min_child_weight": 2.685,
            "learning_rate": 0.4743,
            "subsample": 0.8852,
            "colsample_bylevel": 1.0,
            "colsample_bytree": 0.8929,
            "reg_alpha": 0.0471,
            "reg_lambda": 1.4083,
        }
    
    if model_name == "SKLearn" and model_sklearn == "LightGBM":
        from lightgbm import LGBMRegressor
        model_params["model"] = LGBMRegressor()
        model_params["hyperparams"] = {
            "n_estimators": 169,
            "num_leaves": 8,
            "min_child_samples": 3,
            "learning_rate": 0.0924,
            "colsample_bytree": 0.656,
            "reg_alpha": 0.0262,
            "reg_lambda": 0.1216,
            "verbose": -1,
        }
    
    if model_name == "WWLGPR" and species_type == "adsorbates":
        model_params["hyperparams"] = {
            "inner_weight": 1.00,
            "outer_weight": 0.88,
            "gpr_reg": 0.0186,
            "gpr_len": 12.9,
            "edge_s_s": 0.00,
            "edge_s_a": 0.38,
            "edge_a_a": 0.52,
        }
    
    if model_name == "WWLGPR" and species_type == "reactions":
        model_params["hyperparams"] = {
        
        #    "inner_weight": 0.60,
        #    "outer_weight": 0.05,
        #    "gpr_reg": 0.008,
        #    "gpr_len": 11.5,
        #    "edge_s_s": 0.00,
        #    "edge_s_a": 1.00,
        #    "edge_a_a": 0.70,
        
        #    "inner_weight": 1.00,
        #    "outer_weight": 0.88,
        #    "gpr_reg": 0.0186,
        #    "gpr_len": 12.9,
        #    "edge_s_s": 0.00,
        #    "edge_s_a": 0.38,
        #    "edge_a_a": 0.52,
        
        #    "inner_weight": 0.60,
        #    "outer_weight": 0.05,
        #    "gpr_reg": 0.0083,
        #    "gpr_len": 11.5,
        #    "edge_s_s": 0.00,
        #    "edge_s_a": 1.00,
        #    "edge_a_a": 0.70,
        
        #    "inner_weight": 0.73,
        #    "outer_weight": 0.39,
        #    "gpr_reg": 0.0182,
        #    "gpr_len": 16.6,
        #    "edge_s_s": 0.00,
        #    "edge_s_a": 0.52,
        #    "edge_a_a": 0.55,
        
        
        #    "inner_weight": 0.60,
        #    "outer_weight": 0.10,
        #    "gpr_reg": 0.0300,
        #    "gpr_len": 40.0,
        #    "edge_s_s": 0.50,
        #    "edge_s_a": 1.00,
        #    "edge_a_a": 0.70,
    
            "inner_weight": 0.60,
            "outer_weight": 0.02,
            "gpr_reg": 0.0300,
            "gpr_len": 22.0,
            "edge_s_s": 0.50,
            "edge_s_a": 1.00,
            "edge_a_a": 0.70,
        
        }
    return model_params
    
# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------