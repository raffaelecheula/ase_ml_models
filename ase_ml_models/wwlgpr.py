# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np

# -------------------------------------------------------------------------------------
# WWL-GPR HYPERPARAMETERS
# -------------------------------------------------------------------------------------

def optimize_hyperpars_wwlgpr(
    model: object,
    n_calls: int = 100,
):
    """Set the hyperparameters for the WWL-GPR model."""
    import ray
    from skopt.space import Real, Integer
    from wwlgpr.WWL_GPR import BayOptCv
    ray.shutdown()
    ray.init()
    print("Nodes in the Ray cluster:", ray.nodes())
    # Hyperparameters.
    inner_weight = Real(
        name="Inner bond cutoff weight.",
        low=0,
        high=1,
        prior="uniform",
    )
    outer_weight = Real(
        name="Outer bond cutoff weight.",
        low=0,
        high=1,
        prior="uniform",
    )
    gpr_reg = Real(
        name="GPR regularization.",
        low=1e-3,
        high=1e0,
        prior="uniform",
    )
    gpr_len = Real(
        name="GPR lengthscale.",
        low=1,
        high=100,
        prior="uniform",
    )
    edge_s_s = Real(
        name="Surface-surface edge weight.",
        low=0,
        high=1,
        prior="uniform",
    )
    edge_s_a = Real(
        name="Surface-adsorbate edge weight.",
        low=0,
        high=1,
        prior="uniform",
    )
    edge_a_a = Real(
        name="Adsorbate-adsorbate edge weight.",
        low=0,
        high=1,
        prior="uniform",
    )
    # Fix hyperparameters.
    fix_hypers = {"cutoff": 2, "inner_cutoff": 1, "gpr_sigma": 1}
    # Return hyperparameters.
    opt_dimensions = {
        "inner_weight": inner_weight,
        "outer_weight": outer_weight,
        "gpr_reg": gpr_reg,
        "gpr_len": gpr_len,
        "edge_s_s": edge_s_s,
        "edge_s_a": edge_s_a,
        "edge_a_a": edge_a_a,
    }
    # Default hyperparameters.
    default_para = [
        [1.0, 0.000, 0.0300, 30.00, 0.00, 1.00, 0.00],
        [0.6, 0.054, 0.0082, 11.47, 0.00, 1.00, 0.70],
    ]
    # Optimize hyperparameters.
    res_opt = model.BayOpt(
        opt_dimensions=opt_dimensions,
        default_para=default_para,
        fix_hypers=fix_hypers,
        n_calls=n_calls,
    )
    opt_hypers = dict(zip(opt_dimensions.keys(), res_opt.x))
    print("Hyperparameters:", opt_hypers)
    BayOptCv.default_gpr_hyper.update(opt_hypers)
    return opt_hypers

# -------------------------------------------------------------------------------------
# WWL-GPR TRAIN
# -------------------------------------------------------------------------------------

def wwlgpr_train(
    atoms_train: list,
    num_cpus: int = 8,
    num_iter: int = 1,
    pre_data_type: str = "Standardize",
    drop_list: list = None,
    target: str = "E_form",
    hyperparams: dict = None,
    optimize_hyperpars: bool = False,
    n_calls: int = 100,
):
    """Train the WWL-GPR model."""
    from igraph import Graph
    from wwlgpr.WWL_GPR import BayOptCv
    train_db_graphs = [
        Graph.Adjacency(atoms.info["connectivity"]) for atoms in atoms_train
    ]
    node_attributes = [atoms.info["features"] for atoms in atoms_train]
    energies = [atoms.info[target] for atoms in atoms_train]
    filenames = [atoms.info["name"] for atoms in atoms_train]
    model = BayOptCv(
        num_cpus=num_cpus,
        db_graphs=train_db_graphs,
        db_atoms=np.array(atoms_train, dtype=object),
        node_attributes=node_attributes,
        y=energies,
        drop_list=drop_list,
        num_iter=num_iter,
        pre_data_type=pre_data_type,
        filenames=filenames,
    )
    if hyperparams is not None:
        BayOptCv.default_gpr_hyper.update(hyperparams)
    if optimize_hyperpars is True:
        optimize_hyperpars_wwlgpr(model=model, n_calls=n_calls)
    print("WWL-GPR model trained.")
    return model

# -------------------------------------------------------------------------------------
# WWL-GPR PREDICT
# -------------------------------------------------------------------------------------

def wwlgpr_predict(
    atoms_test: list,
    model: object,
    opt_hypers: dict = {},
    target: str = "E_form",
):
    """Predict the energies using the WWL-GPR model."""
    from igraph import Graph
    test_graphs = [
        Graph.Adjacency(atoms.info["connectivity"]) for atoms in atoms_test
    ]
    test_node_attributes = [atoms.info["features"] for atoms in atoms_test]
    test_energies = [atoms.info[target] for atoms in atoms_test]
    test_file_names = [atoms.info["name"] for atoms in atoms_test]
    test_RMSE, y_pred = model.Predict(
        test_graphs=test_graphs,
        test_atoms=np.array(atoms_test, dtype=object),
        test_node_attributes=test_node_attributes,
        test_target=test_energies,
        opt_hypers=opt_hypers,
    )
    print(f"Test RMSE: {test_RMSE:7.4f} eV")
    if target == "E_act":
        y_pred = [y+atoms.info["E_first"] for y, atoms in zip(y_pred, atoms_test)]
    return list(y_pred)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------