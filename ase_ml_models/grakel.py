# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase import Atoms
from grakel.kernels import WeisfeilerLehmanOptimalAssignment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

# -------------------------------------------------------------------------------------
# GRAKEL KERNEL
# -------------------------------------------------------------------------------------

class GrakelKernel(Kernel):
    """
    Grakel kernel for GPR.
    """
    def __init__(
        self,
        length_scale: int = 1.0,
        grakel_kernel: object = WeisfeilerLehmanOptimalAssignment(normalize=True),
    ):
        self.length_scale = length_scale
        self.grakel_kernel = grakel_kernel
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        self.grakel_kernel.fit_transform(X)
        kernel = self.grakel_kernel.transform(Y).T
        if eval_gradient:
            return kernel, np.zeros((kernel.shape[0], kernel.shape[1], 1))
        return kernel
    
    def diag(self, X):
        return np.ones(X.shape[0])
    
    def is_stationary(self):
        return True

    @property
    def requires_vector_input(self):
        return False

# -------------------------------------------------------------------------------------
# GRAKEL PREPROCESS
# -------------------------------------------------------------------------------------

def grakel_preprocess(
    atoms_list: list,
    kwargs_connectivity: dict = {"method": "ase", "bond_cutoff": 2},
):
    """
    Preprocess the atoms list for Grakel.
    """
    # Get the graph of adsorbed species.
    for atoms in atoms_list:
        atoms.info["graph"] = atoms_to_grakel_graph(
            atoms=atoms,
            kwargs_connectivity=kwargs_connectivity,
        )
        # Propagate features through the graph.
        #atoms.info["graph"] = propagate_features(atoms=atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# ATOMS TO GRAKEL GRAPH
# -------------------------------------------------------------------------------------

def atoms_to_grakel_graph(
    atoms: Atoms,
    kwargs_connectivity: dict = {"method": "ase", "bond_cutoff": 2},
):
    """
    Convert an ASE Atoms object to a grakel graph.
    """
    from grakel import Graph
    from ase_ml_models.utilities import (
        get_reduced_graph_atoms,
        get_edges_list_from_connectivity,
    )
    # Get the graph of adsorbed species.
    if "indices_ads" in atoms.info:
        atoms = get_reduced_graph_atoms(atoms=atoms, **kwargs_connectivity)
    # Get edges list from connectivity.
    edges = get_edges_list_from_connectivity(connectivity=atoms.info["connectivity"])
    # Get weighted edges list.
    if "indices_ads" in atoms.info:
        edges = get_weighted_edges(edges=edges, indices_ads=atoms.info["indices_ads"])
    else:
        edges = [tuple(ee) for ee in edges]
    node_labels = {ii: tuple(feat) for ii, feat in enumerate(atoms.info["features"])}
    graph = Graph(edges, node_labels=node_labels)
    return graph

# -------------------------------------------------------------------------------------
# GET WEIGHTED EDGES
# -------------------------------------------------------------------------------------

def get_weighted_edges(
    edges: list,
    indices_ads: list,
    weights: dict = {0: 0.2, 1: 1.0, 2: 0.5},
):
    """Get weighted edges from atoms."""
    return {tuple(ee): weights[len(set(ee) & set(indices_ads))] for ee in edges}

# -------------------------------------------------------------------------------------
# PROPAGATE FEATURES
# -------------------------------------------------------------------------------------

def propagate_features(
    atoms: Atoms,
):
    """
    Propagate features through the graph.
    """
    from grakel import Graph
    graph = atoms.info["graph"]
    features = atoms.info["features"]
    features_neighbors = np.zeros_like(features)
    for ii in range(len(graph)):
        neighbors = list(graph.neighbors(ii))
        if len(neighbors) > 0:
            features_neighbors[ii] = np.mean(features[neighbors], axis=0)
    features_all = np.hstack([features, features_neighbors])
    # Update the graph with new features.
    node_labels = {ii: tuple(feat) for ii, feat in enumerate(features_all)}
    atoms.info["graph"] = Graph(graph.get_edge_dictionary(), node_labels=node_labels)
    return atoms.info["graph"]

# -------------------------------------------------------------------------------------
# GRAKEL TRAIN
# -------------------------------------------------------------------------------------

def grakel_train(
    atoms_train: list,
    target: str = "E_form",
    grakel_kernel: object = WeisfeilerLehmanOptimalAssignment(normalize=True),
    hyperparams: dict = {"length_scale": 1e-3},
):
    """Train the Grakel model."""
    length_scale = hyperparams.get("length_scale", 1e-3) if hyperparams else 1e-3
    X_train = np.array([atoms.info["graph"] for atoms in atoms_train], dtype=object)
    y_train = np.array([atoms.info[target] for atoms in atoms_train])
    kernel = GrakelKernel(grakel_kernel=grakel_kernel, length_scale=length_scale)
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X=X_train, y=y_train)
    print("Model trained.")
    return model

# -------------------------------------------------------------------------------------
# GEAKEL PREDICT
# -------------------------------------------------------------------------------------

def grakel_predict(
    atoms_test: list,
    model: object,
    target: str = "E_form",
):
    """Predict the energies using the Grakel model."""
    X_test = np.array([atoms.info["graph"] for atoms in atoms_test], dtype=object)
    print("X_test shape:", X_test.shape)
    y_pred, sigma = model.predict(X=X_test, return_std=True)
    if target == "E_bind":
        y_pred = [yy+atoms.info["E_form_gas"] for yy, atoms in zip(y_pred, atoms_test)]
    elif target == "E_act":
        y_pred = [yy+atoms.info["E_first"] for yy, atoms in zip(y_pred, atoms_test)]
    return list(y_pred)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------