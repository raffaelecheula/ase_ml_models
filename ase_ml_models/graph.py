# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import networkx as nx
from ase import Atoms
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ase_ml_models.workflow import change_target_energy

# -------------------------------------------------------------------------------------
# COMPUTE KERNEL
# -------------------------------------------------------------------------------------

def compute_kernel(
    distances: float,
    length_scale: float = 1.0,
    kernel_type: str = "exponential",
    alpha: float = 1.0,
    **kwargs: dict,
) -> float:
    """
    Compute kernel from distances values.
    """
    if kernel_type == "exponential":
        return np.exp(-distances / length_scale)
    elif kernel_type == "gaussian":
        return np.exp(-distances / (2 * length_scale**2))
    elif kernel_type == "laplacian":
        return np.exp(-np.sqrt(distances) / length_scale)
    elif kernel_type == "matern-3/2":
        sqrt3 = np.sqrt(3)
        d_rel = distances / length_scale
        return (1 + sqrt3 * d_rel) * np.exp(-sqrt3 * d_rel)
    elif kernel_type == "matern-5/2":
        sqrt5 = np.sqrt(5)
        d_rel = distances / length_scale
        return (1 + sqrt5 * d_rel + (5/3) * d_rel**2) * np.exp(-sqrt5 * d_rel)
    elif kernel_type == "rational-quad":
        return (1 + distances**2 / (2 * alpha * length_scale**2))**(-alpha)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}.")

# -------------------------------------------------------------------------------------
# COMPUTE DISTANCE
# -------------------------------------------------------------------------------------

def compute_distance(
    features_a: np.ndarray,
    features_b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    method: str = "ot-emd2",
    reg: float = 0.1,
    c_unmatched: float = 1.0,
    **kwargs: dict,
) -> float:
    """
    Compute a single distance.
    """
    if method == "ot-emd2":
        # Earth Mover's Distance, solved with Python Optimal Transport (POT).
        import ot
        dist_matrix = ot.dist(x1=features_a, x2=features_b, metric="euclidean")
        dist = ot.emd2(a=weights_a, b=weights_b, M=dist_matrix)
    elif method == "ot-emd2-numpy":
        # Earth Mover's Distance (faster but compatible only with numpy arrays).
        from ot.utils import euclidean_distances
        from ot.lp import emd_c
        dist_matrix = euclidean_distances(X=features_a, Y=features_b, squared=False)
        dist = emd_c(weights_a, weights_b, dist_matrix, 100000, 1)[1]
    elif method == "ot-sinkhorn2":
        # Entropy-regularized optimal transport.
        import ot
        dist_matrix = ot.dist(x1=features_a, x2=features_b, metric="euclidean")
        dist = ot.sinkhorn2(a=weights_a, b=weights_b, M=dist_matrix, reg=reg)
    elif method == "scipy-wasserstein":
        # SciPy multivariate Wasserstein distance.
        from scipy.stats import wasserstein_distance_nd
        dist = wasserstein_distance_nd(
            u_values=features_a,
            v_values=features_b,
            u_weights=weights_a,
            v_weights=weights_b,
        )
    elif method == "scipy-lsa":
        # Hungarian algorithm to solve optimal assignment (least-cost matching).
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        # Solve the optimal assignment problem (Hungarian algorithm).
        dist_matrix = cdist(XA=features_a, XB=features_b, metric="euclidean")
        weight_matrix = np.outer(weights_a, weights_b)
        cost_matrix = dist_matrix * weight_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix)
        dist = (dist_matrix[row_ind, col_ind] * weight_matrix[row_ind, col_ind]).sum()
        # Find unmatched indices and add their weights to the distance.
        dist += c_unmatched * (
            weights_a[list(set(range(len(features_a)))-set(row_ind))].sum() + 
            weights_b[list(set(range(len(features_b)))-set(col_ind))].sum()
        )
    else:
        raise ValueError(f"Unknown method: {method}.")
    return dist

# -------------------------------------------------------------------------------------
# GRAPH KERNEL
# -------------------------------------------------------------------------------------

class GraphKernel(Kernel):
    """
    Graph kernel.
    """
    def __init__(
        self,
        length_scale: float = 10.0,
        **kwargs: dict,
    ):
        self.length_scale = length_scale
        self.kwargs = kwargs
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        # Get features and weights from Atoms objects.
        features_X = [np.array(atoms.info["features"]) for atoms in X]
        features_Y = [np.array(atoms.info["features"]) for atoms in Y]
        weights_X = [
            np.array(atoms.info["weights"]) / np.sum(atoms.info["weights"])
            for atoms in X
        ]
        weights_Y = [
            np.array(atoms.info["weights"]) / np.sum(atoms.info["weights"])
            for atoms in Y
        ]
        # Calculate distances matrix.
        distances = np.zeros((len(X), len(Y)))
        for ii, (features_a, weights_a) in enumerate(zip(features_X, weights_X)):
            for jj, (features_b, weights_b) in enumerate(zip(features_Y, weights_Y)):
                # Calculate graph distance.
                distances[ii, jj] = compute_distance(
                    features_a=features_a,
                    features_b=features_b,
                    weights_a=weights_a,
                    weights_b=weights_b,
                    **self.kwargs,
                )
        # Calculate kernel.
        kernel = compute_kernel(
            distances=distances,
            length_scale=self.length_scale,
            **self.kwargs,
        )
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
# COMPUTED DISTANCES TASK
# -------------------------------------------------------------------------------------

def compute_distance_task(
    args: list,
) -> tuple:
    """
    Compute a single distance (task for parallel runs).
    """
    ii, jj, features_a, features_b, weights_a, weights_b, kwargs = args
    # Calculate Wasserstein distance.
    dist = compute_distance(
        features_a=features_a,
        features_b=features_b,
        weights_a=weights_a,
        weights_b=weights_b,
        **kwargs,
    )
    return ii, jj, dist

# -------------------------------------------------------------------------------------
# PRECOMPUTED DISTANCES
# -------------------------------------------------------------------------------------

def precompute_distances(
    atoms_list: list,
    store_ids: bool = True,
    method: str = None,
    n_jobs: int = 8,
    **kwargs: dict,
) -> np.ndarray:
    """
    Precompute distances with serial or parallel jobs.
    """
    if method is None:
        method = "multiprocessing" if len(atoms_list) > 300 else "serial"
    # Get features and weights from Atoms objects.
    features = [np.array(atoms.info["features"]) for atoms in atoms_list]
    weights = [
        np.array(atoms.info["weights"]) / np.sum(atoms.info["weights"])
        for atoms in atoms_list
    ]
    # Prepare a list of args.
    nn = len(atoms_list)
    args_list = [
        (ii, jj, features[ii], features[jj], weights[ii], weights[jj], kwargs)
        for ii, jj in [(ii, jj) for ii in range(nn) for jj in range(ii+1, nn)]
    ]
    # Calculate distances with different parallelization methods.
    if method == "serial":
        # Compute distances without parallelization.
        results = [compute_distance_task(args) for args in args_list]
    if method == "multiprocessing":
        # Compute distances with multiprocessing.
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            results = pool.map(compute_distance_task, args_list)
    elif method == "joblib":
        # Compute distances with joblib.
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_distance_task)(args) for args in args_list
        )
    elif method == "ray":
        # Compute distances with ray.
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=n_jobs)
        compute_distance_ray = ray.remote(compute_distance_task)
        futures = [compute_distance_ray.remote(args) for args in args_list]
        results = ray.get(futures)
    # Calculate the distances matrix (symmetric, with zeros on the diagonal).
    distances = np.zeros((nn, nn))
    for ii, jj, dist in results:
        distances[ii, jj] = distances[jj, ii] = dist
    # Store IDs.
    if store_ids is True:
        for ii, atoms in enumerate(atoms_list):
            atoms.info["graph-ID"] = ii
    # Return the distances matrix.
    return distances

# -------------------------------------------------------------------------------------
# PRECOMPUTED DISTANCES GRAPH KERNEL
# -------------------------------------------------------------------------------------

class PrecomputedDistancesGraphKernel(Kernel):
    """
    Precomputed distances graph kernel.
    """
    def __init__(
        self,
        distances: np.ndarray,
        length_scale: float = 10.0,
        length_scale_bounds: tuple = "fixed",
        **kwargs: dict,
    ):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.distances = distances
        self.kwargs = kwargs
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        if isinstance(X[0], Atoms):
            X = [atoms.info["graph-ID"] for atoms in X]
        if isinstance(Y[0], Atoms):
            Y = [atoms.info["graph-ID"] for atoms in Y]
        X = np.asarray(X, dtype=int)
        Y = np.asarray(Y, dtype=int)
        distances = self.distances[np.ix_(X, Y)]
        kernel = compute_kernel(
            distances=distances,
            length_scale=self.length_scale,
            **self.kwargs,
        )
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return kernel, np.zeros((kernel.shape[0], kernel.shape[1], 1))
            else:
                grad = (kernel * distances) / (self.length_scale**2)
                return kernel, grad[:, :, np.newaxis]
        return kernel
    
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)
    
    def diag(self, X):
        return np.ones(X.shape[0])
    
    def is_stationary(self):
        return True

    @property
    def requires_vector_input(self):
        return False

# -------------------------------------------------------------------------------------
# ATOMS TO NETWORKX GRAPH
# -------------------------------------------------------------------------------------

def atoms_to_nx_graph(
    atoms: Atoms,
) -> object:
    """
    Convert an ASE Atoms object to a networkx graph.
    """
    from ase_ml_models.utilities import get_edges_list_from_connectivity
    graph = nx.Graph()
    for ii, atom in enumerate(atoms):
        graph.add_node(ii)
    for edge in get_edges_list_from_connectivity(atoms.info["connectivity"]):
        graph.add_edge(*edge)
    return graph

# -------------------------------------------------------------------------------------
# GET NODE WEIGHTS
# -------------------------------------------------------------------------------------

def get_node_weights(
    graph: nx.Graph,
    indices: list = None,
    node_weight_dict: dict = {"A0": 1.00, "S1": 0.80, "S2": 0.20},
) -> np.ndarray:
    """
    Calculate node weights, according to `node_weight_dict`. For adsorbates, the
    `indices` are the indices of the adsorbate atoms in the Atoms object. The
    `node_weight_dict` is a dictionary that maps the weight type to the
    corresponding weight value, where `A0` corresponds to adsorbate atoms, `S1` to 
    atoms in the first shell, and `S2` to atoms in the second shell.
    """
    if indices is None:
        weights = np.ones(len(graph.nodes()))
    else:
        weight_keys = sorted(node_weight_dict.keys(), key=lambda x: int(x[1:]))
        node_weight_list = [node_weight_dict[ii] for ii in weight_keys]
        indices_assigned = []
        weights = np.zeros(len(graph.nodes()))
        indices_ii = [int(ii) for ii in indices]
        ii_max = len(node_weight_list)-1
        for ii in range(ii_max):
            weights[indices_ii] = node_weight_list[ii]
            indices_assigned += indices_ii
            indices_ii = [
                jj for kk in indices_ii for jj in graph.neighbors(kk)
                if jj not in indices_assigned
            ]
        weights[indices_ii] = node_weight_list[ii_max]
    return weights

# -------------------------------------------------------------------------------------
# GET EDGE WEIGHTS
# -------------------------------------------------------------------------------------

def get_edge_weights(
    graph: nx.Graph,
    indices: list = None,
    edge_weight_dict: dict = {"AA": 0.50, "AS": 1.00, "SS": 0.50},
) -> dict:
    """
    Get edges weights, according to `edge_weight_dict`. For adsorbates, the 
    `indices` are the indices of the adsorbate atoms in the Atoms object. The 
    `edge_weight_dict` is a dictionary that maps the weight type and the
    corresponding edge weight value, where `AA` corresponds to edges between adsorbate
    atoms, `AS` to edges between adsorbate and surface, and `SS` to edges between 
    surface atoms.
    """
    edges = list(graph.edges())
    if indices is None:
        edge_weights = {tuple(sorted(ee)): 1.0 for ee in graph.edges()}
    else:
        edge_weight_list = [edge_weight_dict[ii] for ii in ["SS", "AS", "AA"]]
        edge_weights = {
            tuple(sorted(ee)): edge_weight_list[len(set(ee) & set(indices))]
            for ee in edges
        }
    return edge_weights

# -------------------------------------------------------------------------------------
# PROPAGATE FEATURES
# -------------------------------------------------------------------------------------

def propagate_features(
    graph: nx.Graph,
    features: np.ndarray,
    use_edge_weights: bool = True,
    stack_features: bool = True,
    weight_neigh: float = 0.1,
    edge_weights: dict = None,
    indices: list = None,
    edge_weight_dict: dict = {"AA": 0.50, "AS": 1.00, "SS": 0.50},
    n_iter: int = 1,
    nan: float = 0.0,
) -> np.ndarray:
    """
    Propagate features through the graph with a Weisfeiler-Lehman scheme.
    """
    # If using edge weights and none are provided, compute them.
    if use_edge_weights is True and edge_weights is None:
        edge_weights = get_edge_weights(
            graph=graph,
            indices=indices,
            edge_weight_dict=edge_weight_dict,
        )
    # Replace any NaNs in the features with a fixed value.
    features = np.nan_to_num(features.copy(), nan=nan)
    # Run Weisfeiler-Lehman propagation for n_iter steps.
    for jj in range(n_iter):
        # Initialize the propagated features array.
        features_neighbors = np.zeros_like(features)
        # Loop over all nodes.
        for ii in range(len(graph.nodes())):
            neighbors = list(graph.neighbors(ii))
            if len(neighbors) > 0:
                # If using edge weights, retrieve and normalize them.
                if use_edge_weights is True:
                    weights = np.array(
                        [edge_weights[tuple(sorted((ii, jj)))] for jj in neighbors]
                    )
                else:
                    weights = None
                # Compute (weighted) average of neighbor features.
                features_neighbors[ii] = np.average(
                    features[neighbors],
                    weights=weights,
                    axis=0,
                )
            else:
                # If no neighbors, keep the original feature.
                features_neighbors[ii] = features[ii].copy()
        # Combine original and neighbor features.
        if stack_features:
            # Concatenate propagated features to the original feature set.
            features = np.hstack([features, features_neighbors])
        else:
            # Update features with a blend of original and neighbor-averaged features.
            features = (1-weight_neigh) * features + weight_neigh * features_neighbors
    return features

# -------------------------------------------------------------------------------------
# GRAPH PREPROCESS
# -------------------------------------------------------------------------------------

def graph_preprocess(
    atoms_list: list,
    node_weight_dict: dict = {"A0": 1.00, "S1": 0.80, "S2": 0.20},
    preproc: object = MinMaxScaler(feature_range=(-1, +1)),
    use_edge_weights: bool = True,
    stack_features: bool = True,
    weight_neigh: float = 0.1,
    n_iter: int = 1,
    nan: float = 0.,
):
    """
    Preprocess the Atoms objects for graph-based machine learning.
    """
    for atoms in atoms_list:
        graph = atoms_to_nx_graph(atoms=atoms)
        atoms.info["weights"] = get_node_weights(
            graph=graph,
            indices=atoms.info["indices_ads"],
            node_weight_dict=node_weight_dict,
        )
        atoms.info["features"] = propagate_features(
            graph=graph,
            features=atoms.info["features"],
            use_edge_weights=use_edge_weights,
            indices=atoms.info["indices_ads"],
            stack_features=stack_features,
            weight_neigh=weight_neigh,
            n_iter=n_iter,
            nan=nan,
        )
    # Preprocess features.
    features_all = np.vstack([atoms.info["features"] for atoms in atoms_list])
    preproc.fit(features_all)
    for atoms in atoms_list:
        atoms.info["features"] = preproc.transform(atoms.info["features"])

# -------------------------------------------------------------------------------------
# GRAPH TRAIN
# -------------------------------------------------------------------------------------

def graph_train(
    atoms_train: list,
    target: str = "E_form",
    kwargs_model: dict = {},
    kwargs_kernel: dict = {},
    model_name: str = "GPR",
    distances: np.ndarray = None,
    **kwargs: dict,
):
    """Train the Graph model."""
    # Prepare the data.
    X_train = np.array([atoms for atoms in atoms_train], dtype=object)
    y_train = np.array([atoms.info[target] for atoms in atoms_train])
    # Graph kernel.
    if distances is not None:
        kernel = PrecomputedDistancesGraphKernel(distances=distances, **kwargs_kernel)
    else:
        kernel = GraphKernel(**kwargs_kernel)
    # Kernel-based models.
    if model_name == "GPR":
        from sklearn.gaussian_process import GaussianProcessRegressor
        # Gaussian Process Regression model.
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=kwargs_model.pop("alpha", 0.001),
            copy_X_train=kwargs_model.pop("copy_X_train", False),
            normalize_y=kwargs_model.pop("normalize_y", True),
            **kwargs_model,
        )
    elif model_name == "KRR":
        from sklearn.kernel_ridge import KernelRidge
        # Kernel Ridge Regression model.
        model = KernelRidge(
            kernel="precomputed",
            alpha=kwargs_model.pop("alpha", 0.001),
            **kwargs_model,
        )
        # Store kernel and training data in the model.
        model.info = {"kernel": kernel, "X_train": X_train}
        # Precompute the kernel.
        X_train = kernel(X=X_train)
    elif model_name == "SVR":
        from sklearn.svm import SVR
        # Support Vector Regression model.
        model = SVR(
            kernel="precomputed",
            **kwargs_model,
        )
        # Store kernel and training data in the model.
        model.info = {"kernel": kernel, "X_train": X_train}
        # Precompute the kernel.
        X_train = kernel(X=X_train)
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
    # Fit the model.
    model.fit(X=X_train, y=y_train)
    # Return the model.
    return model

# -------------------------------------------------------------------------------------
# GRAPH PREDICT
# -------------------------------------------------------------------------------------

def graph_predict(
    atoms_test: list,
    model: object,
    target: str = "E_form",
    **kwargs: dict,
):
    """Predict the energies using the Graph model."""
    X_test = np.array([atoms for atoms in atoms_test], dtype=object)
    if model.kernel == "precomputed":
        X_test = model.info["kernel"](X=X_test, Y=model.info["X_train"])
    y_pred = model.predict(X=X_test)
    # Transform the predicted values.
    y_pred = change_target_energy(y_pred=y_pred, atoms_test=atoms_test, target=target)
    # Return predicted formation energies.
    return y_pred

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------