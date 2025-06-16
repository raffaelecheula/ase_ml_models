# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import torch
import numpy as np
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINConv,
    GINEConv,
    GraphConv,
    GATConv,
    GATv2Conv,
    SAGEConv,
    ChebConv, 
    TAGConv,
    ARMAConv,
    GCNConv,
    TransformerConv, 
    GMMConv,
    AGNNConv,
    MFConv,
    GatedGraphConv,
    global_mean_pool,
)

from ase_ml_models.workflow import change_target_energy
from ase_ml_models.utilities import get_edges_list_from_connectivity

# -------------------------------------------------------------------------------------
# PYG REGRESSION
# -------------------------------------------------------------------------------------

class PyGRegression(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        n_conv_layers: int = 3,
        n_lin_layers: int = 2,
        conv_type: str = "ARMAConv",
        dropout: float = 0.0,
        activation: callable = torch.nn.functional.relu,
        pooling: callable = global_mean_pool,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        conv_kwargs: dict = {},
        seed: int = 0,
    ):
        super().__init__()
        n_conv_layers = n_conv_layers if n_conv_layers > 1 else 1
        self.n_conv_layers = n_conv_layers
        self.n_lin_layers = n_lin_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.activation = activation
        self.pooling = pooling
        # Set the random seed for reproducibility.
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Dictionary of convolutional layers.
        conv_dict = {
            "GINConv": GINConv,
            "GINEConv": GINEConv,
            "GraphConv": GraphConv,
            "GATConv": GATConv,
            "GATv2Conv": GATv2Conv,
            "SAGEConv": SAGEConv,
            "ChebConv": ChebConv, # Needs K in conv_kwargs.
            "TAGConv": TAGConv,
            "ARMAConv": ARMAConv,
            "GCNConv": GCNConv,
            "TransformerConv": TransformerConv,
            "AGNNConv": AGNNConv,
            "MFConv": MFConv,
            "GatedGraphConv": GatedGraphConv,
        }
        # Get the convolutional layer class from the dictionary.
        self.conv_type = conv_type
        self.conv_class = conv_dict[conv_type]
        self.conv_kwargs = conv_kwargs or {}
        # Initialize the convolutional layers and batch normalization layers.
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList() if use_batch_norm else None
        # AGNNConv and GatedGraphConv need an input projection.
        if conv_type in ("AGNNConv", "GatedGraphConv"):
            self.input_proj = torch.nn.Linear(num_node_features, hidden_dim)
        else:
            self.input_proj = None
        # Build the convolutional layers.
        for ii in range(n_conv_layers):
            # GATConv and GATv2Conv need special dimensions.
            concat = self.conv_kwargs.get("concat", True)
            heads = self.conv_kwargs.get("heads", 1) if concat else 1
            if "heads" in self.conv_kwargs and ii == n_conv_layers - 1:
                self.conv_kwargs["heads"] = 1
            # Set the input and output channels for the convolutional layer.
            in_channels = num_node_features if ii == 0 else hidden_dim * heads
            out_channels = (
                hidden_dim if ii < n_conv_layers - 1 or n_lin_layers > 0 else 1
            )
            # GINConv needs a Sequential layer.
            if self.conv_type in ("GINConv", "GINEConv"):
                nn = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, out_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_channels, out_channels),
                )
                conv = self.conv_class(nn, **self.conv_kwargs)
            # GatedGraphConv handles multiple layers internally.
            elif self.conv_type == "GatedGraphConv":
                conv = self.conv_class(hidden_dim, n_conv_layers)
                self.n_conv_layers = 1
                if ii != 0:
                    continue
            # For other convolution types, use the default behavior.
            else:
                conv = self.conv_class(in_channels, out_channels, **self.conv_kwargs)
            # Append the convolutional layer to the list.
            self.convs.append(conv)
            # If batch normalization is used, append a BatchNorm layer.
            if use_batch_norm is True:
                self.batch_norms.append(torch.nn.BatchNorm1d(out_channels))
        # Define the linear layers.
        self.lins = []
        for ii in range(n_lin_layers):
            # Set the input and output channels for the linear layer.
            in_channels = hidden_dim
            out_channels = hidden_dim if ii < n_lin_layers - 1 else 1
            lin = torch.nn.Linear(in_channels, out_channels)
            self.lins.append(lin)
        # Define the dropout layer.
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, data):
        # Unpack the data object.
        xx, edge_index, batch = data.x, data.edge_index, data.batch
        # Input projection.
        if self.input_proj is not None:
            xx = self.input_proj(xx)
        # Convolutional layers.
        for ii in range(self.n_conv_layers):
            xx_in = xx
            # Apply the convolutional layer.
            xx = self.convs[ii](xx, edge_index)
            if ii == self.n_conv_layers - 1 and self.n_lin_layers < 1:
                break
            # Apply batch normalization.
            if self.use_batch_norm is True:
                xx = self.batch_norms[ii](xx)
            # Apply activation function.
            xx = self.activation(xx)
            # Apply residual connection.
            if self.use_residual is True and xx.shape == xx_in.shape:
                xx = xx + xx_in
            # Apply dropout.
            if self.dropout > 0:
                xx = self.dropout_layer(xx)
        # Apply pooling.
        xx = self.pooling(xx, batch)
        # Linear layers.
        for ii in range(self.n_lin_layers):
            # Apply the convolutional layer.
            xx = self.lins[ii](xx)
            if ii == self.n_lin_layers - 1:
                break
            # Apply activation function.
            xx = self.activation(xx)
            # Apply dropout.
            if self.dropout > 0:
                xx = self.dropout_layer(xx)
        # Return the output.
        return xx.squeeze()

# -------------------------------------------------------------------------------------
# ASE TO PYG DATA
# -------------------------------------------------------------------------------------

def atoms_to_pyg_data(
    atoms: Atoms,
    target: str = "E_form",
):
    """Convert ASE atoms to a PyTorch Geometric Data object."""
    features_np = np.nan_to_num(atoms.info["features"], nan=-1.0)
    features = torch.tensor(features_np, dtype=torch.float)
    edges = get_edges_list_from_connectivity(atoms.info["connectivity"])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    yy = torch.tensor([atoms.info[target]], dtype=torch.float)
    return Data(x=features, edge_index=edge_index, y=yy)

# -------------------------------------------------------------------------------------
# CREATE PYG DATASET
# -------------------------------------------------------------------------------------

def create_pyg_dataset(
    atoms_list: list,
    target: str = "E_form",
):
    """Create a PyTorch Geometric dataset from a list of ASE atoms."""
    return [atoms_to_pyg_data(atoms=atoms, target=target) for atoms in atoms_list]

# -------------------------------------------------------------------------------------
# PYG TRAIN
# -------------------------------------------------------------------------------------

def pyg_train(
    atoms_train: list,
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    target: str = "E_form",
    kwargs_model: dict = {},
    kwargs_scheduler: dict = {"factor": 0.5, "patience": 5},
    **kwargs,
):
    """Train a PyTorch Geometric model on a list of ASE atoms."""
    # Create dataset and dataloader from ASE atoms list.
    dataset = create_pyg_dataset(atoms_train, target=target)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # Initialize the PyTorch Geometric model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyGRegression(
        num_node_features=dataset[0].num_node_features,
        **(kwargs_model if kwargs_model else {}),
    ).to(device)
    # Define loss function, optimizer, and learning rate scheduler.
    #loss_fn = torch.nn.MSELoss(reduction="mean")
    loss_fn = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        **(kwargs_scheduler if kwargs_scheduler else {}),
    )
    # Train the PyTorch Geometric model.
    print("Training the PyG model.")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(dataset)
        scheduler.step(train_loss)
        print(f"Epoch {epoch+1:4d}: Loss = {train_loss:.4f}")
    # Print final learning rate.
    print(f"Final learning rate: {scheduler.get_last_lr()[0]:6.4e}")
    # Return the trained PyTorch Geometric model.
    return model

# -------------------------------------------------------------------------------------
# PYG TRAIN WITH EARLY STOPPING
# -------------------------------------------------------------------------------------

def pyg_train_with_early_stopping(
    atoms_train: list,
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    target: str = "E_form",
    hyperparams: dict = {},
    val_split: float = 0.1,
    early_stopping_patience: int = 10,
    early_stopping_delta: float = 1e-4,
    save_best_path: str = None,
):
    """Train a PyTorch Geometric model with optional validation and early stopping."""
    # Create tran and val datasets and dataloaders from ASE atoms list.
    from sklearn.model_selection import train_test_split
    atoms_train, atoms_val = train_test_split(atoms_train, test_size=val_split)
    dataset_train = create_pyg_dataset(atoms_train, target=target)
    dataset_val = create_pyg_dataset(atoms_val, target=target)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size)
    loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size)
    # Initialize the PyTorch Geometric model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = hyperparams if hyperparams else {}
    model = PyGRegression(
        num_node_features=dataset_train[0].num_node_features,
        **hyperparams,
    ).to(device)
    # Train the PyTorch Geometric model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    print("Training PyG model with validation and early stopping.")
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(num_epochs):
        # Train.
        model.train()
        total_loss = 0
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(dataset_train)
        # Validation.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)
                out = model(batch)
                val_loss += loss_fn(out, batch.y).item() * batch.num_graphs
        val_loss /= len(dataset_val)
        print(f"Epoch {epoch+1:4d}: Train = {train_loss:.4f} Val = {val_loss:.4f}")
        # Early stopping check.
        if val_loss + early_stopping_delta < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    # Return the trained PyTorch Geometric model.
    return model

# -------------------------------------------------------------------------------------
# PYG PREDICT
# -------------------------------------------------------------------------------------

def pyg_predict(
    atoms_test: list,
    model: object,
    target: str = "E_form",
    **kwargs,
):
    """Predict energies using the PyTorch Geometric model."""
    # Create dataset and dataloader from ASE atoms list.
    dataset_test = create_pyg_dataset(atoms_test, target=target)
    loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    # Predict the target values.
    device = next(model.parameters()).device
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            y_pred.append(pred.cpu().item())
    # Transform the predicted values.
    y_pred = change_target_energy(y_pred=y_pred, atoms_test=atoms_test, target=target)
    # Return predicted formation energies.
    return y_pred

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------