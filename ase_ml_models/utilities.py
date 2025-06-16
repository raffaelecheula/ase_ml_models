# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY ASE
# -------------------------------------------------------------------------------------

def get_connectivity_ase(
    atoms: Atoms,
    indices: list = None,
    cutoffs_dict: dict = {},
    skin: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """ Get the connectivity matrix for an ase Atoms object."""
    from ase.neighborlist import NeighborList
    # Get cutoffs.
    cutoffs = natural_cutoffs(atoms, **cutoffs_dict)
    if indices is not None:
        cutoffs = [cc if ii in indices else 0. for ii, cc in enumerate(cutoffs)]
    # Calculate connectivity matrix.
    nlist = NeighborList(
        cutoffs=cutoffs,
        skin=skin,
        sorted=False,
        self_interaction=False,
        bothways=True,
    )
    nlist.update(atoms)
    return nlist.get_connectivity_matrix(sparse=False)

# -------------------------------------------------------------------------------------
# GET EDGES LIST THRESHOLD
# -------------------------------------------------------------------------------------

def get_edges_list_threshold(
    atoms: Atoms,
    indices: list = None,
    dist_ratio_thr: float = 1.25,
) -> list:
    """Get the edges for selected atoms in an ase Atoms object."""
    from itertools import combinations
    if indices is None:
        indices = range(len(atoms))
    indices = [int(ii) for ii in indices]
    edges_list = []
    cutoffs = natural_cutoffs(atoms=atoms)
    for combo in combinations(list(indices), 2):
        total_distance = atoms.get_distance(combo[0], combo[1], mic=True)
        r1 = cutoffs[combo[0]]
        r2 = cutoffs[combo[1]]
        distance_ratio = total_distance / (r1 + r2)
        if distance_ratio <= dist_ratio_thr:
            edges_list.append([int(ii) for ii in combo])
    return edges_list

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY FROM EDGES LIST
# -------------------------------------------------------------------------------------

def get_connectivity_from_edges_list(
    atoms: Atoms,
    edges_list: list,
) -> np.ndarray:
    """Get the connectivity matrix from a list of edges."""
    connectivity = np.zeros((len(atoms), len(atoms)), dtype=int)
    for aa, bb in edges_list:
        connectivity[aa, bb] += 1
        connectivity[bb, aa] += 1
    return connectivity

# -------------------------------------------------------------------------------------
# GET EDGES LIST FROM CONNECTIVITY
# -------------------------------------------------------------------------------------

def get_edges_list_from_connectivity(
    connectivity: np.ndarray,
) -> list:
    """Get the connectivity matrix from a list of edges."""
    edges_list = []
    for aa, bb in np.argwhere(connectivity > 0):
        if bb > aa:
            edges_list += [[int(aa), int(bb)]] * int(connectivity[aa, bb])
    return edges_list

# -------------------------------------------------------------------------------------
# GET CONNECTVITY THRESHOLD
# -------------------------------------------------------------------------------------

def get_connectivity_threshold(
    atoms: Atoms,
    edges_list: list = None,
    indices: list = None,
    dist_ratio_thr: float = 1.25,
    **kwargs,
) -> np.ndarray:
    """Get the connectivity matrix for selected atoms in an ase Atoms object."""
    if edges_list is None:
        edges_list = get_edges_list_threshold(
            atoms=atoms,
            indices=indices,
            dist_ratio_thr=dist_ratio_thr,
        )
    connectivity = get_connectivity_from_edges_list(
        atoms=atoms,
        edges_list=edges_list,
    )
    return connectivity

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY
# -------------------------------------------------------------------------------------

def get_connectivity(
    atoms: Atoms,
    method: str = "ase",
    ensure_bonding: bool = True,
    **kwargs,
) -> np.ndarray:
    """Get the connectivity matrix for an ase Atoms object."""
    # Get the connectivity.
    if method == "ase":
        connectivity = get_connectivity_ase(atoms=atoms, **kwargs)
    elif method == "threshold":
        connectivity = get_connectivity_threshold(atoms=atoms, **kwargs)
    # Ensure bonding.
    if ensure_bonding is True:
        connectivity = ensure_bonding_ads_surf(atoms=atoms, connectivity=connectivity)
    return connectivity

# -------------------------------------------------------------------------------------
# GET EDGES LIST
# -------------------------------------------------------------------------------------

def get_edges_list(
    atoms: Atoms,
    method: str = "threshold",
    **kwargs,
) -> np.ndarray:
    """Get the connectivity matrix for an ase Atoms object."""
    # Get the connectivity.
    if method == "ase":
        connectivity = get_connectivity_ase(atoms=atoms, **kwargs)
        edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    elif method == "threshold":
        edges_list = get_edges_list_threshold(atoms=atoms, **kwargs)
    return edges_list

# -------------------------------------------------------------------------------------
# ENSURE BONDING ADS SURF
# -------------------------------------------------------------------------------------

def ensure_bonding_ads_surf(
    atoms: Atoms,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Ensure bonding between adsorbates and surface atoms."""
    if "indices_ads" not in atoms.info.keys() or len(atoms.info["indices_ads"]) == 0:
        return connectivity
    indices_ads = atoms.info["indices_ads"]
    indices_surf = [ii for ii in range(len(atoms)) if ii not in indices_ads]
    n_bonds = len([
        ii for ii in indices_surf for jj in indices_ads if connectivity[ii, jj] > 0
    ])
    if n_bonds < 1:
        # Sort atoms by z-coordinate.
        atoms_ads = [aa for aa in atoms if aa.index in indices_ads]
        atoms_surf = [aa for aa in atoms if aa.index in indices_surf]
        aa = sorted(atoms_ads, key=lambda a: a.position[2])[0].index
        bb = sorted(
            atoms_surf, key=lambda a: np.linalg.norm(a.position-atoms[aa].position)
        )[0].index
        connectivity[aa, bb] = 1
        connectivity[bb, aa] = 1
    return connectivity

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY FROM LIST
# -------------------------------------------------------------------------------------

def get_connectivity_from_list(
    atoms_list: list,
    method: str = "ase",
    ensure_bonding: bool = True,
    sum_connectivity: bool = False,
    **kwargs,
):
    """Get connectivity from a list of atoms."""
    for ii, atoms in enumerate(atoms_list):
        if "connectivity" in atoms.info:
            connectivity_ii = atoms.info["connectivity"]
        else:
            connectivity_ii = get_connectivity(
                atoms=atoms,
                method=method,
                ensure_bonding=ensure_bonding,
                **kwargs,
            )
        if ii == 0:
            connectivity = connectivity_ii
        elif sum_connectivity is True:
            connectivity += connectivity_ii
        else:
            for jj, kk in zip(*np.where(connectivity_ii > connectivity)):
                connectivity[jj, kk] = connectivity_ii[jj, kk]
    return connectivity

# -------------------------------------------------------------------------------------
# PLOT CONNECTIVITY
# -------------------------------------------------------------------------------------

def plot_connectivity(
    atoms: Atoms,
    connectivity: np.ndarray = None,
    edges_pbc: bool = False,
    show_plot: bool = True,
    show_axis: bool = False,
    colors: str = "jmol",
    alpha: float = None,
    scale_radii: float = 100,
):
    """Plot the atoms and bonds of an ase.Atoms object."""
    import matplotlib.pyplot as plt
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors
    # Get edges list.
    if connectivity is None:
        connectivity = atoms.info["connectivity"]
    edges_list = get_edges_list_from_connectivity(connectivity=connectivity)
    # Delete edges from pbc.
    if edges_pbc is False:
        remove = []
        for ii, (a0, a1) in enumerate(edges_list):
            distance = atoms.get_distance(a0=a0, a1=a1, mic=False)
            distance_mic = atoms.get_distance(a0=a0, a1=a1, mic=True)
            if distance > distance_mic+1e-6:
                remove.append(ii)
        edges_list = [edge for ii, edge in enumerate(edges_list) if ii not in remove]
    # Get the radii, and colors.
    radii = covalent_radii[atoms.numbers]
    if colors == "jmol":
        colors = jmol_colors[atoms.numbers]
    # Get the 3D edges.
    edges_xyz = np.array([
        (atoms.positions[a0], atoms.positions[a1]) for a0, a1 in edges_list
    ])
    # Prepare a figure.
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(projection="3d")
    # Plot the nodes.
    ax.scatter(*atoms.positions.T, s=scale_radii*radii, c=colors, ec="k", alpha=alpha)
    # Plot the edges.
    for edge in edges_xyz:
        ax.plot(*edge.T, color="grey")
    # Adjust the figure.
    ax.grid(False)
    if show_axis is True:
        for ax_i in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_i.set_ticks([])
            ax_i.set_alpha(0.)
            ax_i.pane.fill = False
            ax_i.pane.set_alpha(0.)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    # Show the figure.
    if show_plot is True:
        plt.show()
    # Return the axis.
    return ax

# -------------------------------------------------------------------------------------
# ENLARGE SURFACE
# -------------------------------------------------------------------------------------

def enlarge_surface(
    atoms: Atoms,
) -> Atoms:
    """Enlarge the structure by adding atoms at the cell boundaries."""
    atoms_enlarged = atoms.copy()
    for ii in [-1, 0, 1]:
        for jj in [-1, 0, 1]:
            if ii == jj == 0:
                continue
            atoms_copy = atoms.copy()
            atoms_copy.translate(np.dot([ii, jj, 0], atoms.cell))
            atoms_enlarged += atoms_copy
    atoms_enlarged.pbc = False
    atoms_enlarged.info["indices_original"] = list(range(len(atoms))) * 9
    return atoms_enlarged

# -------------------------------------------------------------------------------------
# GET INDICES FROM BOND CUTOFF
# -------------------------------------------------------------------------------------

def get_indices_from_bond_cutoff(
    atoms,
    connectivity,
    indices_ads,
    bond_cutoff: int = 2,
    return_list: bool = False,
):
    """Get the indices of atoms within a certain number of bonds."""
    indices_all = []
    indices_dict = {}
    indices_dict[0] = indices_ads
    indices_all += list(indices_dict[0])
    for ii in range(bond_cutoff):
        indices = np.where(connectivity[indices_dict[ii], :] > 0)[1]
        indices_dict[ii+1] = [jj for jj in indices if jj not in indices_all]
        indices_all += indices_dict[ii+1]
    if return_list is True:
        return indices_all
    else:
        return indices_dict

# -------------------------------------------------------------------------------------
# GET REDUCED GRAPH ATOMS
# -------------------------------------------------------------------------------------

def get_reduced_graph_atoms(
    atoms: Atoms,
    indices_ads: list = None,
    method: str = "ase",
    bond_cutoff: int = 2,
) -> Atoms:
    """Get the reduced graph atoms."""
    if indices_ads is None:
        indices_ads = atoms.info["indices_ads"]
    atoms_enlarged = enlarge_surface(atoms=atoms)
    connectivity = get_connectivity(
        atoms=atoms_enlarged,
        method=method,
    )
    indices_list = get_indices_from_bond_cutoff(
        atoms=atoms_enlarged,
        connectivity=connectivity,
        indices_ads=indices_ads,
        bond_cutoff=bond_cutoff,
        return_list=True,
    )
    indices_list = [
        ii for ii in indices_list
        if atoms_enlarged.info["indices_original"][ii] not in indices_ads
        or ii in indices_ads
    ]
    atoms_reduced = atoms_enlarged[indices_list]
    atoms_reduced.info["indices_original"] = (
        list(np.array(atoms_enlarged.info["indices_original"])[indices_list])
    )
    atoms_reduced.info["indices_ads"] = [
        ii for ii, index in enumerate(indices_ads) if index in indices_ads
    ]
    atoms_reduced.info["connectivity"] = get_connectivity(
        atoms=atoms_reduced,
        method=method,
    )
    if "features" in atoms.info.keys():
        atoms_reduced.info["features"] = (
            atoms.info["features"][atoms_reduced.info["indices_original"], :]
        )
    return atoms_reduced

# -------------------------------------------------------------------------------------
# MODIFY NAME
# -------------------------------------------------------------------------------------

def modify_name(
    name: str,
    replace_dict: dict = {},
) -> str:
    """Modify the species name."""
    # Add subscripts to numbers in chemical formulas.
    name_new = ""
    for ii, char in enumerate(name):
        if char.isdecimal() and ii > 0 and name[ii-1].isalpha():
            name_new += f"$_{char}$"
        else:
            name_new += char
    # Replace characters.
    for key in replace_dict:
        name_new = name_new.replace(key, replace_dict[key])
    return name_new

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------