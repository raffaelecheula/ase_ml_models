# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms
from ase.db.core import Database

# -------------------------------------------------------------------------------------
# WRITE ATOMS LIST TO DB
# -------------------------------------------------------------------------------------

def write_atoms_list_to_db(
    atoms_list: list,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    fill_stress: bool = False,
    fill_magmom: bool = False,
):
    """Write list of ase Atoms to ase database."""
    for atoms in atoms_list:
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            keys_store=keys_store,
            keys_match=keys_match,
            fill_stress=fill_stress,
            fill_magmom=fill_magmom,
        )

# -------------------------------------------------------------------------------------
# WRITE ATOMS TO DB
# -------------------------------------------------------------------------------------

def write_atoms_to_db(
    atoms: Atoms,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    fill_stress: bool = False,
    fill_magmom: bool = False,
):
    """Write ase Atoms to ase database."""
    # Fill with zeros stress and magmoms.
    if fill_stress and "stress" not in atoms.calc.results:
        atoms.calc.results["stress"] = np.zeros(6)
    if fill_magmom and "magmoms" not in atoms.calc.results:
        atoms.calc.results["magmoms"] = np.zeros(len(atoms))
    # Get dictionary to store atoms.info into the columns of the db.
    if not keys_store and "keys_store" in atoms.info:
        keys_store = atoms.info["keys_store"]
    kwargs_store = {key: atoms.info[key] for key in keys_store}
    # Get dictionary to check if structure is already in db.
    if keys_match is not None:
        kwargs_match = {key: atoms.info[key] for key in keys_match}
    # Write structure to db.
    if keys_match is None or db_ase.count(**kwargs_match) == 0:
        db_ase.write(atoms=atoms, data=atoms.info, **kwargs_store)
    elif db_ase.count(**kwargs_match) == 1:
        row_id = db_ase.get(**kwargs_match).id
        db_ase.update(id=row_id, atoms=atoms, data=atoms.info, **kwargs_store)

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db(
    db_ase: Database,
    selection: str = "",
    **kwargs,
) -> list:
    """Get list of ase Atoms from ase database."""
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection, **kwargs)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms.info.update(atoms_row.key_value_pairs)
        atoms_list.append(atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# GET ATOMS FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_from_db(
    db_ase: Database,
    selection: str = "",
    none_ok: bool = False,
    **kwargs,
) -> Atoms:
    """Get ase Atoms from ase database."""
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection, **kwargs)
    if none_ok is True and len(atoms_list) < 1:
        return None
    elif len(atoms_list) < 1:
        raise RuntimeError("No atoms structure found in database.")
    elif len(atoms_list) > 1:
        raise RuntimeError("More than one atoms structure found in database.")
    return atoms_list[0]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------