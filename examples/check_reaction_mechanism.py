# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase.db import connect
from ase_cantera_microkinetics import units

from ase_ml_models.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Ase database names.
    db_ads_name = "atoms_adsorbates_DFT.db"
    db_ts_name = "atoms_reactions_DFT.db"

    # Materials.
    material_list = [
        'Rh',
        'Pd',
        'Co',
        'Ni',
        'Cu',
        'Au',
        'Rh+Pt1',
        'Pd+Rh1',
        'Co+Pt1',
        'Ni+Ga1',
        'Cu+Zn1',
        'Cu+Pt1',
        'Cu+Rh1',
        'Cu+Ni1',
        'Au+Ag1',
        'Au+Pt1',
        'Au+Rh1',
        'Au+Ni1',
    ]
    
    miller_index_list = [
        '100',
        '111',
    ]
    
    # Reaction mechanism parameters.
    yaml_file = 'mechanism.yaml'
    
    # Read the reaction mechanism.
    with open(yaml_file, 'r') as fileobj:
        mechanism = yaml.safe_load(fileobj)

    adsorbates_list = [
        species["name"] for species in mechanism["species-adsorbates"]
        if species["name"] != "(Rh)"
    ]
    reactions_list = [
        species["name"] for species in mechanism["species-reactions"]
        if species["name"] != "(Rh)"
        if not species["thermo"].get("sticking", False)
    ]

    # Read Ase databases.
    db_ads = connect(db_ads_name)
    db_ts = connect(db_ts_name)
    atoms_ads_list = get_atoms_list_from_db(db_ase=db_ads)
    atoms_ts_list = get_atoms_list_from_db(db_ase=db_ts)
    
    e_form_dict = {}
    for material in material_list:
        for miller_index in miller_index_list:
            for species_cantera in adsorbates_list:
                atoms_list = [
                    atoms for atoms in atoms_ads_list
                    if atoms.info["material"] == material
                    if atoms.info["miller_index"] == miller_index
                    if atoms.info["species_cantera"] == species_cantera
                ]
                e_form_list = [atoms.info["E_form"] for atoms in atoms_list]
                e_form = np.min(e_form_list) if len(e_form_list) > 0 else None
                if e_form is None:
                    print(material, miller_index, species_cantera, e_form)
            for species_cantera in reactions_list:
                atoms_list = [
                    atoms for atoms in atoms_ts_list
                    if atoms.info["material"] == material
                    if atoms.info["miller_index"] == miller_index
                    if atoms.info["species_cantera"] == species_cantera
                ]
                e_form_list = [atoms.info["E_form"] for atoms in atoms_list]
                e_form = np.min(e_form_list) if len(e_form_list) > 0 else None
                if e_form is None:
                    print(material, miller_index, species_cantera, e_form)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------