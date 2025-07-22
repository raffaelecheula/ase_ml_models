# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np
from ase import Atoms

# -------------------------------------------------------------------------------------
# CUSTOMIZE YAML
# -------------------------------------------------------------------------------------

def customize_yaml(
    float_format: str = "{:10.8E}",
):
    """
    Customize YAML serialization for specific data types.
    """
    # Custom YAML representer for floats.
    def float_representer(dumper, value):
        return dumper.represent_scalar(
            "tag:yaml.org,2002:float", float_format.format(value)
        )
    yaml.add_representer(float, float_representer)
    # Custom YAML representer for dictionaries.
    def dict_representer(dumper, data):
        return yaml.representer.SafeRepresenter.represent_dict(dumper, data.items())
    yaml.add_representer(dict, dict_representer)

# -------------------------------------------------------------------------------------
# CONVERT NUMPY TO PYTHON
# -------------------------------------------------------------------------------------

def convert_numpy_to_python(obj):
    """
    Convert numpy types to native Python types for YAML serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, np.ndarray)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# -------------------------------------------------------------------------------------
# WRITE TO YAML
# -------------------------------------------------------------------------------------

def write_to_yaml(
    filename,
    data,
    mode: str = "w",
    default_flow_style: bool = None,
    width: int = 1000,
    sort_keys: bool = False,
    float_format: str = "{:10.8E}",
):
    """ 
    Write data to a YAML file.
    """
    customize_yaml(float_format=float_format)
    with open(filename, mode=mode) as fileobj:
        yaml.dump(
            data=convert_numpy_to_python(data),
            stream=fileobj,
            default_flow_style=default_flow_style,
            width=width,
            sort_keys=sort_keys,
        )

# -------------------------------------------------------------------------------------
# WRITE ATOMS TO YAML
# -------------------------------------------------------------------------------------

def write_atoms_to_yaml(
    atoms_list: list,
    filename: str = "atoms.yaml",
    mode: str = "w",
    default_flow_style: bool = None,
    width: int = 1000,
    sort_keys: bool = False,
    float_format: str = "{:10.8E}",
):
    """
    Write a list of ASE Atoms objects to a YAML file.
    """
    # Prepare list of dictionaries with atoms data.
    data = []
    for atoms in atoms_list:
        data.append({
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions(),
            "cell": np.array(atoms.get_cell()),
            "pbc": atoms.get_pbc(),
            "info": atoms.info,
        })
    # Write atoms data to yaml file.
    write_to_yaml(
        filename=filename,
        data=data,
        mode=mode,
        default_flow_style=default_flow_style,
        width=width,
        sort_keys=sort_keys,
        float_format=float_format,
    )

# -------------------------------------------------------------------------------------
# READ ATOMS FROM YAML
# -------------------------------------------------------------------------------------

def read_atoms_from_yaml(
    filename: str = "atoms.yaml",
):
    """
    Read a list of ASE Atoms objects from a YAML file.
    """
    # Read atoms data from yaml file.
    return [
        Atoms(**atoms_data) for atoms_data in yaml.safe_load(open(filename, "r"))
    ]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------