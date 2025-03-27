# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import yaml
import numpy as np

# -------------------------------------------------------------------------------------
# GET CONNECTIVITY ASE
# -------------------------------------------------------------------------------------

def customize_yaml(float_format="{:10.8E}"):
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
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------