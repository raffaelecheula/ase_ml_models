# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import yaml
import numpy as np
from ase.io import read
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase_cantera_microkinetics import units
from ase_cantera_microkinetics.reaction_mechanism import NameAnalyzer
from ase_cantera_microkinetics.NASA_fitting import (
    read_vib_energies,
    ase_thermo_to_NASA_coeffs,
)

# -------------------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------------------

# Dictionary of functions to calculate the reference energies of atomic
# species from the reference gas phase molecules.
energy_ref_funs = {
    'H' : lambda energy: (energy['H2'])/2,
    'O' : lambda energy: (energy['H2O'])-2*energy['H'],
    'C' : lambda energy: (energy['CO'])-energy['O'],
    'N' : lambda energy: energy['N2']/2.,
}

# Dictionary of folders containing the molecules.
molecules_dict = {
    'CO': 'CO',
    'H2': 'H2',
    'H2O': 'H2O',
}

# Dictionary of folders containing the adsorbates and their names.
adsorbates_dict = {
    "(Rh)": "reference",
    "CO2(Rh,Rh)": 'Rh-111_CO2[ss]',
    "CO(Rh)": 'Rh-111_CO[s]',
    "O(Rh)": 'Rh-111_O[s]',
    "COOH(Rh,Rh)": 'Rh-111_cCOOH[ss]',
    "HCOO(Rh,Rh)": 'Rh-111_HCOO[ss]',
    "H(Rh)": 'Rh-111_H[s]',
    "OH(Rh)": 'Rh-111_OH[s]',
    "H2O(Rh)": 'Rh-111_H2O[s]',
    "HCO(Rh,Rh)": 'Rh-111_HCO[ss]',
    "COH(Rh)": 'Rh-111_COH[s]',
}

# Dictionary of folders containing the transition states and their names.
reactions_dict = {
    "CO2 + 2 (Rh) <=> CO2(Rh,Rh)": "sticking",
    "CO + (Rh) <=> CO(Rh)": "sticking",
    "H2O + (Rh) <=> H2O(Rh)": "sticking",
    "H2 + 2 (Rh) <=> H(Rh) + H(Rh)": "Rh-111_H2[s]=H[s]+H_1[s]",
    "CO2(Rh,Rh) <=> CO(Rh) + O(Rh)": "Rh-111_CO2[ss]=CO[s]+O[s]",
    "COOH(Rh,Rh) <=> CO(Rh) + OH(Rh)": "Rh-111_cCOOH[s]=CO[s]+OH[s]",
    "COOH(Rh,Rh) + (Rh) <=> CO2(Rh,Rh) + H(Rh)": "Rh-111_tCOOH[ss]=CO2[ss]+H[s]",
    "HCOO(Rh,Rh) + (Rh) <=> CO2(Rh,Rh) + H(Rh)": "Rh-111_HCOO[ss]=CO2[ss]+H[s]",
    "HCOO(Rh,Rh) + (Rh) <=> HCO(Rh,Rh) + O(Rh)": "Rh-111_HCOO[ss]=HCO[ss]+O[s]",
    "H2O(Rh) + (Rh) <=> OH(Rh) + H(Rh)": "Rh-111_H2O[s]=OH[s]+H[s]",
    "OH(Rh) + (Rh) <=> O(Rh) + H(Rh)": "Rh-111_OH[s]=O[s]+H[s]",
    "COOH(Rh,Rh) <=> COH(Rh) + O(Rh)": "Rh-111_cCOOH[ss]=COH[s]+O[s]",
    "COH(Rh) + (Rh) <=> CO(Rh) + H(Rh)": "Rh-111_COH[s]=CO[s]+H[s]",
    "HCO(Rh,Rh) <=> CO(Rh) + H(Rh)": "Rh-111_HCO[s]=CO[s]+H[s]",
    #"HCOOH + (Rh) <=> HCOOH(Rh)": "sticking",
    #"HCOOH <=> HCOO(Rh,Rh) + H(Rh)": "Rh-111_HCOOH[s]=HCOO[ss]+H[s]",
    #"HCOOH <=> COOH(Rh,Rh) + H(Rh)": "Rh-111_HCOOH[s]=cCOOH[ss]+H[s]",
}

# Function to get the energy of atoms structures.
def get_energy_atoms(atoms):
    energy = atoms.get_potential_energy()
    return energy

# -------------------------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------------------------

# Get gas species and change reference energies.
name_analyzer = NameAnalyzer()
with open('gas_NASA_coeffs.yaml', 'r') as fileobj:
    species_gas = yaml.safe_load(fileobj)["species"]
energy = {}
for species in species_gas:
    energy[species["name"]] = species["thermo"]["data"][0][5]
for name in energy_ref_funs:
    energy[name] = energy_ref_funs[name](energy=energy)
for species in species_gas:
    composition = name_analyzer.get_composition(name=species["name"])
    for elem in composition:
        species["thermo"]["data"][0][5] -= energy[elem]*composition[elem]
        species["thermo"]["data"][1][5] -= energy[elem]*composition[elem]
    # Format data for yaml.
    del species["transport"]
    species["thermo"]["temperature-ranges"] = [
        int(ii) for ii in species["thermo"]["temperature-ranges"]
    ]

# Get the ZPE of gas phase molecules.
energy_ZP_dict = {"Rh": 0.0}
for name in molecules_dict:
    # Read DFT energy of molecules.
    atoms = read(os.path.join('gas', molecules_dict[name], 'pw.pwo'))
    energy = get_energy_atoms(atoms)
    # Read vibrations of molecules and calculate ZPE.
    filename = os.path.join('gas', molecules_dict[name], 'vib.log')
    vib_energies = read_vib_energies(filename=filename, imaginary=False)
    # We use monatomic geometry for all molecules because we need to calculate
    # only the ZPE. The Gibbs is calculated from experimental NASA coefficients.
    thermo = IdealGasThermo(vib_energies=vib_energies, geometry='monatomic')
    energy_ZP_dict[name] = thermo.get_ZPE_correction()

# Get reference energy of atomic species.
del energy_ref_funs["N"]
for name in energy_ref_funs:
    energy_ZP_dict[name] = energy_ref_funs[name](energy=energy_ZP_dict)

# Get vibrational contributions to the Gibbs free energies and build NASA
# coefficients for each species from interpolation.
species_all = {
    "species-gas": species_gas,
    "species-adsorbates": [],
    "species-reactions": [],
}
for species_type, species_dict in zip(
    ["adsorbates", "reactions"], [adsorbates_dict, reactions_dict]
):
    for name in species_dict:
        # Get composition and size of the adsorbate.
        composition, size = name_analyzer.get_composition_and_size(name=name)
        if species_dict[name] in ("reference", "sticking"):
            thermo_data = [[0. for ii in range(7)]]
        else:
            # Read vibrational frequencies.
            filename = os.path.join(species_type, species_dict[name], 'vib.log')
            vib_energies = read_vib_energies(filename=filename, imaginary=False)
            # Calculate reference ZPE (formation energies include ZPE).
            energy_ZP_ref = 0.
            for element in composition:
                energy_ZP_ref -= energy_ZP_dict[element]
            # Get NASA coefficients from the DFT vibrations (ZPE is included).
            thermo = HarmonicThermo(
                vib_energies=vib_energies,
                potentialenergy=energy_ZP_ref,
            )
            print(f'{name:50s} {thermo.get_ZPE_correction():+12.6f} [eV/molecule]')
            thermo_data = ase_thermo_to_NASA_coeffs(
                thermo=thermo,
                n_points=1000,
                t_low=200,
                t_max=1000,
                coeffs_ref=None,
                subtract_ZPE=False,
            )
            thermo_data = [[float(ii) for ii in thermo_data]]
        thermo_dict = {
            "model": "NASA7",
            "temperature-ranges": [200, 2000],
            "data": thermo_data,
        }
        if species_dict[name] == "sticking":
            thermo_dict["sticking"] = True
        species = {
            "name": name,
            "composition": composition,
            "size": size,
            "thermo": thermo_dict,
        }
        species_all[f"species-{species_type}"].append(species)

# Custom YAML representer for floats.
def float_representer(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:float', f"{value:+10.8E}")
yaml.add_representer(float, float_representer)

# Custom YAML representer for dictionaries.
def dict_representer(dumper, data):
    return yaml.representer.SafeRepresenter.represent_dict(dumper, data.items())
yaml.add_representer(dict, dict_representer)

# Write the reaction mechanism.
with open('mechanism.yaml', 'w') as fileobj:
    yaml.dump(
        data=species_all,
        stream=fileobj,
        default_flow_style=None,
        width=150,
        sort_keys=False,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------