# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import timeit
import cantera as ct
import numpy as np
import yaml
from ase.db import connect
from ase_cantera_microkinetics import units
from ase_cantera_microkinetics.cantera_utils import (
    get_Y_dict,
    get_X_dict,
    reactions_from_cat_ts,
    advance_sim_to_steady_state,
    degree_rate_control,
    molar_balance_of_element,
)
from ase_cantera_microkinetics.reaction_mechanism_from_yaml import (
    get_mechanism_from_yaml,
    get_e_form_from_ase_atoms,
)

from ase_ml_models.databases import get_atoms_list_from_db

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    # Parameters.
    reaction = "RWGS" # WGS | RWGS
    task = "database" # database | extrapol
    ensemble = False
    calculate_DRC = False
    model_adsorbates = "DFT" # DFT | TSR | SKLearn | WWLGPR
    model_reactions = "DFT" # DFT | BEP | SKLearn | WWLGPR
    
    # Get materials and Miller indices.
    with open("materials.yaml", 'r') as fileobj:
        data = yaml.safe_load(fileobj)
    miller_index_list = data["miller_indices"]
    material_list = data[f"materials_{task}"]
    
    # Main loop.
    e_index_list = [0, 1, 2, 3, 4] if ensemble is True else [None]
    for material in material_list:
        for miller_index in miller_index_list:
            for e_index in e_index_list:
                print(f'\nSurface = {material}({miller_index})')
                kinetics_integration(
                    reaction=reaction,
                    model_adsorbates=model_adsorbates,
                    model_reactions=model_reactions,
                    material=material,
                    miller_index=miller_index,
                    e_index=e_index,
                    task=task,
                    calculate_DRC=calculate_DRC,
                )
    stop = timeit.default_timer()
    print(f'\nExecution time = {stop-start:6.3} s\n')

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def kinetics_integration(
    reaction="WGS",
    model_adsorbates="DFT",
    model_reactions="DFT",
    material='Rh',
    miller_index='100',
    e_index=None,
    task="database",
    calculate_DRC=False,
):

    # Analysis parameters.
    reaction_indices = {"CO-O path": 4, "COO-H path": 6, "H-COO path": 7}

    # Set temperature and pressure of the simulation.
    temperature_celsius = 500. # [C]
    temperature = units.Celsius_to_Kelvin(temperature_celsius) # [K]
    pressure = 1 * units.atm # [Pa]
    
    # Set molar fractions of gas species.
    if reaction == "WGS":
        gas_molfracs_inlet = {
            'CO2': 0.02,
            'H2': 0.02,
            'H2O': 0.28,
            'CO': 0.28,
            'N2': 0.40,
        }
    elif reaction == "RWGS":
        gas_molfracs_inlet = {
            'CO2': 0.28,
            'H2': 0.28,
            'H2O': 0.02,
            'CO': 0.02,
            'N2': 0.40,
        }
    
    # Set the initial coverages of the free catalytic sites.
    cat_coverages_inlet = {'(Rh)': 1.0}

    # Set number of cstr for the discretization of the pfr.
    n_cstr = 1

    # Set the catalyst and reactor parameters.
    alpha_cat = 20000.0 # [m^2/m^3]
    cat_site_density = 1e-8 # [kmol/m^2]
    vol_flow_rate = 1e-8 # [m^3/s]

    # Set the cross section and the total volume of the reactor.
    reactor_length = 1.00 * units.centimeter # [m]
    diameter = 1.00 * units.millimeter # [m]
    cross_section = np.pi*(diameter**2)/4. # [m^2]
    reactor_volume = cross_section*reactor_length # [m^3]
    cat_sites_tot = cat_site_density*alpha_cat*reactor_volume # [kmol]

    # Reaction mechanism parameters.
    yaml_file = 'mechanism.yaml'
    db_ads_name = f"databases/atoms_adsorbates_{model_adsorbates}_{task}.db"
    db_ts_name = f"databases/atoms_reactions_{model_reactions}_{task}.db"
    
    # Formations energies.
    db_ads = connect(db_ads_name)
    db_ts = connect(db_ts_name)
    atoms_ads_list = get_atoms_list_from_db(db_ase=db_ads)
    atoms_ads_list = [
        atoms for atoms in atoms_ads_list
        if atoms.info["material"] == material
        if atoms.info["miller_index"] == miller_index
    ]
    atoms_ts_list = get_atoms_list_from_db(db_ase=db_ts)
    atoms_ts_list = [
        atoms for atoms in atoms_ts_list
        if atoms.info["material"] == material
        if atoms.info["miller_index"] == miller_index
    ]
    e_form_dict = get_e_form_from_ase_atoms(
        yaml_file=yaml_file,
        atoms_ads_list=atoms_ads_list,
        atoms_ts_list=atoms_ts_list,
        e_form_key="E_form" if e_index is None else "E_form_list",
        e_index=e_index,
        free_site="(Rh)",
    )

    # Reaction mechanism.
    gas, cat, cat_ts = get_mechanism_from_yaml(
        yaml_file=yaml_file,
        e_form_dict=e_form_dict,
        site_density=cat_site_density,
        temperature=temperature,
        units_energy=units.eV/units.molecule,
    )
    gas.TPX = temperature, pressure, gas_molfracs_inlet
    cat.TP = temperature, pressure
    cat.coverages = cat_coverages_inlet
    cat_ts.TP = temperature, pressure
    
    # Calculate gas space velocity.
    molar_flow_rate = vol_flow_rate*gas.density/gas.mean_molecular_weight # [kmol/s]
    cat_moles = cat_site_density*alpha_cat*reactor_volume # [kmol]
    gas_space_velocity = molar_flow_rate/cat_moles # [1/s]

    # Update the reactions from the transition states.
    reactions_from_cat_ts(gas=gas, cat=cat, cat_ts=cat_ts)

    # Create an ideal reactor.
    cstr_length = reactor_length/n_cstr # [m]
    cstr_volume = cross_section*cstr_length # [m^3]
    cstr_cat_area = alpha_cat*cstr_volume # [m^2]
    mass_flow_rate = vol_flow_rate*gas.density # [kg/s]
    cstr = ct.IdealGasReactor(gas, energy='off', name='cstr')
    cstr.volume = cstr_volume
    surf = ct.ReactorSurface(cat, cstr, A=cstr_cat_area)
    
    # Add mass flow and pressure controllers.
    upstream = ct.Reservoir(gas, name='upstream')
    master = ct.MassFlowController(
        upstream=upstream,
        downstream=cstr,
        mdot=mass_flow_rate,
    )
    downstream = ct.Reservoir(gas, name='downstream')
    pcontrol = ct.PressureController(
        upstream=cstr,
        downstream=downstream,
        master=master,
        K=1e-6,
    )

    # Define the parameters of the simulation.
    sim = ct.ReactorNet([cstr])
    sim.rtol = 1.0e-15
    sim.atol = 1.0e-18
    sim.max_steps = 1e9
    sim.max_err_test_fails = 1e9
    sim.max_time_step = 1.0

    # Integrate the plug-flow reactor along the z (reactor length) axis.
    print('\n- Reactor integration.')
    gas_array = ct.SolutionArray(gas, extra=['z_reactor'])
    cat_array = ct.SolutionArray(cat, extra=['z_reactor'])
    # Print the header of the table.
    string = 'distance[m]'.rjust(14)
    for spec in gas.species_names:
        string += ('x_'+spec+'[-]').rjust(12)
    print(string)
    # Integrate the reactor.
    for ii in range(n_cstr+1):
        z_reactor = ii*cstr_length
        gas_array.append(state=gas.state, z_reactor=z_reactor)
        cat_array.append(state=cat.state, z_reactor=z_reactor)
        # Print the state of the reactor.
        string = f'  {z_reactor:12f}'
        for spec in gas.species_names:
            string += f'  {gas[spec].X[0]:10f}'
        print(string)
        # Intergate the microkinetic model.
        if ii < n_cstr:
            advance_sim_to_steady_state(sim=sim, n_try_max=1000)
        # Set the gas composition of the inlet of the next cstr equal to the 
        # outlet of the previous cstr.
        gas.TDY = cstr.thermo.TDY
        upstream.syncState()

    # Calculate the conversion and the TOF.
    print('\n- Catalytic activity.')
    gas_reac = "CO" if reaction == "WGS" else "CO2"
    gas_prod = "CO2" if reaction == "WGS" else "CO"
    massfracs_in = get_Y_dict(gas_array[0])
    massfracs_out = get_Y_dict(gas_array[-1])
    conversion_reac = (
        (massfracs_in[gas_reac]-massfracs_out[gas_reac])/massfracs_in[gas_reac]
    )
    print(f'Conversion of {gas_reac} = {conversion_reac*100:+12.6f} [%]')
    mol_weight_prod = 44 if reaction == "WGS" else 28 # [kg/kmol]
    mass_flow_prod = (
        (massfracs_out[gas_prod]-massfracs_in[gas_prod])*mass_flow_rate # [kg/s]
    )
    tof_prod = mass_flow_prod/mol_weight_prod/cat_sites_tot # [1/s]
    print(f'Turnover frequency of {gas_prod} = {tof_prod:+12.6e} [1/s]')
    
    # Get the coverages.
    coverages = {spec: cat[spec].coverages[0] for spec in cat.species_names}
    print("\n- Coverages.")
    for spec, coverage in coverages.items():
        print(f"{spec.ljust(20)} {coverage:7.4e}")
    
    # Calculate elements molar balance.
    delta_moles_fracts = molar_balance_of_element(
        gas_in=gas_array[0],
        gas_out=gas_array[-1],
        mass=mass_flow_rate,
        return_fraction=True,
    )
    print("\n- Molar balances of elements.")
    for elem, delta_molfract in delta_moles_fracts.items():
        print(f'Balance of {elem} = {delta_molfract*100:+7.2f} [%]')
    
    # Calculate reaction paths contribution.
    paths_contrib = {}
    for name, ii in reaction_indices.items():
        react = cat.reaction(ii)
        r_net = cat.net_rates_of_progress[ii]*surf.area
        paths_contrib[name] = abs(r_net)
    r_sum = sum(paths_contrib.values())
    paths_contrib = {
        name: float(r_net/r_sum) for name, r_net in paths_contrib.items()
    }
    print("\n- Reaction paths contributions.")
    for name, value in paths_contrib.items():
        print(f'Contribution of {name:10s} = {value*100:+7.2f} [%]')
    
    # Calculate the degree of rate control.
    if calculate_DRC is True:
        DRC_dict = degree_rate_control(
            gas=gas,
            cat=cat,
            sim=sim,
            mdot=mass_flow_rate,
            upstream=upstream,
            gas_spec_target=gas_prod,
            multip_value=1.05,
            return_dict=True,
        )
        # Print DRC results.
        print("\n- Degree of rate control.")
        for ii, react in enumerate(DRC_dict):
            print(f' {ii:3d} {react:80s} {DRC_dict[react]:+7.4f}')
        sum_DRC = np.sum([DRC_dict[react] for react in DRC_dict])
        print(f'Sum DRC = {sum_DRC:+7.4f}')

    # Update names for yaml file.
    coverages_yaml = {}
    for name, coverage in coverages.items():
        name_yaml = name.replace("(Rh)", "*").replace("(Rh,Rh)", "**")
        coverages_yaml[name_yaml] = float(coverage)
    if calculate_DRC is True:
        DRC_yaml = {}
        for name, drc in DRC_dict.items():
            name_yaml = name.replace("(Rh)", "*").replace("(Rh,Rh)", "**")
            name_yaml = name_yaml.replace(" ", "")
            DRC_yaml[name_yaml] = float(drc)

    # Store results in dictionary.
    yaml_results = f"results_{task}_{reaction}.yaml"
    results_all = {}
    if os.path.isfile(yaml_results):
        with open(yaml_results, 'r') as fileobj:
            results_all = yaml.safe_load(fileobj)
    
    # Models names.
    models_name = "+".join([model_adsorbates, model_reactions])
    # Surface name.
    if "+" in material:
        matrix, dopant = material.split("+")
        surface_name = f"{matrix}({miller_index})+{dopant}"
    else:
        surface_name = f"{material}({miller_index})"
    # Store the results in a dictionary.
    if models_name not in results_all:
        results_all[models_name] = {}
    if surface_name not in results_all[models_name]:
        results_all[models_name][surface_name] = {}
    results_all[models_name][surface_name][e_index] = {
        "TOF": float(tof_prod),
        "paths": paths_contrib,
        "coverages": coverages_yaml,
    }
    if calculate_DRC is True:
        results_all[models_name][surface_name][e_index]["DRC"] = DRC_yaml
    
    # Custom YAML representer for floats.
    def float_representer(dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:float', f"{value:+10.8E}")
    yaml.add_representer(float, float_representer)
    # Custom YAML representer for dictionaries.
    def dict_representer(dumper, data):
        return yaml.representer.SafeRepresenter.represent_dict(dumper, data.items())
    yaml.add_representer(dict, dict_representer)
    # Write the reaction mechanism.
    with open(yaml_results, 'w') as fileobj:
        yaml.dump(
            data=results_all,
            stream=fileobj,
            default_flow_style=None,
            width=1000,
            sort_keys=False,
        )

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print(f'\nExecution time = {stop-start:6.3} s\n')

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
