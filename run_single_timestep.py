import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import linprog
import cvxpy as cp

from pandapower.pypower.makeYbus import makeYbus
from tqdm import tqdm
from pyomo.opt import SolverFactory
import os

import plot_and_estimate_correlations as corr
import config
import optimal_power_flow as opf
import nodal_flexiblity_approximation as nfa



def save_results(results: dict):
    """
    Saves the optimization results 

    Parameters
    ----------
    results : dict returned by solve_OPF()
    """
    output_dir = f"01-RESULTS/single_ts_{config.scenario}_{config.single_timestep.strftime('%m-%d_%H-%M')}"
    os.makedirs(output_dir, exist_ok=True)

    # Save PCC results
    pcc_df = pd.DataFrame({
        'time': list(results['P_pcc'].keys()),
        'P_pcc': list(results['P_pcc'].values()),
        'Q_pcc': list(results['Q_pcc'].values()),
    })
    pcc_df.to_csv(os.path.join(output_dir, f"pcc_results.csv"), index=False)

    # Save flexibility results
    flex_data = []
    for (t, bus), P_flex in results['P_flex'].items():
        Q_flex = results['Q_flex'][(t, bus)]
        flex_data.append({'time': t, 'bus': bus, 'P_flex': P_flex, 'Q_flex': Q_flex})
    flex_df = pd.DataFrame(flex_data)
    flex_df.to_csv(os.path.join(output_dir, f"flex_results.csv"), index=False)


def main():
    """
    Main function to run the optimization routine for a single scenario and time step.
    """
    print("""
          
    ###########################################################
                        Starting program...
    ###########################################################
    """)
    print(f"Running scenario: {config.scenario}")
    print("Single timestep optimization")
    print(f"Time step: {config.single_timestep}")


    print("Mapping load time series and estimating correlations to compute aggregation data")
    load_df = nfa.map_load_time_series()
    correlation_df = corr.estimate_correlations(load_df)
    if config.scenario == "with_battery":
        correlation_df = corr.add_battery_capacity(correlation_df)
    
    ## load temperature of timestep
    temp_df = pd.read_csv("00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv", parse_dates=["date"], index_col="date")
    temperature = temp_df.loc[config.single_timestep, "temperature_2m"]

    hp_flex = nfa.compute_hp_flexibility(correlation_df, temperature)
    pv_df = pd.read_csv("00-INPUT-DATA/PV-DATA/PV_timeseries.csv", parse_dates=["time"], index_col="time")
    if pv_df.index.tz is None:
        pv_df.index = pv_df.index.tz_localize('UTC')
    pv_cf = pv_df.loc[config.single_timestep, "electricity"]

    device_flex = pd.DataFrame(columns=["P_PV_max", "P_hp_max", "P_hp_min","P_chg_max","P_dis_max"], index=correlation_df.index)
    device_flex["P_PV_max"] = pv_cf * correlation_df["cap_pv_mw"]
    device_flex["P_hp_max"] = hp_flex["P_hp_max"]
    device_flex["P_hp_min"] = hp_flex["P_hp_min"]
    if config.scenario == "with_battery":
        device_flex["P_chg_max"] = correlation_df["cap_battery_mw"]
        device_flex["P_dis_max"] = correlation_df["cap_battery_mw"]
    else:
        device_flex.drop(columns=["P_chg_max","P_dis_max"], inplace=True)

    ## Compute nodal approximations for each node and store in DataFrame
    aggregation_df = pd.DataFrame(columns=['H', 'h', 'd_soc'], index=device_flex.index)
    for node in tqdm(device_flex.index, desc="Computing nodal approximations"):
        P_pv_max = device_flex.loc[node, "P_PV_max"]
        P_hp_max = device_flex.loc[node, "P_hp_max"]
        P_hp_min = device_flex.loc[node, "P_hp_min"]
        if config.scenario == "with_battery":
            P_chg_max = device_flex.loc[node, "P_chg_max"]
            P_dis_max = device_flex.loc[node, "P_dis_max"]
            H, h, d_soc = nfa.compute_nodal_approx_for_with_battery(
                P_pv_max, P_hp_max, P_hp_min, P_dis_max, P_chg_max
            )
        else:
            H, h = nfa.compute_nodal_approx_for_linear_constraints(
                P_pv_max, P_hp_max, P_hp_min
            )
            d_soc = 0.0

        aggregation_df.loc[node] = {
            'H': H,
            'h': h,
            'd_soc': d_soc,
        }

    ## Load network and set up OPF model
    print("Loading network data and setting up OPF")
    net_data = opf.load_network_and_extract()
    base_model = opf.setup_base_OPF(load_df,net_data, time_steps=[config.single_timestep])

    ## Add objective function
    print(f"Adding objective function with alpha={config.alpha}, beta={config.beta}")
    model_with_obj = opf.add_objective_function(base_model, config.alpha, config.beta)

    ## changing index
    bus_lookup = net_data["net"]._pd2ppc_lookups['bus']
    aggregation_df.index = [int(bus_lookup[i]) for i in aggregation_df.index]

    ## Add flexibility constraints
    print("Adding aggregated flexibility constraints to OPF model")
    full_model = opf.add_aggregated_flexibility_constraints(model_with_obj, aggregation_df)

    ## Solve OPF
    print("Solving OPF model with Gurobi")
    results = opf.solve_OPF(full_model, config.alpha, config.beta)
    #print(f"Solver status: {results}")

    ## Saving results
    print("Saving results")
    save_results(results)


if __name__ == "__main__":
    main()
