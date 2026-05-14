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
    output_dir = f"01-RESULTS/multi_ts_{config.scenario}_{config.multi_timestep_interval[0].strftime('%m-%d_%H-%M')}--{config.multi_timestep_interval[1].strftime('%m-%d_%H-%M')}"
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
    print("Multi timestep optimization")
    print(f"Time interval: {config.multi_timestep_interval[0]} to {config.multi_timestep_interval[1]}")
    time_steps = pd.date_range(
        start=config.multi_timestep_interval[0],
        end=config.multi_timestep_interval[1],
        freq="h"
    )
    print("Mapping load time series and estimating correlations to compute aggregation data")
    net_data = opf.load_network_and_extract()
    load_df = nfa.map_load_time_series()
    correlation_df = corr.estimate_correlations(load_df)
    if config.scenario == "with_battery":
        correlation_df = corr.add_battery_capacity(correlation_df)
    
    temp_df = pd.read_csv("00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv",parse_dates=['date'],index_col='date')

    hp_flex_df = pd.DataFrame(index=correlation_df.index, columns=time_steps)
    for t in time_steps:
        temp_t = float(temp_df.loc[t, 'temperature_2m'])
        hp_flex_t = nfa.compute_hp_flexibility(correlation_df,temp_t)
        hp_flex_df[t] = hp_flex_t['P_hp_max'] / net_data["S_base"]

    pv_df = pd.read_csv("00-INPUT-DATA/PV-DATA/PV_timeseries.csv", parse_dates=["time"], index_col="time")
    if pv_df.index.tz is None:
        pv_df.index = pv_df.index.tz_localize('UTC')
    pv_cf = pv_df.loc[time_steps, "electricity"]
  
    aggregation_per_timestep = {}  

    for t in time_steps:
        temp_t = float(temp_df.loc[t, 'temperature_2m'])

        device_flex_t = pd.DataFrame(
            columns=["P_PV_max", "P_hp_max", "P_hp_min", "P_chg_max", "P_dis_max"],
            index=correlation_df.index
        )
        device_flex_t["P_PV_max"] = pv_cf[t] * correlation_df["cap_pv_mw"] / net_data["S_base"]
        device_flex_t["P_hp_max"] = hp_flex_df[t]
        device_flex_t["P_hp_min"] = 0.0

        if config.scenario == "with_battery":
            device_flex_t["P_chg_max"] = correlation_df["cap_battery_mw"] / net_data["S_base"]
            device_flex_t["P_dis_max"] = correlation_df["cap_battery_mw"] / net_data["S_base"]
        else:
            device_flex_t.drop(columns=["P_chg_max", "P_dis_max"], inplace=True)

        aggregation_df_t = pd.DataFrame(
            columns=['H', 'h', 'd_soc'],
            index=device_flex_t.index
        )

        for node in tqdm(device_flex_t.index, desc=f"Computing nodal approximations for {t}"):
            P_pv_max = device_flex_t.loc[node, "P_PV_max"]
            P_hp_max = device_flex_t.loc[node, "P_hp_max"]
            P_hp_min = device_flex_t.loc[node, "P_hp_min"]

            if config.scenario == "with_battery":
                P_chg_max = device_flex_t.loc[node, "P_chg_max"]
                P_dis_max = device_flex_t.loc[node, "P_dis_max"]
                H, h, d_soc = nfa.compute_nodal_approx_for_with_battery(
                    P_pv_max, P_hp_max, P_hp_min, P_dis_max, P_chg_max
                )
            else:
                H, h = nfa.compute_nodal_approx_for_linear_constraints(
                    P_pv_max, P_hp_max, P_hp_min
                )
                d_soc = 0.0

            aggregation_df_t.loc[node] = {'H': H, 'h': h, 'd_soc': d_soc}

        aggregation_per_timestep[t] = aggregation_df_t

    ## Load network and set up OPF model
    print("Loading network data and setting up OPF")
    bus_lookup = net_data["net"]._pd2ppc_lookups['bus']
    aggregation_per_timestep = {
        t: df.set_index(df.index.map(lambda i: int(bus_lookup[i])))
        for t, df in aggregation_per_timestep.items()
    }
    hp_flex_df.index = [int(bus_lookup[i]) for i in hp_flex_df.index]

    full_model = opf.setup_multi_timestep_OPF(load_df, net_data, time_steps, aggregation_per_timestep,hp_flex_df,temp_df)


    ## Solve OPF
    print("Solving OPF model with Gurobi")
    results = opf.solve_OPF(full_model, config.alpha, config.beta)

    ## Saving results
    print("Saving results")
    save_results(results)


if __name__ == "__main__":
    main()
