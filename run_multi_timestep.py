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
import nodal_flexibility_approximation as nfa


def generate_ffor_directions(n_directions: int):
    """
    Generates alpha/beta pairs for FFOR construction.

    Since the OPF minimizes alpha*dP + beta*dQ,
    alpha=-cos(theta), beta=-sin(theta) gives the support point
    in direction theta.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)

    directions = []
    for direction_id, theta in enumerate(angles):
        alpha = -float(np.cos(theta))
        beta = -float(np.sin(theta))

        directions.append({
            "direction_id": direction_id,
            "theta_rad": float(theta),
            "theta_deg": float(np.degrees(theta)),
            "alpha": alpha,
            "beta": beta,
        })

    return directions


def save_multi_direction_results(all_results, time_steps, duration_h):
    """
    Saves optimization results for multiple alpha/beta directions.
    """

    output_dir = (
        f"01-RESULTS/multi_ts_{config.scenario}_"
        f"{time_steps[0].strftime('%m-%d_%H-%M')}_"
        f"d{duration_h}h"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary / vertices for later FFOR reconstruction
    vertex_rows = []

    # 2. Full PCC time series for every direction
    pcc_rows = []

    # 3. Full nodal flexibility decomposition for every direction
    flex_rows = []

    for res in all_results:
        direction_id = res["direction_id"]
        theta_rad = res["theta_rad"]
        theta_deg = res["theta_deg"]
        alpha = res["alpha"]
        beta = res["beta"]
        status = res["status"]

        vertex_rows.append({
            "direction_id": direction_id,
            "duration_h": duration_h,
            "theta_rad": theta_rad,
            "theta_deg": theta_deg,
            "alpha": alpha,
            "beta": beta,
            "status": status,
            "solver_status": res.get("solver_status"),
            "solve_attempt": res.get("solve_attempt"),
            "obj_value": res.get("obj_value"),
            "P_flex_pcc": res.get("P_flex_pcc"),
            "Q_flex_pcc": res.get("Q_flex_pcc"),
        })

        if status not in ("optimal", "locallyOptimal"):
            continue

        for t in res["P_pcc"].keys():
            pcc_rows.append({
                "direction_id": direction_id,
                "duration_h": duration_h,
                "theta_rad": theta_rad,
                "theta_deg": theta_deg,
                "alpha": alpha,
                "beta": beta,
                "time": t,
                "P_pcc": res["P_pcc"][t],
                "Q_pcc": res["Q_pcc"][t],
                "P_flex_pcc": res.get("P_flex_pcc"),
                "Q_flex_pcc": res.get("Q_flex_pcc"),
            })

        for (t, bus), P_flex in res["P_flex"].items():
            flex_rows.append({
                "direction_id": direction_id,
                "duration_h": duration_h,
                "theta_rad": theta_rad,
                "theta_deg": theta_deg,
                "alpha": alpha,
                "beta": beta,
                "time": t,
                "bus": bus,
                "P_flex": P_flex,
                "Q_flex": res["Q_flex"][(t, bus)],
                "P_pv_flex": res.get("P_pv_flex", {}).get((t, bus), np.nan),
                "Q_pv_flex": res.get("Q_pv_flex", {}).get((t, bus), np.nan),
                "P_hp_flex": res.get("P_hp_flex", {}).get((t, bus), np.nan),
                "P_bat_flex": res.get("P_bat_flex", {}).get((t, bus), np.nan),
                "Q_bat_flex": res.get("Q_bat_flex", {}).get((t, bus), np.nan),
            })

    vertices_df = pd.DataFrame(vertex_rows)
    pcc_df = pd.DataFrame(pcc_rows)
    flex_df = pd.DataFrame(flex_rows)

    vertices_df.to_csv(os.path.join(output_dir, "ffor_vertices.csv"), index=False)
    pcc_df.to_csv(os.path.join(output_dir, "pcc_results_all_directions.csv"), index=False)
    flex_df.to_csv(os.path.join(output_dir, "flex_results_all_directions.csv"), index=False)

    print(f"Saved results to: {output_dir}")

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

def build_per_timestep_data(time_steps, correlation_df, pv_cf, hp_base_df, net_data):
    """
    For each timestep, builds the device-flex DataFrame and the nodal
    approximation (H, h, d_soc) for every node.

    Returns
    -------
    aggregation_per_timestep : dict[Timestamp, DataFrame] with columns [H, h, d_soc]
    device_flex_per_timestep : dict[Timestamp, DataFrame] with device-flex columns
    """
    aggregation_per_timestep = {}
    device_flex_per_timestep = {}

    for t in time_steps:
        device_flex_t = pd.DataFrame(
            columns=[
                "P_PV_max",
                "P_hp_flex_min",
                "P_hp_flex_max",
                "P_chg_max",
                "P_dis_max",
                "S_bat_max",
            ],
            index=correlation_df.index,
        )

        # PV: positive flexibility via curtailment
        device_flex_t["P_PV_max"] = (
            pv_cf.loc[t] * correlation_df["cap_pv_mw"] / net_data["S_base"]
        )

        # HP baseline (temperature-dependent)
        P_hp_base = hp_base_df[t]
        P_hp_rated = correlation_df["cap_hp_mw"] / net_data["S_base"]

        # Positive flex: HP can ramp down to 0
        device_flex_t["P_hp_flex_max"] = P_hp_base
        # Negative flex: HP can ramp up to rated
        device_flex_t["P_hp_flex_min"] = P_hp_base - P_hp_rated

        if config.scenario == "with_battery":
            P_bat_rated = correlation_df["cap_battery_mw"] / net_data["S_base"]
            device_flex_t["P_chg_max"] = P_bat_rated
            device_flex_t["P_dis_max"] = P_bat_rated
            device_flex_t["S_bat_max"] = P_bat_rated
        else:
            device_flex_t["P_chg_max"] = 0.0
            device_flex_t["P_dis_max"] = 0.0
            device_flex_t["S_bat_max"] = 0.0

        aggregation_df_t = pd.DataFrame(
            columns=["H", "h", "d_soc"], index=device_flex_t.index
        )

        for node in tqdm(device_flex_t.index,
                         desc=f"Nodal approximations for {t}"):
            P_pv_max = float(device_flex_t.loc[node, "P_PV_max"])
            P_hp_flex_min = float(device_flex_t.loc[node, "P_hp_flex_min"])
            P_hp_flex_max = float(device_flex_t.loc[node, "P_hp_flex_max"])
            P_chg_max = float(device_flex_t.loc[node, "P_chg_max"])
            P_dis_max = float(device_flex_t.loc[node, "P_dis_max"])
            S_bat_max = float(device_flex_t.loc[node, "S_bat_max"])

            if config.scenario == "with_battery" and S_bat_max > 1e-9:
                H, h, d_soc = nfa.compute_nodal_approx_for_with_battery(
                    P_pv_max=P_pv_max,
                    P_hp_max=P_hp_flex_max,
                    P_hp_min=P_hp_flex_min,
                    P_dis_max=P_dis_max,
                    P_chg_max=P_chg_max,
                    S_bat_max=S_bat_max,
                )
            else:
                H, h = nfa.compute_nodal_approx_for_linear_constraints(
                    P_pv_max=P_pv_max,
                    P_hp_max=P_hp_flex_max,
                    P_hp_min=P_hp_flex_min,
                )
                d_soc = 0.0

            aggregation_df_t.loc[node] = {"H": H, "h": h, "d_soc": d_soc}

        device_flex_per_timestep[t] = device_flex_t.copy()
        aggregation_per_timestep[t] = aggregation_df_t

    return aggregation_per_timestep, device_flex_per_timestep


def solve_all_directions(full_model, time_steps, duration_h):
    """
    Solves the OPF for every FFOR direction and returns the list of result dicts.
    Each duration gets its own log subdirectory so logs don't get overwritten.
    """
    directions = generate_ffor_directions(config.n_ffor_directions)

    log_dir = (
        f"01-RESULTS/multi_ts_{config.scenario}_"
        f"{time_steps[0].strftime('%m-%d_%H-%M')}_d{duration_h}h/gurobi_logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    all_results = []
    for direction in tqdm(directions, desc=f"FFOR directions (d={duration_h}h)"):
        direction_id = direction["direction_id"]
        theta_deg = direction["theta_deg"]
        alpha = direction["alpha"]
        beta = direction["beta"]

        print(f"\n  Direction {direction_id + 1}/{len(directions)}: "
              f"theta={theta_deg:.1f}°, alpha={alpha:.4f}, beta={beta:.4f}")

        res = opf.solve_OPF(
            full_model,
            alpha,
            beta,
            log_file=os.path.join(log_dir, f"direction_{direction_id:02d}.log"),
        )

        # Tag the result with direction metadata for downstream saving
        res.update({
            "direction_id": direction_id,
            "theta_rad": direction["theta_rad"],
            "theta_deg": theta_deg,
            "alpha": alpha,
            "beta": beta,
        })
        all_results.append(res)

    return all_results

def main():
    print("Starting program...")
    print(f"Scenario: {config.scenario}")

    # --- ONE-TIME SETUP (independent of duration) ---
    net_data = opf.load_network_and_extract()
    load_df = nfa.map_load_time_series()
    correlation_df = corr.estimate_correlations(load_df)
    if config.scenario == "with_battery":
        correlation_df = corr.add_battery_capacity(correlation_df)

    temp_df = pd.read_csv("00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv",
                          parse_dates=['date'], index_col='date')
    temp_df.index = pd.DatetimeIndex(temp_df.index).tz_localize(None)

    pv_df = pd.read_csv("00-INPUT-DATA/PV-DATA/PV_timeseries.csv",
                        parse_dates=["time"], index_col="time")
    pv_df.index = pd.DatetimeIndex(pv_df.index).tz_localize(None)

    # --- DURATION SWEEP ---
    for duration_h in config.sustained_durations_h:
        print(f"\n{'='*60}\nSolving FFOR for sustained duration = {duration_h} h\n{'='*60}")

        time_steps = pd.date_range(
            start=config.start_time,
            periods=duration_h,            # exactly N hourly slots
            freq=config.timestep_freq,
        )

        # Validate data coverage
        missing_temp = time_steps.difference(temp_df.index)
        missing_pv = time_steps.difference(pv_df.index)
        if len(missing_temp) or len(missing_pv):
            raise ValueError(f"Missing data for duration {duration_h}h: "
                             f"temp={list(missing_temp)}, pv={list(missing_pv)}")

        pv_cf = pv_df.loc[time_steps, "electricity"]

        # --- HP baseline per timestep ---
        hp_base_df = pd.DataFrame(index=correlation_df.index, columns=time_steps)
        for t in time_steps:
            temp_t = float(temp_df.loc[t, 'temperature_2m'])
            hp_base_t = nfa.compute_hp_baseline(correlation_df, temp_t)
            hp_base_df[t] = hp_base_t['P_hp_max'] / net_data["S_base"]

        # --- Per-timestep device flex + nodal approximation ---
        aggregation_per_timestep, device_flex_per_timestep = \
            build_per_timestep_data(time_steps, correlation_df, pv_cf,
                                    hp_base_df, net_data)

        # --- Bus reindexing ---
        bus_lookup = net_data["net"]._pd2ppc_lookups['bus']
        device_flex_per_timestep = {
            t: df.set_index(df.index.map(lambda i: int(bus_lookup[i])))
            for t, df in device_flex_per_timestep.items()
        }
        aggregation_per_timestep = {
            t: df.set_index(df.index.map(lambda i: int(bus_lookup[i])))
            for t, df in aggregation_per_timestep.items()
        }
        hp_base_df_renamed = hp_base_df.copy()
        hp_base_df_renamed.index = [int(bus_lookup[i]) for i in hp_base_df.index]

        # --- Build model, solve all directions, save ---
        full_model = opf.setup_multi_timestep_OPF(
            load_df, net_data, time_steps,
            aggregation_per_timestep, device_flex_per_timestep,
            hp_base_df_renamed, temp_df,
        )

        all_results = solve_all_directions(full_model, time_steps, duration_h)
        save_multi_direction_results(all_results, time_steps, duration_h)


if __name__ == "__main__":
    main()
