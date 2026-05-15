import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import plot_and_estimate_correlations as corr
import config
import optimal_power_flow as opf
import nodal_flexibility_approximation as nfa
import pyomo.environ as pyo


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
        directions.append({
            "direction_id": direction_id,
            "theta_rad": float(theta),
            "theta_deg": float(np.degrees(theta)),
            "alpha": -float(np.cos(theta)),
            "beta": -float(np.sin(theta)),
        })

    return directions


def save_multi_direction_results(all_results: list[dict]):
    """
    Saves optimization results for multiple alpha/beta directions.
    """

    output_dir = (
        f"01-RESULTS/single_ts_{config.scenario}_"
        f"{config.single_timestep.strftime('%m-%d_%H-%M')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    vertex_rows = []
    pcc_rows = []
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
            "theta_rad": theta_rad,
            "theta_deg": theta_deg,
            "alpha": alpha,
            "beta": beta,
            "status": status,
            "obj_value": res.get("obj_value"),
            "P_flex_pcc": res.get("P_flex_pcc"),
            "Q_flex_pcc": res.get("Q_flex_pcc"),
        })

        if status not in ("optimal", "locallyOptimal"):
            continue

        for t in res["P_pcc"].keys():
            pcc_rows.append({
                "direction_id": direction_id,
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
                "theta_rad": theta_rad,
                "theta_deg": theta_deg,
                "alpha": alpha,
                "beta": beta,
                "time": t,
                "bus": bus,
                "P_flex": P_flex,
                "Q_flex": res["Q_flex"][(t, bus)],
            })

    pd.DataFrame(vertex_rows).to_csv(
        os.path.join(output_dir, "ffor_vertices.csv"),
        index=False,
    )
    pd.DataFrame(pcc_rows).to_csv(
        os.path.join(output_dir, "pcc_results_all_directions.csv"),
        index=False,
    )
    pd.DataFrame(flex_rows).to_csv(
        os.path.join(output_dir, "flex_results_all_directions.csv"),
        index=False,
    )

    print(f"Saved results to: {output_dir}")


def main():
    print("""
          
    ###########################################################
                    Starting single-timestep program...
    ###########################################################
    """)

    print(f"Running scenario: {config.scenario}")
    print(f"Single timestep optimization: {config.single_timestep}")

    t = pd.Timestamp(config.single_timestep).tz_localize(None)
    time_steps = [t]

    print("Loading network and mapping load time series")
    net_data = opf.load_network_and_extract()
    load_df = nfa.map_load_time_series()

    print("Estimating correlations to compute aggregation data")
    correlation_df = corr.estimate_correlations(load_df)

    if config.scenario == "with_battery":
        correlation_df = corr.add_battery_capacity(correlation_df)
    else:
        correlation_df["cap_battery_mw"] = 0.0

    # --- Temperature / HP baseline ---
    temp_df = pd.read_csv(
        "00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv",
        parse_dates=["date"],
        index_col="date",
    )
    temp_df.index = pd.DatetimeIndex(temp_df.index).tz_localize(None)

    if t not in temp_df.index:
        raise ValueError(f"Missing temperature data for single_timestep={t}")

    temperature = float(temp_df.loc[t, "temperature_2m"])

    hp_base_t = nfa.compute_hp_baseline(correlation_df, temperature)
    P_hp_base = hp_base_t["P_hp_max"] / net_data["S_base"]
    P_hp_rated = correlation_df["cap_hp_mw"] / net_data["S_base"]

    # --- PV ---
    pv_df = pd.read_csv(
        "00-INPUT-DATA/PV-DATA/PV_timeseries.csv",
        parse_dates=["time"],
        index_col="time",
    )
    pv_df.index = pd.DatetimeIndex(pv_df.index).tz_localize(None)

    if t not in pv_df.index:
        raise ValueError(f"Missing PV data for single_timestep={t}")

    pv_cf = float(pv_df.loc[t, "electricity"])

    # --- Device flexibility table in p.u. ---
    device_flex = pd.DataFrame(
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

    # PV: positive flexibility by curtailment relative to baseline generation
    device_flex["P_PV_max"] = (
        pv_cf * correlation_df["cap_pv_mw"] / net_data["S_base"]
    )

    # HP sign convention:
    # P_flex > 0 means less consumption / higher net injection.
    # P_hp_flex = P_hp_base - P_hp_actual.
    device_flex["P_hp_flex_max"] = P_hp_base
    device_flex["P_hp_flex_min"] = P_hp_base - P_hp_rated

    if config.scenario == "with_battery":
        P_bat_rated = correlation_df["cap_battery_mw"] / net_data["S_base"]
        device_flex["P_chg_max"] = P_bat_rated
        device_flex["P_dis_max"] = P_bat_rated
        device_flex["S_bat_max"] = P_bat_rated
    else:
        device_flex["P_chg_max"] = 0.0
        device_flex["P_dis_max"] = 0.0
        device_flex["S_bat_max"] = 0.0

    # --- Compute nodal approximations ---
    aggregation_df = pd.DataFrame(
        columns=["H", "h", "d_soc"],
        index=device_flex.index,
    )

    for node in tqdm(device_flex.index, desc="Computing nodal approximations"):
        P_pv_max = float(device_flex.loc[node, "P_PV_max"])

        P_hp_flex_min = float(device_flex.loc[node, "P_hp_flex_min"])
        P_hp_flex_max = float(device_flex.loc[node, "P_hp_flex_max"])

        P_chg_max = float(device_flex.loc[node, "P_chg_max"])
        P_dis_max = float(device_flex.loc[node, "P_dis_max"])
        S_bat_max = float(device_flex.loc[node, "S_bat_max"])

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

        aggregation_df.loc[node] = {
            "H": H,
            "h": h,
            "d_soc": d_soc,
        }

    # --- Map pandapower bus IDs to internal ppc bus IDs ---
    bus_lookup = net_data["net"]._pd2ppc_lookups["bus"]
    aggregation_df.index = [
        int(bus_lookup[i]) for i in aggregation_df.index
    ]

    # --- Build OPF model ---
    print("Setting up single-timestep OPF")
    base_model = opf.setup_base_OPF(load_df, net_data, time_steps=time_steps)

    # Add objective once; solve_OPF updates alpha/beta later.
    full_model = opf.add_objective_function(base_model, config.alpha, config.beta)

    print("Adding aggregated flexibility constraints")
    full_model = opf.add_aggregated_flexibility_constraints(
        full_model,
        aggregation_df,
    )

    # For single-timestep FFOR, define PCC flexibility variables explicitly.
    # This makes ffor_vertices.csv consistent with the multi-timestep case.
    if full_model.find_component("P_flex_pcc") is None:
        full_model.P_flex_pcc = pyo.Var(initialize=0.0, bounds=(-2.0, 2.0))
    if full_model.find_component("Q_flex_pcc") is None:
        full_model.Q_flex_pcc = pyo.Var(initialize=0.0, bounds=(-2.0, 2.0))

    def single_P_flex_rule(m, tt):
        return m.P_pcc[tt] - m.P_pcc_base[tt] == m.P_flex_pcc

    def single_Q_flex_rule(m, tt):
        return m.Q_pcc[tt] - m.Q_pcc_base[tt] == m.Q_flex_pcc

    full_model.c_single_P_flex = pyo.Constraint(
        full_model.T,
        rule=single_P_flex_rule,
    )
    full_model.c_single_Q_flex = pyo.Constraint(
        full_model.T,
        rule=single_Q_flex_rule,
    )

    # --- Solve for multiple FFOR directions ---
    print("Solving OPF for multiple FFOR directions")

    directions = generate_ffor_directions(config.n_ffor_directions)
    all_results = []

    for direction in tqdm(directions, desc="Solving FFOR directions"):
        direction_id = direction["direction_id"]
        theta_rad = direction["theta_rad"]
        theta_deg = direction["theta_deg"]
        alpha = direction["alpha"]
        beta = direction["beta"]

        print(
            f"\nSolving direction {direction_id + 1}/{len(directions)}: "
            f"theta={theta_deg:.1f}°, alpha={alpha:.4f}, beta={beta:.4f}"
        )

        res = opf.solve_OPF(full_model, alpha, beta)

        res["direction_id"] = direction_id
        res["theta_rad"] = theta_rad
        res["theta_deg"] = theta_deg
        res["alpha"] = alpha
        res["beta"] = beta

        all_results.append(res)

    print("Saving single-timestep FFOR results")
    save_multi_direction_results(all_results)


if __name__ == "__main__":
    main()