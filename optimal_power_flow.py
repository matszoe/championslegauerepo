import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import config
import pandapower_read_csv as pp_read
import pandapower as pp

from pandapower.pypower.makeYbus import makeYbus
from pyomo.opt import SolverFactory


def _to_naive_timestamp(t):
    """
    Converts a timestamp to timezone-naive pandas Timestamp.
    """
    ts = pd.Timestamp(t)
    if ts.tz is not None:
        return ts.tz_convert(None)
    return ts


def load_network_and_extract() -> dict:
    """
    Loads the network and extracts the necessary data for the OPF problem.
    """
    net = pp_read.read_net_from_csv("00-INPUT-DATA/norway_data/")

    X_THRESHOLD = 1e-4
    Z_base = net.bus["vn_kv"].iloc[0] ** 2 / 10.0  # baseMVA=10

    for idx in net.line.index:
        r_pu = (
            net.line.at[idx, "r_ohm_per_km"]
            * net.line.at[idx, "length_km"]
            / Z_base
        )
        x_pu = (
            net.line.at[idx, "x_ohm_per_km"]
            * net.line.at[idx, "length_km"]
            / Z_base
        )

        # Regularize very small impedances.
        if r_pu < X_THRESHOLD:
            net.line.at[idx, "r_ohm_per_km"] = (
                X_THRESHOLD * Z_base
            ) / net.line.at[idx, "length_km"]

        if x_pu < X_THRESHOLD:
            net.line.at[idx, "x_ohm_per_km"] = (
                X_THRESHOLD * Z_base
            ) / net.line.at[idx, "length_km"]

    pp.runpp(net, algorithm="nr", numba=False)

    ppc = net._ppc

    S_base = ppc["baseMVA"]
    Ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Ybus = Ybus.toarray()

    G = Ybus.real
    B = Ybus.imag
    n = G.shape[0]

    J_Pth = -B.copy()
    for i in range(n):
        J_Pth[i, i] = np.sum(B[i, :]) - B[i, i]

    J_QU = -B.copy()
    for i in range(n):
        J_QU[i, i] = np.sum(B[i, :]) - B[i, i]

    J_PU = -G.copy()
    for i in range(n):
        J_PU[i, i] = np.sum(G[i, :]) - G[i, i]

    J_Qth = -J_PU.copy()

    V_base = net.bus["vn_kv"].iloc[0]
    rate_A = (
        net.line["max_i_ka"].values
        * V_base
        * np.sqrt(3)
        / S_base
    )

    P_pcc_base = net.res_ext_grid["p_mw"].values[0] / S_base
    Q_pcc_base = net.res_ext_grid["q_mvar"].values[0] / S_base

    bus_lookup = net._pd2ppc_lookups["bus"]

    slack_pp_bus = int(net.ext_grid["bus"].iloc[0])
    slack_ppc_idx = int(bus_lookup[slack_pp_bus])

    return {
        "J_Pth": J_Pth,
        "J_PU": J_PU,
        "J_Qth": J_Qth,
        "J_QU": J_QU,
        "rate_A": rate_A,
        "P_pcc_base": P_pcc_base,
        "Q_pcc_base": Q_pcc_base,
        "S_base": S_base,
        "bus_lookup": bus_lookup,
        "n_buses": n,
        "slack_idx": slack_ppc_idx,
        "ppc": ppc,
        "net": net,
    }


def setup_base_OPF(
    load_df: pd.DataFrame,
    net_data: dict,
    time_steps: list,
) -> pyo.ConcreteModel:
    """
    Builds the base network OPF without flexibility.
    """
    JACOBIAN_THRESHOLD = 1e-6

    time_steps = [_to_naive_timestamp(t) for t in time_steps]

    net = net_data["net"]
    J_Pth = net_data["J_Pth"].copy()
    J_PU = net_data["J_PU"].copy()
    J_Qth = net_data["J_Qth"].copy()
    J_QU = net_data["J_QU"].copy()

    J_Pth[np.abs(J_Pth) < JACOBIAN_THRESHOLD] = 0.0
    J_PU[np.abs(J_PU) < JACOBIAN_THRESHOLD] = 0.0
    J_Qth[np.abs(J_Qth) < JACOBIAN_THRESHOLD] = 0.0
    J_QU[np.abs(J_QU) < JACOBIAN_THRESHOLD] = 0.0

    rate_A = net_data["rate_A"]
    S_base = net_data["S_base"]
    n = net_data["n_buses"]
    slack = net_data["slack_idx"]
    bus_lookup = net_data["bus_lookup"]

    buses = list(range(n))
    lines = list(range(len(net.line)))

    # Branch classification.
    X_THRESHOLD = 1e-4
    Z_base = net.bus["vn_kv"].iloc[0] ** 2 / S_base

    normal_lines = []
    ideal_lines = []

    for l in lines:
        r_pu = float(
            net.line["r_ohm_per_km"].iloc[l]
            * net.line["length_km"].iloc[l]
            / Z_base
        )
        x_pu = float(
            net.line["x_ohm_per_km"].iloc[l]
            * net.line["length_km"].iloc[l]
            / Z_base
        )

        if x_pu <= X_THRESHOLD and r_pu <= X_THRESHOLD:
            ideal_lines.append(l)
        else:
            normal_lines.append(l)

    print(f"Normal lines: {len(normal_lines)}, Ideal lines: {len(ideal_lines)}")

    def get_load_pu(t, bus_internal):
        bus_i = net.bus.index[bus_internal]
        if bus_i in load_df.columns and t in load_df.index:
            return float(load_df.loc[t, bus_i]) / S_base
        return 0.0

    pf_ratio_series = pd.Series(0.0, index=range(n))

    for _, load_row in net.load.iterrows():
        pp_idx = int(bus_lookup[load_row["bus"]])
        p = load_row["p_mw"]
        q = load_row["q_mvar"]
        pf_ratio_series[pp_idx] = (q / p) if p != 0 else 0.0

    model = pyo.ConcreteModel()

    model.T = pyo.Set(initialize=time_steps)
    model.B = pyo.Set(initialize=buses)
    model.L = pyo.Set(initialize=lines)
    model.L_normal = pyo.Set(initialize=normal_lines)
    model.L_ideal = pyo.Set(initialize=ideal_lines)

    model.V_min = pyo.Param(initialize=float(net.bus["min_vm_pu"].iloc[0]))
    model.V_max = pyo.Param(initialize=float(net.bus["max_vm_pu"].iloc[0]))
    model.slack = pyo.Param(initialize=slack)

    model.alpha = pyo.Param(initialize=0.0, mutable=True)
    model.beta = pyo.Param(initialize=0.0, mutable=True)

    model.pf_ratio = pyo.Param(
        model.B,
        initialize={i: float(pf_ratio_series[i]) for i in buses},
    )

    model.rate_A = pyo.Param(
        model.L,
        initialize={l: float(rate_A[l]) for l in lines},
    )

    model.P_load = pyo.Param(
        model.T,
        model.B,
        initialize={
            (t, b): get_load_pu(t, b)
            for t in time_steps
            for b in buses
        },
    )

    def get_pcc_base_pu(t):
        return sum(get_load_pu(t, b) for b in buses if b != slack)

    def get_qcc_base_pu(t):
        return sum(
            get_load_pu(t, b) * float(pf_ratio_series[b])
            for b in buses
            if b != slack
        )

    model.P_pcc_base = pyo.Param(
        model.T,
        initialize={t: float(get_pcc_base_pu(t)) for t in time_steps},
    )

    model.Q_pcc_base = pyo.Param(
        model.T,
        initialize={t: float(get_qcc_base_pu(t)) for t in time_steps},
    )

    model.J_Pth = pyo.Param(
        model.B,
        model.B,
        initialize={(i, k): float(J_Pth[i, k]) for i in buses for k in buses},
    )

    model.J_PU = pyo.Param(
        model.B,
        model.B,
        initialize={(i, k): float(J_PU[i, k]) for i in buses for k in buses},
    )

    model.J_Qth = pyo.Param(
        model.B,
        model.B,
        initialize={(i, k): float(J_Qth[i, k]) for i in buses for k in buses},
    )

    model.J_QU = pyo.Param(
        model.B,
        model.B,
        initialize={(i, k): float(J_QU[i, k]) for i in buses for k in buses},
    )

    model.br_f = pyo.Param(
        model.L,
        initialize={
            l: int(net._pd2ppc_lookups["bus"][net.line["from_bus"].iloc[l]])
            for l in lines
        },
    )

    model.br_t = pyo.Param(
        model.L,
        initialize={
            l: int(net._pd2ppc_lookups["bus"][net.line["to_bus"].iloc[l]])
            for l in lines
        },
    )

    model.br_r = pyo.Param(
        model.L,
        initialize={
            l: float(
                net.line["r_ohm_per_km"].iloc[l]
                * net.line["length_km"].iloc[l]
                / Z_base
            )
            for l in lines
        },
    )

    model.br_x = pyo.Param(
        model.L,
        initialize={
            l: float(
                net.line["x_ohm_per_km"].iloc[l]
                * net.line["length_km"].iloc[l]
                / Z_base
            )
            for l in lines
        },
    )

    model.theta = pyo.Var(model.T, model.B, initialize=0.0, bounds=(-0.5, 0.5))
    model.dV = pyo.Var(model.T, model.B, initialize=0.0, bounds=(-0.5, 0.5))

    SCALE_FACTOR = 1.0

    P_total_max = float(load_df.sum(axis=1).max()) / S_base
    P_node_max = float(load_df.max().max()) / S_base

    P_grid_max = max(2.0 * P_total_max, 1.0)
    P_node_bound = max(2.0 * P_node_max, 1.0)

    def inj_bounds(m, t, i):
        if i == slack:
            return (-P_grid_max, P_grid_max)
        return (-P_node_bound, P_node_bound)

    model.P_inj = pyo.Var(
        model.T,
        model.B,
        initialize=0.0,
        bounds=inj_bounds,
    )

    model.Q_inj = pyo.Var(
        model.T,
        model.B,
        initialize=0.0,
        bounds=inj_bounds,
    )

    model.P_pcc = pyo.Var(
        model.T,
        initialize=0.0,
        bounds=(-P_grid_max, P_grid_max),
    )

    model.Q_pcc = pyo.Var(
        model.T,
        initialize=0.0,
        bounds=(-P_grid_max, P_grid_max),
    )

    model.P_line = pyo.Var(
        model.T,
        model.L_normal,
        initialize=0.0,
        bounds=lambda m, t, l: (-m.rate_A[l], m.rate_A[l]),
    )
    model.Q_line = pyo.Var(
        model.T,
        model.L_normal,
        initialize=0.0,
        bounds=lambda m, t, l: (-m.rate_A[l], m.rate_A[l]),
    )

    nonzero_Pth = {
        (i, k)
        for i in buses
        for k in buses
        if abs(J_Pth[i, k]) > JACOBIAN_THRESHOLD
    }

    nonzero_PU = {
        (i, k)
        for i in buses
        for k in buses
        if abs(J_PU[i, k]) > JACOBIAN_THRESHOLD
    }

    nonzero_Qth = {
        (i, k)
        for i in buses
        for k in buses
        if abs(J_Qth[i, k]) > JACOBIAN_THRESHOLD
    }

    nonzero_QU = {
        (i, k)
        for i in buses
        for k in buses
        if abs(J_QU[i, k]) > JACOBIAN_THRESHOLD
    }

    pf_P_scale = {
        i: 1.0
        / max(
            1.0,
            max(
                (
                    abs(J_Pth[i, k]) if (i, k) in nonzero_Pth else 0.0
                )
                for k in buses
            ),
            max(
                (
                    abs(J_PU[i, k]) if (i, k) in nonzero_PU else 0.0
                )
                for k in buses
            ),
        )
        for i in buses
    }

    pf_Q_scale = {
        i: 1.0
        / max(
            1.0,
            max(
                (
                    abs(J_Qth[i, k]) if (i, k) in nonzero_Qth else 0.0
                )
                for k in buses
            ),
            max(
                (
                    abs(J_QU[i, k]) if (i, k) in nonzero_QU else 0.0
                )
                for k in buses
            ),
        )
        for i in buses
    }

    line_flow_scale = {}
    for l in normal_lines:
        r = float(net.line["r_ohm_per_km"].iloc[l] * net.line["length_km"].iloc[l] / Z_base)
        x = float(net.line["x_ohm_per_km"].iloc[l] * net.line["length_km"].iloc[l] / Z_base)
        d = r**2 + x**2 + 1e-9
        line_flow_scale[l] = 1.0 / max(1.0, abs(r / d), abs(x / d))

    model.c_slack_theta = pyo.Constraint(
        model.T,
        rule=lambda m, t: m.theta[t, slack] == 0.0,
    )

    model.c_slack_dV = pyo.Constraint(
        model.T,
        rule=lambda m, t: m.dV[t, slack] == 0.0,
    )

    model.c_pf_P = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: pf_P_scale[i] * m.P_inj[t, i]
        == sum(
            pf_P_scale[i] * m.J_Pth[i, k] * m.theta[t, k] / SCALE_FACTOR
            + pf_P_scale[i] * m.J_PU[i, k] * m.dV[t, k] / SCALE_FACTOR
            for k in buses
            if (i, k) in nonzero_Pth or (i, k) in nonzero_PU
        ),
    )

    model.c_pf_Q = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: pf_Q_scale[i] * m.Q_inj[t, i]
        == sum(
            pf_Q_scale[i] * m.J_Qth[i, k] * m.theta[t, k] / SCALE_FACTOR
            + pf_Q_scale[i] * m.J_QU[i, k] * m.dV[t, k] / SCALE_FACTOR
            for k in buses
            if (i, k) in nonzero_Qth or (i, k) in nonzero_QU
        ),
    )

    model.c_bal_P = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: m.P_inj[t, i]
        == (m.P_pcc[t] if i == slack else -m.P_load[t, i]),
    )

    model.c_bal_Q = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: m.Q_inj[t, i]
        == (
            m.Q_pcc[t]
            if i == slack
            else -m.P_load[t, i] * m.pf_ratio[i]
        ),
    )

    model.c_v_min = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: pyo.Constraint.Skip
        if i == slack
        else m.dV[t, i] >= (m.V_min - 1.0) * SCALE_FACTOR,
    )

    model.c_v_max = pyo.Constraint(
        model.T,
        model.B,
        rule=lambda m, t, i: pyo.Constraint.Skip
        if i == slack
        else m.dV[t, i] <= (m.V_max - 1.0) * SCALE_FACTOR,
    )

    def line_flow_P(m, t, l):
        f = m.br_f[l]
        tb = m.br_t[l]
        r = m.br_r[l]
        x = m.br_x[l]
        d = r**2 + x**2 + 1e-9
        s = line_flow_scale[l]

        return s * m.P_line[t, l] == (
            s * (r / d) * (m.dV[t, f] - m.dV[t, tb]) / SCALE_FACTOR
            + s * (x / d) * (m.theta[t, f] - m.theta[t, tb]) / SCALE_FACTOR
        )

    def line_flow_Q(m, t, l):
        f = m.br_f[l]
        tb = m.br_t[l]
        r = m.br_r[l]
        x = m.br_x[l]
        d = r**2 + x**2 + 1e-9
        s = line_flow_scale[l]

        return s * m.Q_line[t, l] == (
            s * (x / d) * (m.dV[t, f] - m.dV[t, tb]) / SCALE_FACTOR
            - s * (r / d) * (m.theta[t, f] - m.theta[t, tb]) / SCALE_FACTOR
        )

    model.c_line_P = pyo.Constraint(model.T, model.L_normal, rule=line_flow_P)
    model.c_line_Q = pyo.Constraint(model.T, model.L_normal, rule=line_flow_Q)

    model.c_ideal_theta = pyo.Constraint(
        model.T,
        model.L_ideal,
        rule=lambda m, t, l: pyo.Constraint.Skip,
    )

    model.c_ideal_dV = pyo.Constraint(
        model.T,
        model.L_ideal,
        rule=lambda m, t, l: pyo.Constraint.Skip,
    )

    model.c_line_lim = pyo.Constraint(
        model.T,
        model.L_normal,
        rule=lambda m, t, l: m.P_line[t, l] ** 2 + m.Q_line[t, l] ** 2
        <= m.rate_A[l] ** 2,
    )

    return model


def add_objective_function(
    model: pyo.ConcreteModel,
    alpha: float,
    beta: float,
) -> pyo.ConcreteModel:
    """
    Adds / updates the directional objective function.
    """
    model.alpha.set_value(alpha)
    model.beta.set_value(beta)

    if model.find_component("obj") is not None:
        model.del_component("obj")

    def obj_rule(m):
        return sum(
            m.alpha * (m.P_pcc[t] - m.P_pcc_base[t])
            + m.beta * (m.Q_pcc[t] - m.Q_pcc_base[t])
            for t in m.T
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model


def add_static_flexibility_layer(
    model: pyo.ConcreteModel,
    aggregation_per_timestep,
    device_flex_per_timestep=None,
) -> pyo.ConcreteModel:
    """
    Adds static nodal flexibility.

    Shared by single- and multi-timestep OPFs.

    Adds:
      - P_flex, Q_flex
      - optional device variables:
          P_pv_flex, Q_pv_flex,
          P_hp_flex,
          P_bat_flex, Q_bat_flex
      - device constraints if device_flex_per_timestep is provided
      - aggregate H/h constraints
      - aggregate cone-radius constraints
      - nodal balance with flexibility
    """
    time_list = list(model.T)

    if isinstance(aggregation_per_timestep, pd.DataFrame):
        if len(time_list) != 1:
            raise ValueError(
                "Single aggregation DataFrame can only be used with one timestep."
            )
        aggregation_per_timestep = {time_list[0]: aggregation_per_timestep}

    if isinstance(device_flex_per_timestep, pd.DataFrame):
        if len(time_list) != 1:
            raise ValueError(
                "Single device_flex DataFrame can only be used with one timestep."
            )
        device_flex_per_timestep = {time_list[0]: device_flex_per_timestep}

    bus_ids = aggregation_per_timestep[time_list[0]].index.tolist()

    # Remove old flexibility layer if present.
    components_to_delete = [
        "c_bal_P",
        "c_bal_Q",
        "c_p_flex_decomposition",
        "c_q_flex_decomposition",
        "c_pv_p_min",
        "c_pv_p_max",
        "c_pv_q_max",
        "c_pv_q_min",
        "c_hp_p_min",
        "c_hp_p_max",
        "c_bat_p_dis",
        "c_bat_p_chg",
        "c_bat_s",
        "c_flex_linear",
        "c_flex_soc",
        "P_flex",
        "Q_flex",
        "P_pv_flex",
        "Q_pv_flex",
        "P_hp_flex",
        "P_bat_flex",
        "Q_bat_flex",
        "P_pv_max",
        "P_hp_min",
        "P_hp_max",
        "P_chg_max",
        "P_dis_max",
        "S_bat_max",
        "FLEX_LIN_IDX",
        "H_0",
        "H_1",
        "h_rhs",
        "SOC_IDX",
        "d_soc",
        "B_flex",
    ]

    for comp_name in components_to_delete:
        if model.find_component(comp_name) is not None:
            model.del_component(comp_name)

    model.B_flex = pyo.Set(initialize=bus_ids)

    model.P_flex = pyo.Var(
        model.T,
        model.B_flex,
        initialize=0.0,
        bounds=(-2.0, 2.0),
    )

    model.Q_flex = pyo.Var(
        model.T,
        model.B_flex,
        initialize=0.0,
        bounds=(-2.0, 2.0),
    )

    use_device_decomposition = device_flex_per_timestep is not None

    if use_device_decomposition:
        model.P_pv_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)
        model.Q_pv_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)

        model.P_hp_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)

        model.P_bat_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)
        model.Q_bat_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)

        def get_device_value(t, b, col, default=0.0):
            df_t = device_flex_per_timestep[t]
            if b not in df_t.index:
                return default
            if col not in df_t.columns:
                return default
            val = df_t.loc[b, col]
            return float(val) if not pd.isna(val) else default

        P_pv_max_data = {
            (t, b): get_device_value(t, b, "P_PV_max")
            for t in time_list
            for b in bus_ids
        }

        P_hp_min_data = {
            (t, b): get_device_value(t, b, "P_hp_flex_min")
            for t in time_list
            for b in bus_ids
        }

        P_hp_max_data = {
            (t, b): get_device_value(t, b, "P_hp_flex_max")
            for t in time_list
            for b in bus_ids
        }

        P_chg_max_data = {
            (t, b): get_device_value(t, b, "P_chg_max")
            for t in time_list
            for b in bus_ids
        }

        P_dis_max_data = {
            (t, b): get_device_value(t, b, "P_dis_max")
            for t in time_list
            for b in bus_ids
        }

        S_bat_max_data = {
            (t, b): get_device_value(t, b, "S_bat_max")
            for t in time_list
            for b in bus_ids
        }

        model.P_pv_max = pyo.Param(
            model.T,
            model.B_flex,
            initialize=P_pv_max_data,
        )

        model.P_hp_min = pyo.Param(
            model.T,
            model.B_flex,
            initialize=P_hp_min_data,
        )

        model.P_hp_max = pyo.Param(
            model.T,
            model.B_flex,
            initialize=P_hp_max_data,
        )

        model.P_chg_max = pyo.Param(
            model.T,
            model.B_flex,
            initialize=P_chg_max_data,
        )

        model.P_dis_max = pyo.Param(
            model.T,
            model.B_flex,
            initialize=P_dis_max_data,
        )

        model.S_bat_max = pyo.Param(
            model.T,
            model.B_flex,
            initialize=S_bat_max_data,
        )

        tan_phi_pv = (
            np.sqrt(1.0 - config.pv_cf_lower_limit**2)
            / config.pv_cf_lower_limit
        )

        model.c_pv_p_min = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_pv_flex[t, b] >= 0.0,
        )

        model.c_pv_p_max = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_pv_flex[t, b] <= m.P_pv_max[t, b],
        )

        model.c_pv_q_max = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.Q_pv_flex[t, b]
            <= tan_phi_pv * m.P_pv_flex[t, b],
        )

        model.c_pv_q_min = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: -m.Q_pv_flex[t, b]
            <= tan_phi_pv * m.P_pv_flex[t, b],
        )

        model.c_hp_p_min = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_hp_flex[t, b] >= m.P_hp_min[t, b],
        )

        model.c_hp_p_max = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_hp_flex[t, b] <= m.P_hp_max[t, b],
        )

        model.c_bat_p_dis = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_bat_flex[t, b] <= m.P_dis_max[t, b],
        )

        model.c_bat_p_chg = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: -m.P_bat_flex[t, b] <= m.P_chg_max[t, b],
        )

        def bat_s_rule(m, t, b):
            if pyo.value(m.S_bat_max[t, b]) <= 1e-12:
                return m.Q_bat_flex[t, b] == 0.0

            return (
                m.P_bat_flex[t, b] ** 2
                + m.Q_bat_flex[t, b] ** 2
                <= m.S_bat_max[t, b] ** 2
            )

        model.c_bat_s = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=bat_s_rule,
        )

        model.c_p_flex_decomposition = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.P_flex[t, b]
            == m.P_pv_flex[t, b]
            + m.P_hp_flex[t, b]
            + m.P_bat_flex[t, b],
        )

        model.c_q_flex_decomposition = pyo.Constraint(
            model.T,
            model.B_flex,
            rule=lambda m, t, b: m.Q_flex[t, b]
            == m.Q_pv_flex[t, b]
            + m.Q_bat_flex[t, b],
        )

    linear_index = []
    H_0 = {}
    H_1 = {}
    h_rhs = {}

    for t in time_list:
        agg_t = aggregation_per_timestep[t]

        for bus_id, row in agg_t.iterrows():
            H = np.asarray(row["H"], dtype=float)
            h = np.asarray(row["h"], dtype=float)

            for i in range(len(h)):
                key = (t, bus_id, i)
                linear_index.append(key)
                H_0[key] = float(H[i, 0])
                H_1[key] = float(H[i, 1])
                h_rhs[key] = float(h[i])

    model.FLEX_LIN_IDX = pyo.Set(initialize=linear_index, dimen=3)
    model.H_0 = pyo.Param(model.FLEX_LIN_IDX, initialize=H_0)
    model.H_1 = pyo.Param(model.FLEX_LIN_IDX, initialize=H_1)
    model.h_rhs = pyo.Param(model.FLEX_LIN_IDX, initialize=h_rhs)

    def flex_linear_rule(m, t, bus_id, i):
        return (
            m.H_0[t, bus_id, i] * m.P_flex[t, bus_id]
            + m.H_1[t, bus_id, i] * m.Q_flex[t, bus_id]
            <= m.h_rhs[t, bus_id, i]
        )

    model.c_flex_linear = pyo.Constraint(
        model.FLEX_LIN_IDX,
        rule=flex_linear_rule,
    )

    soc_index = []
    d_soc_data = {}

    for t in time_list:
        agg_t = aggregation_per_timestep[t]
        for bus_id in agg_t[agg_t["d_soc"] > 0].index:
            soc_index.append((t, bus_id))
            d_soc_data[(t, bus_id)] = float(agg_t.loc[bus_id, "d_soc"])

    model.SOC_IDX = pyo.Set(initialize=soc_index, dimen=2)
    model.d_soc = pyo.Param(model.SOC_IDX, initialize=d_soc_data)

    def soc_rule(m, t, k):
        return (
            m.P_flex[t, k] ** 2 + m.Q_flex[t, k] ** 2
            <= m.d_soc[t, k] ** 2
        )

    if not use_device_decomposition:
        model.c_flex_soc = pyo.Constraint(model.SOC_IDX, rule=soc_rule)

    slack = pyo.value(model.slack)

    def bal_P_rule(m, t, i):
        if i == slack:
            return m.P_inj[t, i] == m.P_pcc[t]

        flex = m.P_flex[t, i] if i in bus_ids else 0.0
        return m.P_inj[t, i] == -m.P_load[t, i] + flex

    def bal_Q_rule(m, t, i):
        if i == slack:
            return m.Q_inj[t, i] == m.Q_pcc[t]

        flex = m.Q_flex[t, i] if i in bus_ids else 0.0
        return m.Q_inj[t, i] == -m.P_load[t, i] * m.pf_ratio[i] + flex

    model.c_bal_P = pyo.Constraint(model.T, model.B, rule=bal_P_rule)
    model.c_bal_Q = pyo.Constraint(model.T, model.B, rule=bal_Q_rule)

    return model


def add_pcc_flexibility_constraints(
    model: pyo.ConcreteModel,
    p_bound: float = 2.0,
    q_bound: float = 2.0,
) -> pyo.ConcreteModel:
    """
    Adds PCC flexibility variables and constraints.

    Single timestep:
        gives the one-step PCC deviation.

    Multi timestep:
        enforces sustained constant PCC deviation over all timesteps.
    """
    for comp_name in [
        "P_flex_pcc",
        "Q_flex_pcc",
        "c_const_P_flex",
        "c_const_Q_flex",
        "c_single_P_flex",
        "c_single_Q_flex",
    ]:
        if model.find_component(comp_name) is not None:
            model.del_component(comp_name)

    model.P_flex_pcc = pyo.Var(initialize=0.0, bounds=(-p_bound, p_bound))
    model.Q_flex_pcc = pyo.Var(initialize=0.0, bounds=(-q_bound, q_bound))

    model.c_const_P_flex = pyo.Constraint(
        model.T,
        rule=lambda m, t: m.P_pcc[t] - m.P_pcc_base[t] == m.P_flex_pcc,
    )

    model.c_const_Q_flex = pyo.Constraint(
        model.T,
        rule=lambda m, t: m.Q_pcc[t] - m.Q_pcc_base[t] == m.Q_flex_pcc,
    )

    return model


def add_aggregated_flexibility_constraints(
    model: pyo.ConcreteModel,
    approximation_df: pd.DataFrame,
) -> pyo.ConcreteModel:
    """
    Backward-compatible wrapper.

    Adds only aggregate flexibility constraints without device decomposition.
    """
    return add_static_flexibility_layer(
        model=model,
        aggregation_per_timestep=approximation_df,
        device_flex_per_timestep=None,
    )


def setup_single_timestep_OPF(
    load_df: pd.DataFrame,
    net_data: dict,
    timestep,
    aggregation_df: pd.DataFrame,
    device_flex_df: pd.DataFrame | None = None,
) -> pyo.ConcreteModel:
    """
    Builds a single-timestep OPF with the current static flexibility layer.
    """
    t = _to_naive_timestamp(timestep)

    model = setup_base_OPF(load_df, net_data, time_steps=[t])
    model = add_objective_function(model, config.alpha, config.beta)

    model = add_static_flexibility_layer(
        model=model,
        aggregation_per_timestep={t: aggregation_df},
        device_flex_per_timestep={t: device_flex_df}
        if device_flex_df is not None
        else None,
    )

    model = add_pcc_flexibility_constraints(model)

    return model


def setup_multi_timestep_OPF(
    load_df: pd.DataFrame,
    net_data: dict,
    time_steps: list,
    aggregation_per_timestep: dict,
    device_flex_per_timestep: dict,
    hp_base_df: pd.DataFrame,
    temperature_series: pd.DataFrame,
    delta_t: float = 1.0,
    T_preferred: float = config.T_room_initial,
    T_min: float = 18.0,
    T_max: float = 22.0,
) -> pyo.ConcreteModel:
    """
    Builds a multi-timestep OPF.

    Uses the same static flexibility layer as the single-timestep OPF,
    and then adds time-coupled HP temperature and battery SOC dynamics.
    """
    time_list = [_to_naive_timestamp(t) for t in time_steps]

    model = setup_base_OPF(load_df, net_data, time_steps=time_list)
    model = add_objective_function(model, config.alpha, config.beta)

    model = add_static_flexibility_layer(
        model=model,
        aggregation_per_timestep=aggregation_per_timestep,
        device_flex_per_timestep=device_flex_per_timestep,
    )

    model = add_pcc_flexibility_constraints(model)

    S_base = net_data["S_base"]
    bus_ids = list(model.B_flex)

    # --- HP thermal dynamics ---
    q_heat_profile = pd.read_csv(
        "00-INPUT-DATA/HP-DATA/hp_profile.csv",
        index_col=0,
    )
    q_heat_profile.index = q_heat_profile.index.astype(int)
    q_heat_profile.columns = q_heat_profile.columns.astype(int)

    hp_buses = [
        b
        for b in bus_ids
        if b in hp_base_df.index and hp_base_df.loc[b].max() > 1e-9
    ]

    model.B_hp = pyo.Set(initialize=hp_buses)

    model.T_room = pyo.Var(
        model.T,
        model.B_hp,
        initialize=T_preferred,
        bounds=(T_min, T_max),
    )

    q_heat_data = {}
    p_hp_base_data = {}

    for t in time_list:
        hour = t.hour
        temp = float(temperature_series.loc[t, "temperature_2m"])
        temp_class = int(np.clip(round(temp), -14, 18))
        q_val = float(q_heat_profile.loc[hour, temp_class])

        for b in hp_buses:
            q_heat_data[(t, b)] = q_val
            val = hp_base_df.loc[b, t] if b in hp_base_df.index else 0.0
            p_hp_base_data[(t, b)] = float(val) if not pd.isna(val) else 0.0

    model.q_heat = pyo.Param(model.T, model.B_hp, initialize=q_heat_data)
    model.P_hp_base = pyo.Param(model.T, model.B_hp, initialize=p_hp_base_data)

    model.T_IDX = pyo.Set(initialize=range(len(time_list)))

    def thermal_evolution_rule(m, t_idx, b):
        if t_idx == 0:
            return pyo.Constraint.Skip

        t_curr = time_list[t_idx]
        t_prev = time_list[t_idx - 1]

        p_base = p_hp_base_data.get((t_prev, b), 0.0)

        if p_base < 1e-6:
            return m.P_hp_flex[t_prev, b] == 0.0

        return (
            p_base * m.T_room[t_curr, b]
            == p_base * m.T_room[t_prev, b]
            - m.P_hp_flex[t_prev, b] * m.q_heat[t_prev, b] * delta_t
        )

    model.c_thermal = pyo.Constraint(
        model.T_IDX,
        model.B_hp,
        rule=thermal_evolution_rule,
    )

    for b in hp_buses:
        model.T_room[time_list[0], b].fix(T_preferred)

    # --- Battery SOC dynamics ---
    soc_buses = sorted({k for (_, k) in list(model.SOC_IDX)})

    if soc_buses:
        model.B_soc_dyn = pyo.Set(initialize=soc_buses)

        model.SOC = pyo.Var(
            model.T,
            model.B_soc_dyn,
            initialize=config.SOC_inital,
            bounds=(config.soc_min, config.soc_max),
        )

        def soc_evolution_rule(m, t_idx, b):
            if t_idx == 0:
                return pyo.Constraint.Skip

            t_curr = time_list[t_idx]
            t_prev = time_list[t_idx - 1]

            E_bat_pu = config.battery_capacity_at_FC / S_base

            return (
                m.SOC[t_curr, b]
                == m.SOC[t_prev, b]
                - m.P_bat_flex[t_prev, b] * delta_t / E_bat_pu
            )

        model.c_soc_evolution = pyo.Constraint(
            model.T_IDX,
            model.B_soc_dyn,
            rule=soc_evolution_rule,
        )

        for b in soc_buses:
            model.SOC[time_list[0], b].fix(config.SOC_inital)

    return model


def solve_OPF(
    model: pyo.ConcreteModel,
    alpha: float,
    beta: float,
    time_limit: int = 300,
    mip_gap: float = 1e-4,
    log_file: str | None = None,
    retry_on_error: bool = True,
) -> dict:
    """
    Solves the OPF model using Gurobi for a given PQ direction.
    """
    model.alpha.set_value(alpha)
    model.beta.set_value(beta)

    solve_attempts = [
        ("default", {}),
        (
            "numeric_focus",
            {
                "NumericFocus": 3,
                "ScaleFlag": 2,
                "BarHomogeneous": 1,
            },
        ),
        (
            "numeric_focus_no_aggregate",
            {
                "NumericFocus": 3,
                "ScaleFlag": 2,
                "BarHomogeneous": 1,
                "Aggregate": 0,
            },
        ),
        (
            "bar_qcp_tol_1e-5",
            {
                "NumericFocus": 3,
                "ScaleFlag": 2,
                "BarHomogeneous": 1,
                "BarQCPConvTol": 1e-5,
            },
        ),
        (
            "bar_qcp_tol_1e-4",
            {
                "NumericFocus": 3,
                "ScaleFlag": 2,
                "BarHomogeneous": 1,
                "BarQCPConvTol": 1e-4,
            },
        ),
    ]

    if not retry_on_error:
        solve_attempts = solve_attempts[:1]

    results = None
    status = None
    solver_status = None
    solve_attempt = None

    for attempt_idx, (attempt_name, attempt_options) in enumerate(solve_attempts):
        solver = SolverFactory("gurobi", solver_io="lp")

        solver.options["TimeLimit"] = time_limit
        solver.options["OptimalityTol"] = mip_gap
        solver.options["OutputFlag"] = 1

        for key, value in attempt_options.items():
            solver.options[key] = value

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            root, ext = os.path.splitext(log_file)
            attempt_log_file = (
                log_file
                if attempt_idx == 0
                else f"{root}_{attempt_name}{ext or '.log'}"
            )
            solver.options["LogFile"] = attempt_log_file

        results = solver.solve(model, tee=False, load_solutions=False)

        status = str(results.solver.termination_condition)
        solver_status = str(results.solver.status)
        solve_attempt = attempt_name

        if status in ("optimal", "locallyOptimal"):
            break

    if status not in ("optimal", "locallyOptimal"):
        return {
            "status": status,
            "solver_status": solver_status,
            "solve_attempt": solve_attempt,
            "alpha": alpha,
            "beta": beta,
            "P_flex_pcc": None,
            "Q_flex_pcc": None,
            "P_pcc": {},
            "Q_pcc": {},
            "P_flex": {},
            "Q_flex": {},
            "P_pv_flex": {},
            "Q_pv_flex": {},
            "P_hp_flex": {},
            "P_bat_flex": {},
            "Q_bat_flex": {},
            "obj_value": None,
        }

    model.solutions.load_from(results)

    P_pcc = {t: pyo.value(model.P_pcc[t]) for t in model.T}
    Q_pcc = {t: pyo.value(model.Q_pcc[t]) for t in model.T}

    P_flex = {
        (t, k): pyo.value(model.P_flex[t, k])
        for t in model.T
        for k in model.B_flex
    }

    Q_flex = {
        (t, k): pyo.value(model.Q_flex[t, k])
        for t in model.T
        for k in model.B_flex
    }

    def extract_indexed_var_2d(var_name: str):
        var = model.find_component(var_name)

        if var is None:
            return {}

        return {
            (t, k): pyo.value(var[t, k])
            for t in model.T
            for k in model.B_flex
        }

    P_pv_flex = extract_indexed_var_2d("P_pv_flex")
    Q_pv_flex = extract_indexed_var_2d("Q_pv_flex")
    P_hp_flex = extract_indexed_var_2d("P_hp_flex")
    P_bat_flex = extract_indexed_var_2d("P_bat_flex")
    Q_bat_flex = extract_indexed_var_2d("Q_bat_flex")

    P_flex_pcc = None
    Q_flex_pcc = None

    if model.find_component("P_flex_pcc") is not None:
        P_flex_pcc = pyo.value(model.P_flex_pcc)

    if model.find_component("Q_flex_pcc") is not None:
        Q_flex_pcc = pyo.value(model.Q_flex_pcc)

    return {
        "status": status,
        "solver_status": solver_status,
        "solve_attempt": solve_attempt,
        "alpha": alpha,
        "beta": beta,
        "P_flex_pcc": P_flex_pcc,
        "Q_flex_pcc": Q_flex_pcc,
        "P_pcc": P_pcc,
        "Q_pcc": Q_pcc,
        "P_flex": P_flex,
        "Q_flex": Q_flex,
        "P_pv_flex": P_pv_flex,
        "Q_pv_flex": Q_pv_flex,
        "P_hp_flex": P_hp_flex,
        "P_bat_flex": P_bat_flex,
        "Q_bat_flex": Q_bat_flex,
        "obj_value": pyo.value(model.obj),
    }
