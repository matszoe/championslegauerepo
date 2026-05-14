import numpy as np
import pandas as pd
import pyomo.environ as pyo
import plot_and_estimate_correlations as corr
import config
from scipy.optimize import linprog
import cvxpy as cp
import pandapower_read_csv as pp_read
import pandapower as pp
from pandapower.pypower.makeYbus import makeYbus
from tqdm import tqdm
from pyomo.opt import SolverFactory
import os
import ast

def load_network_and_extract() -> dict:
    """
    Loads the network and extracts the necessary data for the OPF problem:
    """
    net = pp_read.read_net_from_csv("00-INPUT-DATA/norway_data/")

    X_THRESHOLD = 1e-4
    Z_base = net.bus['vn_kv'].iloc[0]**2 / 10.0  # baseMVA=10

    for idx in net.line.index:
        r_pu = net.line.at[idx, 'r_ohm_per_km'] * net.line.at[idx, 'length_km'] / Z_base
        x_pu = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km'] / Z_base
        
        # WICHTIG: Die ABSOLUTE p.u. Impedanz limitieren und auf ohm_per_km hochrechnen
        if r_pu < X_THRESHOLD:
            net.line.at[idx, 'r_ohm_per_km'] = (X_THRESHOLD * Z_base) / net.line.at[idx, 'length_km']
        if x_pu < X_THRESHOLD:
            net.line.at[idx, 'x_ohm_per_km'] = (X_THRESHOLD * Z_base) / net.line.at[idx, 'length_km']

    pp.runpp(net, algorithm='nr', numba=False)

    ppc = net._ppc

    S_base = ppc['baseMVA']
    Ybus, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
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

    V_base = net.bus['vn_kv'].iloc[0]
    rate_A = (net.line['max_i_ka'].values
              * V_base * np.sqrt(3)) / S_base

    P_pcc_base = net.res_ext_grid['p_mw'].values[0]   / S_base
    Q_pcc_base = net.res_ext_grid['q_mvar'].values[0] / S_base

    bus_lookup = net._pd2ppc_lookups['bus']


    slack_pp_bus = int(net.ext_grid['bus'].iloc[0])
    slack_ppc_idx = int(bus_lookup[slack_pp_bus])

    return {
        'J_Pth':       J_Pth,
        'J_PU':        J_PU,
        'J_Qth':       J_Qth,
        'J_QU':        J_QU,
        'rate_A':      rate_A,
        'P_pcc_base':  P_pcc_base,
        'Q_pcc_base':  Q_pcc_base,
        'S_base':      S_base,
        'bus_lookup':  bus_lookup,
        'n_buses':     n,
        'slack_idx':   slack_ppc_idx,
        'ppc':         ppc,
        'net':         net,
    }


def setup_base_OPF(load_df: pd.DataFrame,
                   net_data: dict,
                   time_steps: list[int]) -> pyo.ConcreteModel:

    JACOBIAN_THRESHOLD = 1e-6

    net    = net_data['net']
    J_Pth  = net_data['J_Pth'].copy()
    J_PU   = net_data['J_PU'].copy()
    J_Qth  = net_data['J_Qth'].copy()
    J_QU   = net_data['J_QU'].copy()

    J_Pth[np.abs(J_Pth) < JACOBIAN_THRESHOLD] = 0.0
    J_PU [np.abs(J_PU)  < JACOBIAN_THRESHOLD] = 0.0
    J_Qth[np.abs(J_Qth) < JACOBIAN_THRESHOLD] = 0.0
    J_QU [np.abs(J_QU)  < JACOBIAN_THRESHOLD] = 0.0

    rate_A = net_data['rate_A']
    S_base = net_data['S_base']
    n      = net_data['n_buses']
    slack  = net_data['slack_idx']
    bus_lookup = net_data['bus_lookup']

    buses = list(range(n))
    lines = list(range(len(net.line)))

    # --- Leitungen aufteilen ---
    X_THRESHOLD = 1e-4
    Z_base = net.bus['vn_kv'].iloc[0]**2 / S_base

    normal_lines = []
    ideal_lines  = []
    for l in lines:
        r_pu = float(net.line['r_ohm_per_km'].iloc[l]
                     * net.line['length_km'].iloc[l] / Z_base)
        x_pu = float(net.line['x_ohm_per_km'].iloc[l]
                     * net.line['length_km'].iloc[l] / Z_base)
        if x_pu < X_THRESHOLD and r_pu < X_THRESHOLD:
            ideal_lines.append(l)
        else:
            normal_lines.append(l)

    print(f"Normal lines: {len(normal_lines)}, Ideal lines: {len(ideal_lines)}")

    # --- Load und pf_ratio ---
    def get_load_pu(t, bus_internal):
        bus_i = net.bus.index[bus_internal]
        if bus_i in load_df.columns and t in load_df.index:
            return float(load_df.loc[t, bus_i]) / S_base
        return 0.0

    pf_ratio_series = pd.Series(0.0, index=range(n))
    for _, load_row in net.load.iterrows():
        pp_idx = int(bus_lookup[load_row['bus']])
        p = load_row['p_mw']
        q = load_row['q_mvar']
        pf_ratio_series[pp_idx] = (q / p) if p != 0 else 0.0

    # --- Modell ---
    model = pyo.ConcreteModel()

    # SETS
    model.T        = pyo.Set(initialize=time_steps)
    model.B        = pyo.Set(initialize=buses)
    model.L        = pyo.Set(initialize=lines)
    model.L_normal = pyo.Set(initialize=normal_lines)
    model.L_ideal  = pyo.Set(initialize=ideal_lines)

    # PARAMETERS
    model.V_min  = pyo.Param(initialize=float(net.bus['min_vm_pu'].iloc[0]))
    model.V_max  = pyo.Param(initialize=float(net.bus['max_vm_pu'].iloc[0]))
    model.slack  = pyo.Param(initialize=slack)
    model.alpha  = pyo.Param(initialize=0.0, mutable=True)
    model.beta   = pyo.Param(initialize=0.0, mutable=True)

    model.pf_ratio = pyo.Param(
        model.B,
        initialize={i: float(pf_ratio_series[i]) for i in buses})

    model.rate_A = pyo.Param(
        model.L,
        initialize={l: float(rate_A[l]) for l in lines})

    model.P_load = pyo.Param(
        model.T, model.B,
        initialize={(t, b): get_load_pu(t, b)
                    for t in time_steps for b in buses})

    model.P_pcc_base = pyo.Param(
        model.T,
        initialize={t: float(net_data['P_pcc_base']) for t in time_steps})
    model.Q_pcc_base = pyo.Param(
        model.T,
        initialize={t: float(net_data['Q_pcc_base']) for t in time_steps})

    model.J_Pth = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_Pth[i, k]) for i in buses for k in buses})
    model.J_PU  = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_PU[i, k])  for i in buses for k in buses})
    model.J_Qth = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_Qth[i, k]) for i in buses for k in buses})
    model.J_QU  = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_QU[i, k])  for i in buses for k in buses})

    model.br_f = pyo.Param(model.L, initialize={
        l: int(net._pd2ppc_lookups['bus'][net.line['from_bus'].iloc[l]])
        for l in lines})
    model.br_t = pyo.Param(model.L, initialize={
        l: int(net._pd2ppc_lookups['bus'][net.line['to_bus'].iloc[l]])
        for l in lines})
    model.br_r = pyo.Param(model.L, initialize={
        l: float(net.line['r_ohm_per_km'].iloc[l]
                 * net.line['length_km'].iloc[l] / Z_base)
        for l in lines})
    model.br_x = pyo.Param(model.L, initialize={
        l: float(net.line['x_ohm_per_km'].iloc[l]
                 * net.line['length_km'].iloc[l] / Z_base)
        for l in lines})

    # VARIABLES
    model.theta = pyo.Var(model.T, model.B,
                          initialize=0.0, bounds=(-500, 500))
    model.dV    = pyo.Var(model.T, model.B,
                          initialize=0.0, bounds=(-100, 100))
    SCALE_FACTOR = 1000

    P_max = float(load_df.max().max()) / S_base * 2
    model.P_inj = pyo.Var(model.T, model.B,
                           initialize=0.0, bounds=(-P_max, P_max))
    model.Q_inj = pyo.Var(model.T, model.B,
                           initialize=0.0, bounds=(-P_max, P_max))
    model.P_pcc = pyo.Var(model.T,
                           initialize=0.0, bounds=(-P_max * n, P_max * n))
    model.Q_pcc = pyo.Var(model.T,
                           initialize=0.0, bounds=(-P_max * n, P_max * n))

    # P_line/Q_line nur für normale Leitungen
    model.P_line = pyo.Var(model.T, model.L_normal,
                            initialize=0.0,
                            bounds=lambda m, t, l: (-m.rate_A[l], m.rate_A[l]))
    model.Q_line = pyo.Var(model.T, model.L_normal,
                            initialize=0.0,
                            bounds=lambda m, t, l: (-m.rate_A[l], m.rate_A[l]))

    # CONSTRAINTS
    nonzero_Pth = {(i,k) for i in buses for k in buses if abs(J_Pth[i,k]) > JACOBIAN_THRESHOLD}
    nonzero_PU  = {(i,k) for i in buses for k in buses if abs(J_PU[i,k])  > JACOBIAN_THRESHOLD}
    nonzero_Qth = {(i,k) for i in buses for k in buses if abs(J_Qth[i,k]) > JACOBIAN_THRESHOLD}
    nonzero_QU  = {(i,k) for i in buses for k in buses if abs(J_QU[i,k])  > JACOBIAN_THRESHOLD}

    # 1. Slack
    model.c_slack_theta = pyo.Constraint(
        model.T, rule=lambda m, t: m.theta[t, slack] == 0.0)
    model.c_slack_dV = pyo.Constraint(
        model.T, rule=lambda m, t: m.dV[t, slack] == 0.0)

    # 2. Linearisierter Lastfluss
    model.c_pf_P = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.P_inj[t, i] == sum(
            m.J_Pth[i, k] * m.theta[t, k] / SCALE_FACTOR + m.J_PU[i, k] * m.dV[t, k] / SCALE_FACTOR
            for k in buses
            if (i, k) in nonzero_Pth or (i, k) in nonzero_PU))

    model.c_pf_Q = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.Q_inj[t, i] == sum(
            m.J_Qth[i, k] * m.theta[t, k] / SCALE_FACTOR + m.J_QU[i, k] * m.dV[t, k] / SCALE_FACTOR
            for k in buses
            if (i, k) in nonzero_Qth or (i, k) in nonzero_QU))

    # 3. Knotenleistungsbilanz
    model.c_bal_P = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.P_inj[t, i] == (m.P_pcc[t] if i == slack else -m.P_load[t, i]))

    model.c_bal_Q = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.Q_inj[t, i] == (m.Q_pcc[t] if i == slack
                           else -m.P_load[t, i] * m.pf_ratio[i]))

    # 4. Spannungsgrenzen
    model.c_v_min = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        pyo.Constraint.Skip if i == slack
        else m.dV[t, i] >= model.V_min - 1.0)

    model.c_v_max = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        pyo.Constraint.Skip if i == slack
        else m.dV[t, i] <= model.V_max - 1.0)

    # 5. Leitungsflussgleichungen — nur normale Leitungen
    def line_flow_P(m, t, l):
        f, tb = m.br_f[l], m.br_t[l]
        r, x  = m.br_r[l], m.br_x[l]
        d = r**2 + x**2 
        return m.P_line[t, l] == (
            (r/d) * (m.dV[t, f]    - m.dV[t, tb]) / SCALE_FACTOR +
            (x/d) * (m.theta[t, f] - m.theta[t, tb]) / SCALE_FACTOR)

    def line_flow_Q(m, t, l):
        f, tb = m.br_f[l], m.br_t[l]
        r, x  = m.br_r[l], m.br_x[l]
        d = r**2 + x**2 
        return m.Q_line[t, l] == (
            (x/d) * (m.dV[t, f]    - m.dV[t, tb]) / SCALE_FACTOR -
            (r/d) * (m.theta[t, f] - m.theta[t, tb]))

    model.c_line_P = pyo.Constraint(model.T, model.L_normal, rule=line_flow_P)
    model.c_line_Q = pyo.Constraint(model.T, model.L_normal, rule=line_flow_Q)

    # 6. Ideale Leitungen — Knoten elektrisch identisch
    model.c_ideal_theta = pyo.Constraint(model.T, model.L_ideal,
        rule=lambda m, t, l: m.theta[t, m.br_f[l]] == m.theta[t, m.br_t[l]])
    model.c_ideal_dV = pyo.Constraint(model.T, model.L_ideal,
        rule=lambda m, t, l: m.dV[t, m.br_f[l]] == m.dV[t, m.br_t[l]])

    # 7. Kapazitätsgrenzen — nur normale Leitungen
    model.c_line_lim = pyo.Constraint(model.T, model.L_normal,
        rule=lambda m, t, l: ((100*m.P_line[t, l])**2 + (100*m.Q_line[t, l])**2
                               <= (100*m.rate_A[l])**2))

    return model

def add_objective_function(
    model: pyo.ConcreteModel,
    alpha: float,
    beta: float,
) -> pyo.ConcreteModel:
    """
    Adds the objective function to the OPF model following Eq. (5a)
    of the Walenstadt paper:

        min_{P_pcc, Q_pcc} sum_t [ alpha * (P_pcc_t - P_pcc_base_t)
                                  + beta  * (Q_pcc_t - Q_pcc_base_t) ]

    Parameters
    ----------
    model       : Pyomo model from setup_OPF()
    alpha       : weight for active power deviation (defines PQ direction)
    beta        : weight for reactive power deviation

    Returns
    -------
    model with objective function added
    """
    model.alpha.set_value(alpha)
    model.beta.set_value(beta)

    if model.find_component('obj'):
        model.del_component('obj')

    # objective function
    def obj_rule(m):
        return sum(
            m.alpha * (m.P_pcc[t] - m.P_pcc_base[t]) +
            m.beta  * (m.Q_pcc[t] - m.Q_pcc_base[t])
            for t in m.T
        )

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

def add_aggregated_flexibility_constraints(
    model: pyo.ConcreteModel,
    approximation_df: pd.DataFrame,
) -> pyo.ConcreteModel:
    """
    Adds the aggregated nodal flexibility constraints to the OPF model.

    For each node k, two types of constraints are added:
      1. Linear:  H_k @ [P_k, Q_k] <= h_k   (from Barot & Taylor linear approx)
      2. SOC:     ||[P_k, Q_k]|| <= d_soc_k  (only if d_soc_k > 0)

    Parameters
    ----------
    model           : Pyomo model from setup_OPF()
    approximation_df: DataFrame with index=bus_id, columns=['H', 'h', 'd_soc']
                      H     : (m x 2) np.ndarray — linear constraint matrix
                      h     : (m,)   np.ndarray  — linear constraint RHS
                      d_soc : float              — SOC radius (0 if no battery)

    Returns
    -------
    model with flexibility constraints added
    """

    # Remove existing flexibility constraints if present (allows re-calling)
    if model.find_component('c_flex_linear'):
        model.del_component('c_flex_linear')
    if model.find_component('c_flex_soc'):
        model.del_component('c_flex_soc')
    if model.find_component('P_flex'):
        model.del_component('P_flex')
        model.del_component('Q_flex')

    # --- Flexibility variables: net injection deviation per node ---
    # P_flex[t, bus] = flexible active power injection at node k [p.u.]
    # Q_flex[t, bus] = flexible reactive power injection at node k [p.u.]
    bus_ids = approximation_df.index.tolist()
    model.B_flex = pyo.Set(initialize=bus_ids)
    model.P_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)
    model.Q_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)

    # --- Precompute constraint data as plain dicts for Pyomo ---
    # Linear constraints: for node k, row i → H[i,0]*P + H[i,1]*Q <= h[i]
    # Index: (bus_id, row_index)
    linear_index = []
    H_0 = {}   # H[i, 0]: P coefficient
    H_1 = {}   # H[i, 1]: Q coefficient
    h_rhs = {} # h[i]: RHS

    for bus_id, row in approximation_df.iterrows():
        H = np.array(row['H'])   # (m x 2)
        h = np.array(row['h'])   # (m,)
        for i in range(len(h)):
            key = (bus_id, i)
            linear_index.append(key)
            H_0[key]   = float(H[i, 0])
            H_1[key]   = float(H[i, 1])
            h_rhs[key] = float(h[i])

    model.FLEX_LIN_IDX = pyo.Set(initialize=linear_index, dimen=2)
    model.H_0   = pyo.Param(model.FLEX_LIN_IDX, initialize=H_0)
    model.H_1   = pyo.Param(model.FLEX_LIN_IDX, initialize=H_1)
    model.h_rhs = pyo.Param(model.FLEX_LIN_IDX, initialize=h_rhs)

    # SOC constraints: only for nodes with d_soc > 0
    soc_buses  = approximation_df[approximation_df['d_soc'] > 0].index.tolist()
    model.B_soc  = pyo.Set(initialize=soc_buses)
    model.d_soc  = pyo.Param(model.B_soc,
                              initialize=approximation_df.loc[soc_buses, 'd_soc'].to_dict())

    # --- Linear flexibility constraints ---
    # H[i,0] * P_flex[t,k] + H[i,1] * Q_flex[t,k] <= h[i]
    def flex_linear_rule(m, t, bus_id, i):
        return (m.H_0[bus_id, i] * m.P_flex[t, bus_id] +
                m.H_1[bus_id, i] * m.Q_flex[t, bus_id]
                <= m.h_rhs[bus_id, i])

    model.c_flex_linear = pyo.Constraint(
        model.T, model.FLEX_LIN_IDX,
        rule=flex_linear_rule
    )

    # --- SOC flexibility constraints ---
    # Pyomo doesn't support SOC natively → linearise as 8-sided polygon
    # ||[P, Q]|| <= d_soc  approximated by:
    # |P| <= d,  |Q| <= d,  |P±Q| <= sqrt(2)*d

    def soc_rule(m, t, k):
        return  (m.P_flex[t, k]**2 + m.Q_flex[t, k]**2
            <=   m.d_soc[k]**2)

    model.c_flex_soc = pyo.Constraint(model.T, model.B_soc, rule=soc_rule)

    # --- Update nodal power balance to include flexibility ---
    # P_inj[t, k] = -P_load[t, k] + P_flex[t, k]  for non-slack buses
    # This replaces the balance constraint from setup_OPF()
    if model.find_component('c_bal_P'):
        model.del_component('c_bal_P')
    if model.find_component('c_bal_Q'):
        model.del_component('c_bal_Q')

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


def setup_multi_timestep_OPF(
    load_df: pd.DataFrame,
    net_data: dict,
    time_steps: list,
    aggregation_per_timestep: dict,   # {Timestamp: aggregation_df}
    hp_flex_df: pd.DataFrame,         # index=bus_id, columns=Timestamps
    temperature_series: pd.Series,
    delta_t: float = 1.0,
    T_preferred: float = config.T_room_initial,
    T_min: float = 18.0,
    T_max: float = 22.0,
) -> pyo.ConcreteModel:

    time_list = list(time_steps)

    # --- Schritt 1: Basismodell aufbauen (identisch zu Single-Timestep) ---
    model = setup_base_OPF(load_df, net_data, time_steps=time_list)

    # --- Schritt 2: Objective unverändert wiederverwenden ---
    # add_objective_function bleibt identisch — summiert über alle t in model.T
    # Das ist korrekt: sum_t [alpha*(P_pcc[t]-P_base[t]) + beta*(Q_pcc[t]-Q_base[t])]
    # wird durch c_const_flex unten auf einen einzigen Wert gezwungen
    model = add_objective_function(model, config.alpha, config.beta)

    # --- Schritt 3: Zeitabhängige Flexibility-Constraints ---
    # aggregation_per_timestep hat pro Zeitschritt andere H, h, d_soc
    # → wir erweitern die Indexierung um t

    bus_ids = aggregation_per_timestep[time_list[0]].index.tolist()
    model.B_flex = pyo.Set(initialize=bus_ids)
    model.P_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)
    model.Q_flex = pyo.Var(model.T, model.B_flex, initialize=0.0)

    # H, h zeitabhängig: Index (t, bus_id, row_i)
    linear_index = []
    H_0, H_1, h_rhs = {}, {}, {}

    for t in time_list:
        agg_t = aggregation_per_timestep[t]
        for bus_id, row in agg_t.iterrows():
            H = np.array(row['H'])
            h = np.array(row['h'])
            for i in range(len(h)):
                key = (t, bus_id, i)
                linear_index.append(key)
                H_0[key]   = float(H[i, 0])
                H_1[key]   = float(H[i, 1])
                h_rhs[key] = float(h[i])

    model.FLEX_LIN_IDX = pyo.Set(initialize=linear_index, dimen=3)
    model.H_0   = pyo.Param(model.FLEX_LIN_IDX, initialize=H_0)
    model.H_1   = pyo.Param(model.FLEX_LIN_IDX, initialize=H_1)
    model.h_rhs = pyo.Param(model.FLEX_LIN_IDX, initialize=h_rhs)

    def flex_linear_rule(m, t, bus_id, i):
        return (m.H_0[t, bus_id, i] * m.P_flex[t, bus_id] +
                m.H_1[t, bus_id, i] * m.Q_flex[t, bus_id]
                <= m.h_rhs[t, bus_id, i])

    model.c_flex_linear = pyo.Constraint(
        model.FLEX_LIN_IDX, rule=flex_linear_rule)

    # SOC-Constraints zeitabhängig (d_soc kann pro t variieren)
    soc_index = []
    d_soc_data = {}
    for t in time_list:
        agg_t = aggregation_per_timestep[t]
        for bus_id in agg_t[agg_t['d_soc'] > 0].index:
            soc_index.append((t, bus_id))
            d_soc_data[(t, bus_id)] = float(agg_t.loc[bus_id, 'd_soc'])

    model.SOC_IDX = pyo.Set(initialize=soc_index, dimen=2)
    model.d_soc   = pyo.Param(model.SOC_IDX, initialize=d_soc_data)

    def soc_rule(m, t, k):
        return (m.P_flex[t, k]**2 + m.Q_flex[t, k]**2
                <= m.d_soc[t, k]**2)

    model.c_flex_soc = pyo.Constraint(model.SOC_IDX, rule=soc_rule)

    # Knotenleistungsbilanz mit Flexibilität (identisch zu add_aggregated_flexibility_constraints)
    if model.find_component('c_bal_P'):
        model.del_component('c_bal_P')
    if model.find_component('c_bal_Q'):
        model.del_component('c_bal_Q')

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

    # --- Schritt 4: Eq. 6f/6g — konstante Flexibilität am PCC ---
    # Skalare Variablen für die konstante PCC-Abweichung
    model.P_flex_pcc = pyo.Var(initialize=0.0,bounds=(-2.0, 2.0))
    model.Q_flex_pcc = pyo.Var(initialize=0.0,bounds=(-2.0, 2.0))

    def const_P_flex_rule(m, t):
        return m.P_pcc[t] - m.P_pcc_base[t] == m.P_flex_pcc

    def const_Q_flex_rule(m, t):
        return m.Q_pcc[t] - m.Q_pcc_base[t] == m.Q_flex_pcc

    model.c_const_P_flex = pyo.Constraint(model.T, rule=const_P_flex_rule)
    model.c_const_Q_flex = pyo.Constraint(model.T, rule=const_Q_flex_rule)

    # --- Schritt 5: HP Thermalmodell (Eq. 2 / 6b) ---
    q_heat_profile = pd.read_csv(
        "00-INPUT-DATA/HP-DATA/hp_profile.csv",
        index_col=0
    )
    q_heat_profile.index = q_heat_profile.index.astype(int)
    q_heat_profile.columns = q_heat_profile.columns.astype(int)

    hp_buses = [b for b in bus_ids
                if hp_flex_df.loc[b].max() > 0]
    model.B_hp = pyo.Set(initialize=hp_buses)

    # T_room als Zustandsvariable
    model.T_room = pyo.Var(
        model.T, model.B_hp,
        initialize=T_preferred,
        bounds=(T_min, T_max)
    )

    # q_heat und P_hp_base als Parameter
    q_heat_data, p_hp_base_data = {}, {}
    for t in time_list:
        hour = t.hour
        temp  = float(temperature_series.loc[t,"temperature_2m"])
        temp_class = int(np.clip(round(temp), -14, 18))
        q_val = float(q_heat_profile.loc[hour, temp_class])
        for b in hp_buses:
            q_heat_data[(t, b)]   = q_val
            val = hp_flex_df.loc[b, t] if b in hp_flex_df.index else 0.0
            p_hp_base_data[(t, b)] = float(val) if not pd.isna(val) else 0.0

    model.q_heat    = pyo.Param(model.T, model.B_hp, initialize=q_heat_data)
    model.P_hp_base = pyo.Param(model.T, model.B_hp, initialize=p_hp_base_data)

    # Thermale Evolution: T_room[t+1] = T_room[t] + (P_flex[t]/P_base[t]) * q_heat[t] * dt
    model.T_IDX = pyo.Set(initialize=range(len(time_list)))

    def thermal_evolution_rule(m, t_idx, b):
        if t_idx == 0:
            return pyo.Constraint.Skip
        t_curr = time_list[t_idx]
        t_prev = time_list[t_idx - 1]
        p_base = p_hp_base_data.get((t_prev, b), 0.0)
        if p_base < 1e-3:
            return m.T_room[t_curr, b] == m.T_room[t_prev, b]
        return (p_base * m.T_room[t_curr, b] ==
            p_base * m.T_room[t_prev, b] +
            m.P_flex[t_prev, b] * m.q_heat[t_prev, b] * delta_t)

    model.c_thermal = pyo.Constraint(
        model.T_IDX, model.B_hp, rule=thermal_evolution_rule)

    # Startraumtemperatur fixieren
    for b in hp_buses:
        model.T_room[time_list[0], b].fix(T_preferred)

    # --- Schritt 6: BESS SOC-Dynamik ---
    soc_buses = list({k for (_, k) in soc_index})
    if soc_buses:
        model.B_soc_dyn = pyo.Set(initialize=soc_buses)
        model.SOC = pyo.Var(
            model.T, model.B_soc_dyn,
            initialize=config.SOC_inital,
            bounds=(config.soc_min, config.soc_max)
        )

        def soc_evolution_rule(m, t_idx, b):
            if t_idx == 0:
                return pyo.Constraint.Skip
            t_curr = time_list[t_idx]
            t_prev = time_list[t_idx - 1]
            return (m.SOC[t_curr, b] ==
                    m.SOC[t_prev, b] -
                    m.P_flex[t_prev, b] * delta_t / config.battery_capacity_at_FC)

        model.c_soc_evolution = pyo.Constraint(
            model.T_IDX, model.B_soc_dyn, rule=soc_evolution_rule)

        for b in soc_buses:
            model.SOC[time_list[0], b].fix(config.SOC_inital)

    return model




def solve_OPF(
    model: pyo.ConcreteModel,
    alpha: float,
    beta: float,
    time_limit: int = 300,
    mip_gap: float = 1e-4,
) -> dict:
    """
    Solves the OPF model using Gurobi for a given PQ direction (alpha, beta).

    Parameters
    ----------
    model      : Pyomo model with all constraints and objective set up
    alpha      : active power weight for current FFOR direction
    beta       : reactive power weight for current FFOR direction
    time_limit : Gurobi time limit in seconds
    mip_gap    : Gurobi optimality gap

    Returns
    -------
    dict with:
        'status'   : solver status string
        'P_pcc'    : {t: float} optimal active power at PCC [p.u.]
        'Q_pcc'    : {t: float} optimal reactive power at PCC [p.u.]
        'P_flex'   : {(t, bus): float} optimal flexible active injections
        'Q_flex'   : {(t, bus): float} optimal flexible reactive injections
        'obj_value': float objective value
    """

    # Update direction (mutable params → no model rebuild needed)
    model.alpha.set_value(alpha)
    model.beta.set_value(beta)

    # Solve with Gurobi via Pyomo SolverFactory
    solver = SolverFactory('gurobi', solver_io='python') ## can also try solver_io='lp' to boost speed 

    solver.options['TimeLimit']    = time_limit
    solver.options['OptimalityTol'] = mip_gap
    solver.options['NonConvex']    = 2   # needed for quadratic constraints (SOCP)
    solver.options['OutputFlag']   = 1
    
    solver.options["ScaleFlag"] = 2
    solver.options["NumericFocus"] = 3
    solver.options["BarHomogeneous"] = 1
    solver.options['Method'] = 2
    solver.options["BarConvTol"] = 1e-5
    solver.options["Aggregate"] = 0


    #model.write("debug.lp")
    results = solver.solve(model, tee=False)


    # Check solver status
    status = str(results.solver.termination_condition)
    if status not in ('optimal', 'locallyOptimal'):
        return {
            'status':    status,
            'P_pcc':     {},
            'Q_pcc':     {},
            'P_flex':    {},
            'Q_flex':    {},
            'obj_value': None,
        }

    # Extract results
    P_pcc   = {t: pyo.value(model.P_pcc[t])   for t in model.T}
    Q_pcc   = {t: pyo.value(model.Q_pcc[t])   for t in model.T}
    P_flex  = {(t, k): pyo.value(model.P_flex[t, k])
               for t in model.T for k in model.B_flex}
    Q_flex  = {(t, k): pyo.value(model.Q_flex[t, k])
               for t in model.T for k in model.B_flex}

    return {
        'status':    status,
        'P_pcc':     P_pcc,
        'Q_pcc':     Q_pcc,
        'P_flex':    P_flex,
        'Q_flex':    Q_flex,
        'obj_value': pyo.value(model.obj),
    }
