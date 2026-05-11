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
    """
    
    """
    net = net_data['net']
    J_Pth  = net_data['J_Pth']
    J_PU   = net_data['J_PU']
    J_Qth  = net_data['J_Qth']
    J_QU   = net_data['J_QU']
    rate_A = net_data['rate_A']
    S_base = net_data['S_base']
    n      = net_data['n_buses']
    slack  = net_data['slack_idx']

    buses   = list(range(n))
    lines   = list(range(len(net.line)))

    
    bus_lookup = net_data['bus_lookup']

    def get_load_pu(t, bus_internal):
        pp_bus_idx = bus_internal
        bus_i = net.bus.index[pp_bus_idx]
        if bus_i in load_df.columns:
            if t in load_df.index:
                return float(load_df.loc[t, bus_i]) / S_base
        return 0.0

    
    pf_ratio_series = pd.Series(0.0, index=range(n))
    for _, load_row in net.load.iterrows():
        pp_idx = int(bus_lookup[load_row['bus']])
        p = load_row['p_mw']
        q = load_row['q_mvar']
        pf_ratio_series[pp_idx] = (q / p) if p != 0 else 0.0


    # ----------------------------------------------------------------
    model = pyo.ConcreteModel()

    # SETS
    model.T = pyo.Set(initialize=time_steps)
    model.B = pyo.Set(initialize=buses)
    model.L = pyo.Set(initialize=lines)

    # PARAMETERS
    model.V_min  = pyo.Param(initialize=float(net.bus['min_vm_pu'].iloc[0]))
    model.V_max  = pyo.Param(initialize=float(net.bus['max_vm_pu'].iloc[0]))
    model.slack  = pyo.Param(initialize=slack)

    model.alpha = pyo.Param(initialize=0.0, mutable=True)
    model.beta  = pyo.Param(initialize=0.0, mutable=True)

    model.pf_ratio = pyo.Param(
    model.B,
    initialize={i: float(pf_ratio_series[i]) for i in buses},
    )

    model.rate_A = pyo.Param(model.L,
                              initialize={l: float(rate_A[l]) for l in lines})

    load_data = {(t, b): get_load_pu(t, b)
                 for t in time_steps for b in buses}
    model.P_load = pyo.Param(model.T, model.B, initialize=load_data)

    model.P_pcc_base = pyo.Param(model.T,initialize={t: float(net_data['P_pcc_base']) for t in time_steps})
    model.Q_pcc_base = pyo.Param(model.T,initialize={t: float(net_data['Q_pcc_base']) for t in time_steps})

    # Jacobian
    model.J_Pth = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_Pth[i, k]) for i in buses for k in buses})
    model.J_PU  = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_PU[i, k])  for i in buses for k in buses})
    model.J_Qth = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_Qth[i, k]) for i in buses for k in buses})
    model.J_QU  = pyo.Param(model.B, model.B,
        initialize={(i, k): float(J_QU[i, k])  for i in buses for k in buses})

    # Branch from/to 
    model.br_f = pyo.Param(model.L, initialize={
        l: int(net._pd2ppc_lookups['bus'][net.line['from_bus'].iloc[l]])
        for l in lines})
    model.br_t = pyo.Param(model.L, initialize={
        l: int(net._pd2ppc_lookups['bus'][net.line['to_bus'].iloc[l]])
        for l in lines})
    model.br_r = pyo.Param(model.L, initialize={
        l: float(net.line['r_ohm_per_km'].iloc[l]
                 * net.line['length_km'].iloc[l]
                 / (net.bus['vn_kv'].iloc[0]**2 / S_base))
        for l in lines})
    model.br_x = pyo.Param(model.L, initialize={
        l: float(net.line['x_ohm_per_km'].iloc[l]
                 * net.line['length_km'].iloc[l]
                 / (net.bus['vn_kv'].iloc[0]**2 / S_base))
        for l in lines})

    # VARIABLES
    model.theta  = pyo.Var(model.T, model.B, initialize=0.0)
    model.dV     = pyo.Var(model.T, model.B, initialize=0.0)
    model.P_inj  = pyo.Var(model.T, model.B, initialize=0.0)
    model.Q_inj  = pyo.Var(model.T, model.B, initialize=0.0)
    model.P_pcc  = pyo.Var(model.T, initialize=0.0)
    model.Q_pcc  = pyo.Var(model.T, initialize=0.0)
    model.P_line = pyo.Var(model.T, model.L, initialize=0.0)
    model.Q_line = pyo.Var(model.T, model.L, initialize=0.0)

    # CONSTRAINTS

    # 1. Slack-Bus
    model.c_slack_theta = pyo.Constraint(
        model.T, rule=lambda m, t: m.theta[t, slack] == 0.0)
    model.c_slack_dV = pyo.Constraint(
        model.T, rule=lambda m, t: m.dV[t, slack] == 0.0)

    # 2. linear power flow
    model.c_pf_P = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.P_inj[t, i] == sum(
            m.J_Pth[i, k] * m.theta[t, k] + m.J_PU[i, k] * m.dV[t, k]
            for k in buses))

    model.c_pf_Q = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.Q_inj[t, i] == sum(
            m.J_Qth[i, k] * m.theta[t, k] + m.J_QU[i, k] * m.dV[t, k]
            for k in buses))

    # 3. Power balances
    model.c_bal_P = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.P_inj[t, i] == (m.P_pcc[t] if i == slack
                          else -m.P_load[t, i]))

    model.c_bal_Q = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        m.Q_inj[t, i] == (m.Q_pcc[t] if i == slack
                          else -m.P_load[t, i] * m.pf_ratio[i]))

    # 4. Voltages limits
    model.c_v_min = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        pyo.Constraint.Skip if i == slack
        else m.dV[t, i] >= model.V_min - 1.0)

    model.c_v_max = pyo.Constraint(model.T, model.B, rule=lambda m, t, i:
        pyo.Constraint.Skip if i == slack
        else m.dV[t, i] <= model.V_max - 1.0)

    # 5. line flows
    def line_flow_P(m, t, l):
        f, tb = m.br_f[l], m.br_t[l]
        r, x  = m.br_r[l], m.br_x[l]
        d = r**2 + x**2
        return m.P_line[t, l] == (
            (r/d) * (m.dV[t, f]  - m.dV[t, tb]) +
            (x/d) * (m.theta[t, f] - m.theta[t, tb]))

    def line_flow_Q(m, t, l):
        f, tb = m.br_f[l], m.br_t[l]
        r, x  = m.br_r[l], m.br_x[l]
        d = r**2 + x**2
        return m.Q_line[t, l] == (
            (x/d) * (m.dV[t, f]  - m.dV[t, tb]) -
            (r/d) * (m.theta[t, f] - m.theta[t, tb]))

    model.c_line_P = pyo.Constraint(model.T, model.L, rule=line_flow_P)
    model.c_line_Q = pyo.Constraint(model.T, model.L, rule=line_flow_Q)

    def line_lim_rule(m, t, l):
        return (m.P_line[t, l]**2 + m.Q_line[t, l]**2
                <= m.rate_A[l]**2)

    model.c_line_lim = pyo.Constraint(model.T, model.L, rule=line_lim_rule)

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
        return (m.P_flex[t, k]**2 + m.Q_flex[t, k]**2
                <= m.d_soc[k]**2)

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
    solver.options['OutputFlag']   = 0  

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
