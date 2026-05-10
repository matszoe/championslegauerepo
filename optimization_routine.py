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


def load_existing_aggregation_data(scenario_name):
    input_path_path = f"01-PROCESSED-DATA/aggregation-data/{scenario_name}_scenario/aggregation_data.csv"
    return pd.read_csv(input_path_path)

def save_aggregation_data(aggregation_df, scenario_name):
    output_path_path = f"01-PROCESSED-DATA/aggregation-data/{scenario_name}_scenario/aggregation_data.csv"
    aggregation_df.to_csv(output_path_path)

def map_load_time_series():
    mapping_file_path = "00-INPUT-DATA/norway_data/mapping_loads_to_CINELDI_MV_reference_grid.csv"
    load_file_path = "00-INPUT-DATA/norway_data/load_data_CINELDI_MV_reference_system.csv"
    
    mapping = pd.read_csv(mapping_file_path, sep=";")
    load_ts = pd.read_csv(load_file_path, index_col=0, sep=";")

    mapping["time_series_ID"] = mapping["time_series_ID"].astype(int)
    load_ts.columns = load_ts.columns.astype(int)
    load_ts = load_ts.reset_index(drop=True)  
    # a little bit hard coded here but avoids loading another csv
    n_timesteps = 8760
    n_busses = 124
    time_index = pd.date_range("2018-01-01", periods=8760, freq="h")
    full_load_df = pd.DataFrame(0.0, index=time_index, columns=range(1, n_busses + 1))

    existing = mapping[mapping["existing_load"] == True]

    for _, row in existing.iterrows():
        bus_id = int(row["bus_i"])
        ts_id  = int(row["time_series_ID"])

        if ts_id in load_ts.columns:
            full_load_df[bus_id] = load_ts[ts_id].values

    return full_load_df


def compute_hp_flexibility(
    correlation_df: pd.DataFrame,
    T_min_op: float = -8.0,    # Außentemperatur bei der WP auf Volllast läuft
    T_max_op: float = 15.0,    # Außentemperatur bei der WP abschaltet
) -> pd.DataFrame:
    """
    Computes the time-varying HP flexibility bounds for a single timestep
    using a simplified temperature-based scaling (Option B).

    P_hp_max(t) = cap_hp_mw * f(T)   where f(T) = clip((T_max_op - T) / (T_max_op - T_min_op), 0, 1)
    P_hp_min(t) = 0                   (HP can always be switched off)

    Parameters
    ----------
    correlation_df : DataFrame with index=bus_id, column 'cap_hp_mw'
    temperature    : current outside temperature [°C]
    T_min_op       : outside temperature at full HP load [°C]
    T_max_op       : outside temperature at zero HP load [°C]

    Returns
    -------
    DataFrame with index=bus_id, columns=['P_hp_max', 'P_hp_min']
    """

    ## load temperature of timestep
    temp_df = pd.read_csv("00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv", parse_dates=["date"], index_col="date")
    temperature = temp_df.loc[config.timestep_under_consideration, "temperature_2m"]

    # normalized temperature factor (0 at T_max_op, 1 at T_min_op)
    f_temp = np.clip(
        (T_max_op - temperature) / (T_max_op - T_min_op),
        0.0, 1.0
    )

    hp_flex = pd.DataFrame(index=correlation_df.index)
    hp_flex['P_hp_max'] = correlation_df['cap_hp_mw'] * f_temp
    hp_flex['P_hp_min'] = 0.0

    return hp_flex


def compute_nodal_approx_for_linear_constraints(
    P_pv_max: float,
    P_hp_max: float,
    P_hp_min: float,
    cos_phi_min: float = config.pv_cf_lower_limit,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gives the exact Minkowski sum for PV and HP as these only include linear constraints
    Returns
    -------
    H : (m x 2) Constraint-Matrix
    h : (m,)   right side for FOR_agg = {(P,Q) | H @ [P,Q] <= h}
    """
    tan_phi = np.sqrt(1 - cos_phi_min**2) / cos_phi_min

    # --- Constraint matrices ---

    # Common constraint matrix for P limits
    E = np.array([
        [ 1,  0],   # P <= P_max
        [-1,  0],   # P >= P_min or 0
    ])

    # Device-specific constraints
    # PV: operating limits 
    A_pv = np.array([
        [-tan_phi,  1],   # Q <= tan_phi * P
        [ tan_phi, -1],   # Q >= -tan_phi * P
    ])
    b1_pv = np.array([P_pv_max, 0.0])   # E constraints for PV
    b2_pv = np.array([0.0, 0.0])         # A constraints for PV

    # HP: Q=0 (cos_phi=1) maybe has to be changed at a later point
    A_hp = np.array([
        [ 0,  1],   # Q <= 0
        [ 0, -1],   # Q >= 0  so in total Q = 0
    ])
    b1_hp = np.array([P_hp_max, -P_hp_min])  # E constraints for HP
    b2_hp = np.array([0.0, 0.0])              # C constraints for HP

    # b_hat = max_{x in F_PV} A_hp @ x ---
    # (upper limit of HP constraints over PV space)
    H_pv = np.vstack([E, A_pv])
    rhs_pv = np.concatenate([b1_pv, b2_pv])

    b_hat = np.zeros(len(A_hp))
    for i in range(len(A_hp)):
        res = linprog(
            c=-A_hp[i],           # maximize A_hp[i] @ x
            A_ub=H_pv, b_ub=rhs_pv,
            bounds=[(None, None), (None, None)],
            method="highs"
        )
        b_hat[i] = -res.fun 

    #  d_hat = max_{x in F_HP} A_pv @ x ---
    # (upper limit of PV constraints over HP space)
    H_hp = np.vstack([E, A_hp])
    rhs_hp = np.concatenate([b1_hp, b2_hp])

    d_hat = np.zeros(len(A_pv))
    for i in range(len(A_pv)):
        res = linprog(
            c=-A_pv[i],
            A_ub=H_hp, b_ub=rhs_hp,
            bounds=[(None, None), (None, None)],
            method="highs"
        )
        d_hat[i] = -res.fun

    # --- Assemble outer approx ---
    # H = [E; A_pv; A_hp]
    # h = [b1_pv + b1_hp; b2_pv + d_hat; b_hat + b2_hp]
    H = np.vstack([E, A_pv, A_hp])
    h = np.concatenate([
        b1_pv + b1_hp,    
        b2_pv + d_hat,   
        b_hat + b2_hp,
    ])

    return H, h


def compute_nodal_approx_for_with_battery(
    P_pv_max: float,
    P_hp_max: float,
    P_hp_min: float,
    P_dis_max: float,
    P_chg_max: float,
    S_bat_max: float = config.battery_s_max,
    cos_phi_min: float = config.pv_cf_lower_limit,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extends the linear PV+HP outer approximation with a BESS SOC constraint
    following Barot & Taylor Lemma 2 + general case (Eq. 12 & 13).

    The aggregated FOR is described by:
      - Linear constraints : H_lin @ [P,Q] <= h_lin
      - SOC constraint     : ||[P,Q]|| <= d_soc

    Returns
    -------
    H_lin : (m x 2) linear constraint matrix
    h_lin : (m,)    linear constraint RHS
    d_soc : float   SOC radius, i.e. ||x|| <= d_soc
    """

    # Step 1: linear outer approx for PV + HP
    H_lin, h_lin = compute_nodal_approx_for_linear_constraints(
        P_pv_max, P_hp_max, P_hp_min, cos_phi_min
    )

    if S_bat_max <= 0:
        # no battery: return linear result, SOC radius = 0
        return H_lin, h_lin, 0.0

    # ------------------------------------------------------------------
    # Step 2: Eq. (12) — d_hat_i = -min_{x in BESS} c_i^T x - sum_j |A_i^j x|
    # For linear constraints: c_i = 0, A_i = i-th row of H_lin (scalar)
    # → d_hat_i = max_{x in BESS} |H_lin[i] @ x|
    # Note: H_lin[i] is a row vector (1x2), so H_lin[i] @ x is a scalar
    # → |H_lin[i] @ x| = max(H_lin[i] @ x, -H_lin[i] @ x)
    # which is simply maximizing a linear function over the BESS FOR
    # ------------------------------------------------------------------
    d_hat = np.zeros(len(H_lin))
    for i in range(len(H_lin)):
        x = cp.Variable(2)
        constraints = [
            cp.norm(x, 2) <= S_bat_max,
            x[0] <= P_dis_max,
            x[0] >= -P_chg_max,
        ]
        # max |H_lin[i] @ x| = max over both signs
        val_pos = cp.Problem(
            cp.Maximize( H_lin[i] @ x), constraints
        ).solve(solver=cp.CLARABEL)
        val_neg = cp.Problem(
            cp.Maximize(-H_lin[i] @ x), constraints
        ).solve(solver=cp.CLARABEL)
        d_hat[i] = max(val_pos, val_neg)

    # ------------------------------------------------------------------
    # Step 3: Eq. (13) — h_hat = -min_{x in PV+HP} g^T x - sum_j |E^j x|
    # For BESS: g_i = 0, E_i = I  → sum_j |I^j x| = |P| + |Q|
    # → h_hat = max_{x in PV+HP} (|P| + |Q|)
    # ------------------------------------------------------------------
    x = cp.Variable(2)
    t = cp.Variable(2)   # t[0] = |P|, t[1] = |Q|

    prob = cp.Problem(
        cp.Maximize(t[0] + t[1]),
        [
            H_lin @ x <= h_lin,   # PV+HP FOR constraints
            t[0] >=  x[0],        # t_P >= P
            t[0] >= -x[0],        # t_P >= -P
            t[1] >=  x[1],        # t_Q >= Q
            t[1] >= -x[1],        # t_Q >= -Q
        ]
    )
    prob.solve(solver=cp.CLARABEL)
    h_hat = prob.value if prob.status == "optimal" else 0.0

    # ------------------------------------------------------------------
    # Step 4: Assemble aggregated FOR
    # Linear constraints widened by d_hat (room for BESS flexibility)
    # SOC radius expanded by h_hat (absorbs PV+HP contribution)
    # ------------------------------------------------------------------
    H_lin_final = H_lin
    h_lin_final = h_lin + d_hat
    d_soc_final = S_bat_max + h_hat

    return H_lin_final, h_lin_final, d_soc_final



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
    solver.options['OutputFlag']   = 1   

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


def main():
    """
    Main function to run the optimization routine for a single scenario and time step.
    """
    print("Starting program...")
    if config.load_existing_aggregation:
        print(f"Loading existing aggregation data for scenario '{config.scenario_name}'")
        aggregation_df = load_existing_aggregation_data(config.scenario_name)

        print("Loading load time series for OPF model")
        load_df = map_load_time_series()
    else:
        print("Mapping load time series and estimating correlations to compute aggregation data")
        load_df = map_load_time_series()
        correlation_df = corr.estimate_correlations(load_df)
        if config.scenario == "with_battery":
            correlation_df = corr.add_battery_capacity(correlation_df)
        
        hp_flex = compute_hp_flexibility(correlation_df)
        pv_df = pd.read_csv("00-INPUT-DATA/PV-DATA/PV_timeseries.csv", parse_dates=["time"], index_col="time")
        if pv_df.index.tz is None:
            pv_df.index = pv_df.index.tz_localize('UTC')
        pv_cf = pv_df.loc[config.timestep_under_consideration, "electricity"]

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
                H, h, d_soc = compute_nodal_approx_for_with_battery(
                    P_pv_max, P_hp_max, P_hp_min, P_dis_max, P_chg_max
                )
            else:
                H, h = compute_nodal_approx_for_linear_constraints(
                    P_pv_max, P_hp_max, P_hp_min
                )
                d_soc = 0.0

            aggregation_df.loc[node] = {
                'H': H,
                'h': h,
                'd_soc': d_soc,
            }
        print("Saving aggregation data")
        save_aggregation_data(aggregation_df, config.scenario)

    ## Load network and set up OPF model
    print("Loading network data and setting up OPF")
    net_data = load_network_and_extract()
    base_model = setup_base_OPF(load_df,net_data, time_steps=[config.timestep_under_consideration])

    ## Add objective function
    print(f"Adding objective function with alpha={config.alpha}, beta={config.beta}")
    model_with_obj = add_objective_function(base_model, config.alpha, config.beta)

    ## changing index
    bus_lookup = net_data["net"]._pd2ppc_lookups['bus']
    aggregation_df.index = [int(bus_lookup[i]) for i in aggregation_df.index]

    ## Add flexibility constraints
    print("Adding aggregated flexibility constraints to OPF model")
    full_model = add_aggregated_flexibility_constraints(model_with_obj, aggregation_df)

    ## Solve OPF
    print("Solving OPF model with Gurobi")
    results = solve_OPF(full_model, config.alpha, config.beta)
    print(f"Solver status: {results}")



if __name__ == "__main__":
    main()
