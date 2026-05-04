import numpy as np
import pandas as pd
import pyomo as pyo
import plot_and_estimate_correlations as corr
import config
from scipy.optimize import linprog
import cvxpy as cp

def load_existing_aggregation_data(scenario_name):
    input_path_path = f"01-PROCESSED-DATA/aggregated_data/{scenario_name}_scenario/aggregation_data.csv"
    return pd.read_csv(input_path_path)


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
    full_load_df = pd.DataFrame(0.0, index=range(n_timesteps), columns=range(1, n_busses + 1))

    existing = mapping[mapping["existing_load"] == True]

    for _, row in existing.iterrows():
        bus_id = int(row["bus_i"])
        ts_id  = int(row["time_series_ID"])

        if ts_id in load_ts.columns:
            full_load_df[bus_id] = load_ts[ts_id].values

    return full_load_df



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
        [ 0, -1],   # Q >= 0  so  Q = 0
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
    prob = cp.Problem(
        cp.Maximize(cp.norm1(x)),        # ||x||_1 = |P| + |Q|
        [H_lin @ x <= h_lin]
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


def setup_OPF_problem():
    
    model = pyo.ConcreteModel()

    ## Define Variables

    ## Define Constraints

    ## Define Objective Function


    return model


def solve_OPF(model):
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(model)

    return results

def process_results(results):
    
    pass