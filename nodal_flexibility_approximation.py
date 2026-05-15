import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import linprog
import cvxpy as cp

from pandapower.pypower.makeYbus import makeYbus
from tqdm import tqdm
from pyomo.opt import SolverFactory
import os
import config



def get_q_heat(profile: pd.DataFrame, hour: int, temperature: float) -> float:
    """
    Returns q_heat [K/h] for a given hour and outside temperature.
    Temperature is clipped to [-14, 18] and rounded to nearest int.
    """
    temp_class = int(np.clip(round(temperature), -14, 18))
    return float(profile.loc[hour, temp_class])



def compute_hp_baseline(
    correlation_df: pd.DataFrame,
    temperature: float,
    T_min_op: float = -8.0, 
    T_max_op: float = 15.0,
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

    # normalized temperature factor (0 at T_max_op, 1 at T_min_op)
    f_temp = np.clip(
        (T_max_op - temperature) / (T_max_op - T_min_op),
        0.0, 1.0
    )

    hp_flex = pd.DataFrame(index=correlation_df.index)
    hp_flex['P_hp_max'] = correlation_df['cap_hp_mw'] * f_temp
    hp_flex['P_hp_min'] = 0.0

    return hp_flex


def _max_linear_over_polytope(
    c: np.ndarray,
    H: np.ndarray,
    h: np.ndarray,
    name: str = "LP",
) -> float:
    """
    Solves max c^T x subject to Hx <= h.
    Returns the optimal value.
    """
    c = np.asarray(c, dtype=float)
    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float)

    res = linprog(
        c=-c,
        A_ub=H,
        b_ub=h,
        bounds=[(None, None)] * len(c),
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"{name} failed: {res.message}")

    return float(-res.fun)


def compute_nodal_approx_for_linear_constraints(
    P_pv_max: float,
    P_hp_max: float,
    P_hp_min: float,
    cos_phi_min: float = config.pv_cf_lower_limit,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Outer approximation of the Minkowski sum of PV and HP flexibility sets.

    Sign convention:
        P_flex > 0 means increased net injection at the bus,
        i.e. generation increase or load reduction.

    PV flexibility:
        0 <= P_pv <= P_pv_max
        -tan(phi) P_pv <= Q_pv <= tan(phi) P_pv

    HP flexibility:
        P_hp_min <= P_hp <= P_hp_max
        Q_hp = 0

    Returns:
        H, h such that {z | H z <= h} outer-approximates
        F_pv ⊕ F_hp, with z = [P, Q].
    """

    P_pv_max = max(float(P_pv_max), 0.0)
    P_hp_max = float(P_hp_max)
    P_hp_min = float(P_hp_min)

    if P_hp_min > P_hp_max:
        raise ValueError(
            f"Invalid HP bounds: P_hp_min={P_hp_min}, P_hp_max={P_hp_max}"
        )

    if not (0.0 < cos_phi_min <= 1.0):
        raise ValueError(f"Invalid cos_phi_min={cos_phi_min}")

    tan_phi = np.sqrt(max(0.0, 1.0 - cos_phi_min**2)) / cos_phi_min

    has_pv = P_pv_max > 1e-9
    has_hp = (P_hp_max - P_hp_min) > 1e-9 or abs(P_hp_max) > 1e-9 or abs(P_hp_min) > 1e-9

    if not has_pv and not has_hp:
        H = np.array([
            [ 1.0,  0.0],
            [-1.0,  0.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0],
        ])
        h = np.zeros(4)
        return H, h

    # Common P-limit directions
    E = np.array([
        [ 1.0,  0.0],
        [-1.0,  0.0],
    ])

    # PV set
    # 0 <= P_pv <= P_pv_max
    # Q_pv <= tan_phi * P_pv
    # -Q_pv <= tan_phi * P_pv
    A_pv = np.array([
        [-tan_phi,  1.0],
        [-tan_phi, -1.0],
    ])

    b1_pv = np.array([P_pv_max, 0.0])
    b2_pv = np.array([0.0, 0.0])

    H_pv = np.vstack([E, A_pv])
    rhs_pv = np.concatenate([b1_pv, b2_pv])

    # HP set
    # P_hp_min <= P_hp <= P_hp_max
    # Q_hp = 0
    A_hp = np.array([
        [0.0,  1.0],
        [0.0, -1.0],
    ])

    b1_hp = np.array([P_hp_max, -P_hp_min])
    b2_hp = np.array([0.0, 0.0])

    H_hp = np.vstack([E, A_hp])
    rhs_hp = np.concatenate([b1_hp, b2_hp])

    # Missing HP-specific constraints evaluated over PV set:
    # b_hat_i = max_{x in PV} A_hp_i x
    b_hat = np.array([
        _max_linear_over_polytope(A_hp[i], H_pv, rhs_pv, name=f"PV_for_HP_row_{i}")
        for i in range(len(A_hp))
    ])

    # Missing PV-specific constraints evaluated over HP set:
    # d_hat_i = max_{x in HP} A_pv_i x
    d_hat = np.array([
        _max_linear_over_polytope(A_pv[i], H_hp, rhs_hp, name=f"HP_for_PV_row_{i}")
        for i in range(len(A_pv))
    ])

    H = np.vstack([E, A_pv, A_hp])
    h = np.concatenate([
        b1_pv + b1_hp,
        b2_pv + d_hat,
        b_hat + b2_hp,
    ])

    h[np.abs(h) < 1e-10] = 0.0
    return H, h


def compute_nodal_approx_for_with_battery(
    P_pv_max: float,
    P_hp_max: float,
    P_hp_min: float,
    P_dis_max: float,
    P_chg_max: float,
    S_bat_max: float,
    cos_phi_min: float = config.pv_cf_lower_limit,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Outer approximation of:
        F_pv ⊕ F_hp ⊕ F_bat

    Battery set:
        ||[P_bat, Q_bat]||_2 <= S_bat_max
        -P_chg_max <= P_bat <= P_dis_max

    Returns:
        H_lin, h_lin, d_soc
    where:
        H_lin @ [P,Q] <= h_lin
        ||[P,Q]||_2 <= d_soc

    Note:
        This is an outer approximation. It is suitable for screening or
        upper-bound flexibility studies, but can include infeasible aggregate
        points.
    """

    P_pv_max = max(float(P_pv_max), 0.0)
    P_hp_max = float(P_hp_max)
    P_hp_min = float(P_hp_min)
    P_dis_max = max(float(P_dis_max), 0.0)
    P_chg_max = max(float(P_chg_max), 0.0)
    S_bat_max = max(float(S_bat_max), 0.0)

    if P_hp_min > P_hp_max:
        raise ValueError(
            f"Invalid HP bounds: P_hp_min={P_hp_min}, P_hp_max={P_hp_max}"
        )

    H_lin, h_lin = compute_nodal_approx_for_linear_constraints(
        P_pv_max=P_pv_max,
        P_hp_max=P_hp_max,
        P_hp_min=P_hp_min,
        cos_phi_min=cos_phi_min,
    )

    if S_bat_max <= 1e-9 or (P_dis_max <= 1e-9 and P_chg_max <= 1e-9):
        return H_lin, h_lin, 0.0

    # Battery cannot have active power capability greater than apparent power capability.
    P_dis_max = min(P_dis_max, S_bat_max)
    P_chg_max = min(P_chg_max, S_bat_max)

    def max_linear_over_battery(c: np.ndarray) -> float:
        c = np.asarray(c, dtype=float)

        x = cp.Variable(2)
        constraints = [
            cp.norm(x, 2) <= S_bat_max,
            x[0] <= P_dis_max,
            x[0] >= -P_chg_max,
        ]

        prob = cp.Problem(cp.Maximize(c @ x), constraints)
        value = prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Battery support-function solve failed: {prob.status}")

        return float(value)

    # Eq. (12)-style widening:
    # max over battery of each linear row direction.
    d_hat = np.array([
        max(
            max_linear_over_battery(H_lin[i]),
            max_linear_over_battery(-H_lin[i]),
        )
        for i in range(len(H_lin))
    ])

    # Eq. (13)-style SOC widening:
    # Need max_{x in PV+HP} ||x||_1 = max |P| + |Q|.
    # In 2D this can be solved by checking four sign patterns.
    sign_patterns = np.array([
        [ 1.0,  1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [-1.0, -1.0],
    ])

    h_hat = max(
        _max_linear_over_polytope(s, H_lin, h_lin, name=f"PV_HP_l1_sign_{idx}")
        for idx, s in enumerate(sign_patterns)
    )

    H_lin_final = H_lin
    h_lin_final = h_lin + d_hat
    d_soc_final = S_bat_max + h_hat

    h_lin_final[np.abs(h_lin_final) < 1e-10] = 0.0
    return H_lin_final, h_lin_final, float(d_soc_final)

def map_load_time_series():
    mapping_file_path = "00-INPUT-DATA/norway_data/mapping_loads_to_CINELDI_MV_reference_grid.csv"
    load_file_path = "00-INPUT-DATA/norway_data/load_data_CINELDI_MV_reference_system.csv"
    bus_file_path = "00-INPUT-DATA/norway_data/CINELDI_MV_reference_grid_base_bus.csv"
    
    mapping = pd.read_csv(mapping_file_path, sep=";")
    load_ts = pd.read_csv(load_file_path, index_col=0, sep=";")
    bus_data = pd.read_csv(bus_file_path, sep=";")

    mapping["time_series_ID"] = mapping["time_series_ID"].astype(int)
    load_ts.columns = load_ts.columns.astype(int)
    load_ts = load_ts.reset_index(drop=True)  
    base_load_mw = bus_data.set_index("bus_i")["Pd"].astype(float)
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
            full_load_df[bus_id] = load_ts[ts_id].values * float(base_load_mw.get(bus_id, 0.0))

    return full_load_df
