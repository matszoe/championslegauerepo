import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime

def estimate_correlations(load_df):
    pv_data_path = "00-INPUT-DATA/PV-DATA/PV_timeseries.csv"
    temperature_data_path = "00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv"

    R2_THRESHOLD = 0.10
    cf = pd.read_csv(pv_data_path, parse_dates=['time'], index_col='time')
    T  = pd.read_csv(temperature_data_path, parse_dates=['date'], index_col='date')

    time_index = pd.date_range("2018-01-01", periods=8760, freq="h")

    results = {}

    for node in load_df.columns:
        P = pd.Series(load_df[node].values, index=time_index)

        daily_mean = P.resample("D").transform("mean")
        dP = (P - daily_mean).values

        mask_day = cf.values > 0.05
        X_pv = cf.values[mask_day].reshape(-1, 1)
        y_pv = dP[mask_day.flatten()]

        reg_pv = LinearRegression().fit(X_pv, y_pv)
        r2_pv  = r2_score(y_pv, reg_pv.predict(X_pv))
        alpha  = reg_pv.coef_[0] if r2_pv > R2_THRESHOLD else 0.0

        mask_night = cf.values < 0.01
        T_night    = T.values[mask_night]
        P_night    = P.values[mask_night.flatten()]  

        poly   = PolynomialFeatures(degree=2)
        X_hp   = poly.fit_transform(T_night.reshape(-1, 1))
        reg_hp = LinearRegression().fit(X_hp, P_night)
        r2_hp  = r2_score(P_night, reg_hp.predict(X_hp))

        if r2_hp > R2_THRESHOLD:
            P_full = reg_hp.predict(poly.transform([[-8]]))[0]
            P_zero = reg_hp.predict(poly.transform([[15]]))[0]
            cap_hp = max(0.0, P_full - P_zero)
        else:
            cap_hp = 0.0

        results[node] = {
            "cap_pv_mw": max(0.0, alpha),
            "r2_pv":     r2_pv,
            "cap_hp_mw": cap_hp,
            "r2_hp":     r2_hp,
        }

    results_df = pd.DataFrame(results).T

    return results_df

def add_battery_capacity(correlation_df):
    battery_cap_path = "00-INPUT-DATA/norway_data/scenario_LEC_and_FCS.csv"
    battery_data = pd.read_csv(battery_cap_path, index_col=["bus_i"], sep=";")

    correlation_df["cap_battery_mw"] = 0
    for i in battery_data.index:
        if battery_data.loc[i,"label"] == "FCS_highway" or battery_data.loc[i,"label"] == "FCS_shopping_mall":
            correlation_df.loc[i, "cap_battery_mw"] = battery_data.loc[i,"load_added_MW"]

    return correlation_df

def plot_temperature_load_correlation():
    load_data_path = "00-INPUT-DATA/norway_data/load_data_CINELDI_MV_reference_system.csv"

    temperature_data_path = "00-INPUT-DATA/TEMP-DATA/TEMP_timeseries.csv"

    temps = pd.read_csv(temperature_data_path, parse_dates=['date'], index_col='date')
    load_full_data = pd.read_csv(load_data_path, sep=';', parse_dates=['Time'], index_col='Time')
    relevant_load_data = load_full_data[["5", "9", "14", "21","38", "40", "62" ,"93","104"]]

    # Align data by matching timestamps
    try:
        if temps.index.tz is not None:
            temps.index = temps.index.tz_convert(None)
    except:
        pass
    try:
        if load_full_data.index.tz is not None:
            load_full_data.index = load_full_data.index.tz_convert(None)
    except:
        pass


    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx, col in enumerate(relevant_load_data.columns):
        ax = axes[idx]
        # Get matching data
        mask = relevant_load_data[col].notna()
        x = temps['temperature_2m'].values[:len(relevant_load_data)][mask]
        y = relevant_load_data[col].values[mask]
        ax.scatter(x, y, alpha=0.3, s=10)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Load')
        ax.set_title(f'Load {col} vs Temperature')

    plt.tight_layout()
    plt.show()

def plot_pv_load_correlation():
    load_data_path = "00-INPUT-DATA/norway_data/load_data_CINELDI_MV_reference_system.csv"

    pv_data_path = "00-INPUT-DATA/PV-DATA/PV_timeseries.csv"

    pvs = pd.read_csv(pv_data_path, parse_dates=['time'], index_col='time')
    load_full_data = pd.read_csv(load_data_path, sep=';', parse_dates=['Time'], index_col='Time')
    relevant_load_data = load_full_data[["5", "9", "14", "21","38", "40", "62" ,"93","104"]]

    # Align data by matching timestamps
    try:
        if pvs.index.tz is not None:
            pvs.index = pvs.index.tz_convert(None)
    except:
        pass
    try:
        if load_full_data.index.tz is not None:
            load_full_data.index = load_full_data.index.tz_convert(None)
    except:
        pass

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for idx, col in enumerate(relevant_load_data.columns):
        ax = axes[idx]
        # Get matching data
        mask = relevant_load_data[col].notna()
        x = pvs['electricity'].values[:len(relevant_load_data)][mask]
        y = relevant_load_data[col].values[mask]
        ax.scatter(x, y, alpha=0.3, s=10)
        ax.set_xlabel('PV Generation (kW)')
        ax.set_ylabel('Load')
        ax.set_title(f'Load {col} vs PV Generation')

    plt.tight_layout()
    plt.show()

