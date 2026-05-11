import pandas as pd

## script is used to convert the heat demand profile to hourly values and save as csv file
def load_heat_demand_profile() -> pd.DataFrame:
    """
    """
    df_raw = pd.read_excel(
        "00-INPUT-DATA/HP-DATA/lastprofil-waermepumpe-muenchen.xls",
        engine='xlrd',
        sheet_name='Wärmepumpen-Lastprofil',
        header=None,
    )
    temp_row = df_raw.iloc[4, 1:].values   
    temp_classes = []
    for v in temp_row:
        if isinstance(v, str) and '<' in str(v):
            temp_classes.append(-14)
        elif isinstance(v, str) and '>=' in str(v):
            temp_classes.append(18)
        else:
            temp_classes.append(int(float(v)))
    data = df_raw.iloc[5:101, 1:].values.astype(float)


    hourly = data.reshape(24, 4, 33).mean(axis=1)

    profile = pd.DataFrame(
        hourly,
        index=range(24),
        columns=temp_classes,
    )

    return profile


test = load_heat_demand_profile()

test.to_csv("00-INPUT-DATA/HP-DATA/hp_profile.csv", index=False)