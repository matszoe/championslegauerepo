import pandas as pd

load_existing_aggregation = False
scenario = "with_battery" # "base", "with_battery", "benchmark"
pv_cf_lower_limit = 0.9
battery_s_max = 1
SOC_inital = 0.5
battery_capacity_at_FC = 1

optinization_mode = "SingleTimestep" # "SingleTimestep", "MultiTimestep"
timestep_under_consideration = pd.Timestamp("2018-09-01 12:00:00", tz='UTC')

alpha = 1.0
beta = 1.0