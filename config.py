import pandas as pd

scenario = "with_battery" # "base", "with_battery", "benchmark"
pv_cf_lower_limit = 0.9
battery_s_max = 1
SOC_inital = 0.5
battery_capacity_at_FC = 1

single_timestep = pd.Timestamp("2018-09-01 12:00:00", tz='UTC')

multi_timestep_interval = [pd.Timestamp("2018-09-01 12:00:00", tz='UTC'),
                            pd.Timestamp("2018-09-01 13:00:00", tz='UTC')] ## start and end timesteps

alpha = 1.0
beta = 1.0