import pandas as pd

scenario = "with_battery" # "base", "with_battery", "benchmark"
pv_cf_lower_limit = 0.9
SOC_inital = 0.5

soc_min = 0.2
soc_max = 0.8
battery_capacity_at_FC = 1
T_room_initial = 20

single_timestep = pd.Timestamp("2018-04-01 12:00:00")

multi_timestep_interval = [pd.Timestamp("2018-04-01 12:00:00"),
                            pd.Timestamp("2018-04-01 13:00:00")] ## start and end timesteps

alpha = 1.0
beta = 0
n_ffor_directions = 32