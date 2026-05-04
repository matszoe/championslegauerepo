import requests
import pandas as pd
import json
from io import StringIO

token = 'ffda759b054c0a27da0ded16383a4406aebc847f'
api_base = 'https://www.renewables.ninja/api/'

s = requests.session()
# Send token header with each request
s.headers = {'Authorization': 'Token ' + token}


##
# PV example
##

url = api_base + 'data/pv'

args = {
    'lat': 63.4257,
    'lon': 10.4043,
    'date_from': '2018-01-01',
    'date_to': '2018-12-31',
    'dataset': 'merra2',
    'capacity': 1.0,
    'system_loss': 0.1,
    'tracking': 0,
    'tilt': 35,
    'azim': 180,
    'format': 'csv'
}

r = s.get(url, params=args)


lines = r.text.split('\n')
# Find the line with column headers
header_idx = next(i for i, line in enumerate(lines) if line.startswith('time,electricity'))
csv_data = '\n'.join(lines[header_idx:])
data = pd.read_csv(StringIO(csv_data), parse_dates=['time'])
data.set_index('time', inplace=True)

print(data.head())

# Save as CSV
output_path = '00-INPUT-DATA/PV-DATA/PV_timeseries.csv'
data.to_csv(output_path)
print(f"\nData saved to {output_path}")