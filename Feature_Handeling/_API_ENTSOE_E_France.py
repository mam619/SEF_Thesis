# =============================================================================
# Using ENTSOE - E API
# 
# Get data from France (Load and Generation)
# =============================================================================

from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key = 'cc9c9fbb-2142-405b-88e8-8ab410f35694')

start = pd.Timestamp('20160101', tz='UTC')
end = pd.Timestamp('20190102', tz='UTC')
country_code = 'FR'  


# methods that return Pandas Series

prices_france = client.query_day_ahead_prices(country_code, start=start, end=end)
prices_france.to_csv('.France_DA_prices(2).csv')

load_forecast_france = client.query_load_forecast(country_code, start=start, end=end)
load_forecast_france.to_csv('.France_Load_forecast(2).csv')

generation_forecast = client.query_generation_forecast(country_code, start=start, end=end)
generation_forecast.to_csv('.France_Generation_forecast(2).csv')



