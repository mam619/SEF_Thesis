# =============================================================================
# Using ENTSOE - E API
# 
# Get data from France (Load and Generation)
# =============================================================================

from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key = 'eeee8e14-4cee-4674-8b76-6db4d5066cb1')

start = pd.Timestamp('20160101', tz='Europe/Brussels')
end = pd.Timestamp('20181231', tz='Europe/Brussels')
country_code = 'FR'  

# methods that return Pandas Series

prices_france = client.query_day_ahead_prices(country_code, start=start, end=end)
prices_france.to_csv('DA_France_prices.csv')

load_forecast_france = client.query_load_forecast(country_code, start=start, end=end)
load_forecast_france.to_csv('load_forecast_france.csv')

generation_forecast = client.query_generation_forecast(country_code, start=start, end=end)
generation_forecast.to_csv('Generation_forecast_france.csv')
