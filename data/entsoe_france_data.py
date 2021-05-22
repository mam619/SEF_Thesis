# Using ENTSOE built in python package get data from France - load, generation and DA prices

from entsoe import EntsoePandasClient
import pandas as pd


client = EntsoePandasClient(api_key="cc9c9fbb-2142-405b-88e8-8ab410f35694")
start = pd.Timestamp("20160101", tz="UTC")
end = pd.Timestamp("20190102", tz="UTC")
country_code = "FR"


if __name__ == "__main__":

    DA_prices = client.query_day_ahead_prices(country_code, start=start, end=end)
    DA_prices.to_csv("france_DA_prices.csv")

    load_france = client.query_load_forecast(country_code, start=start, end=end)
    load_france.to_csv("france_load_forecast.csv")

    gen_france = client.query_generation_forecast(country_code, start=start, end=end)
    gen_france.to_csv("france_generation_forecast.csv")
