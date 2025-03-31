import pandas as pd
import requests
# https://ourworldindata.org/co2-dataset-sources
# Fetch the data.
df = pd.read_csv(
    "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.csv?v=1&csvType=full&useColumnShortNames=false",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'})

# Fetch the metadata
metadata = requests.get(
    "https://ourworldindata.org/grapher/annual-co2-emissions-per-country.metadata.json?v=1&csvType=full&useColumnShortNames=false").json()
print(df.head())
print(metadata)