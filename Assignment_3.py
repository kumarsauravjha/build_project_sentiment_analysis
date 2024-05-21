#%%
'''This is not complete yet'''
import pandas as pd
import matplotlib.pyplot as plt
# %%
api_key = '4bd0b251e15c9d8a96893a35bd1b1611'
# %%
import requests

url = 'https://api.openweathermap.org/data/2.5/onecall'

# Set the parameters
params = {
    'lat': 51.507351,
    'lon':-0.127758,
    'appid': api_key,
    'units': 'metric'
}

# Make the GET request
response = requests.get(url, params=params)

# Print the status code (should be 200)
print(response.status_code)
# %%
import json

# Convert the response to JSON
data = response.json()

# Print the weather description
print(data['weather'][0]['description'])
# %%
