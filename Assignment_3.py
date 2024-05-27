#%%
'''This is not complete yet'''
import pandas as pd
import matplotlib.pyplot as plt
# %%
api_key = '4bd0b251e15c9d8a96893a35bd1b1611'
# %%
import requests

# Your API key
# api_key = 'your_api_key'

# Set the API endpoint
url = 'http://api.openweathermap.org/data/2.5/forecast'

cities = ['Bombay', 'New Delhi', 'Kolkata', 'Pune', 'Doha']

def get_weather_forecast(city):
    # Set the parameters
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    # Make the GET request
    response = requests.get(url, params=params)
    # Check the status code
    if response.status_code == 200:
        # Convert the response to JSON
        data = response.json()
        
        forecasted_list = data['list']
        dates = []
        temperatures = []
        humidities = []
        pressures = []

        for list in forecasted_list:
            dates.append(list['dt_txt'])
            temperatures.append(list['main']['temp'])
            humidities.append(list['main']['humidity'])
            pressures.append(list['main']['pressure'])

        df = pd.DataFrame({
            'date': dates,
            'temperature':temperatures,
            'humidity':humidities,
            'pressure':pressures,
            'city':city
        })

        df.date = pd.to_datetime(df.date)

        return df
    else:
        print(f"Error: {response.status_code} - {response.text}")


all_data = pd.concat([get_weather_forecast(city) for city in cities])

#%%
plt.figure(figsize=(15, 10))

bombay_data = all_data[all_data['city'] == 'Bombay']


# %%
plt.figure(figsize=(15, 15))

plt.subplot(3, 1, 1)
for city in cities:
    city_data = all_data[all_data['city'] == city]
    plt.plot(city_data['date'], city_data['temperature'], label=f'{city} Temperature (°C)')
plt.title('5 Day Weather Forecast')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.subplot(3, 1, 2)
for city in cities:
    city_data = all_data[all_data['city'] == city]
    plt.plot(city_data['date'], city_data['pressure'], label=f'{city} Pressure (hPa)')
plt.ylabel('Pressure (hPa)')
plt.legend()

plt.subplot(3, 1, 3)
for city in cities:
    city_data = all_data[all_data['city'] == city]
    plt.plot(city_data['date'], city_data['humidity'], label=f'{city} Humidity (%)')
plt.ylabel('Humidity (%)')
plt.xlabel('Date')
plt.legend()

plt.tight_layout()
plt.show()
# %%
