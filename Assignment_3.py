#%%
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
api_key = '4bd0b251e15c9d8a96893a35bd1b1611'
# %%
#API endpoint
url = 'http://api.openweathermap.org/data/2.5/forecast'

cities = ['Bombay', 'New Delhi', 'Kolkata', 'Pune', 'Doha']

def get_weather_forecast(city):
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    #GET request
    response = requests.get(url, params=params)
    # Checking the status code
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
# Calculate average values for each city
average_data = all_data.groupby('city').mean().reset_index()

# Sort the data in descending order for each parameter
average_data_temp = average_data.sort_values(by='temperature', ascending=False)
average_data_pressure = average_data.sort_values(by='pressure', ascending=False)
average_data_humidity = average_data.sort_values(by='humidity', ascending=False)

# Plot comparative bar chart in descending order
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(x='city', y='temperature', data=average_data_temp, ax=ax[0])
ax[0].set_title('Average Temperature (°C)')
ax[0].set_ylabel('Temperature (°C)')

sns.barplot(x='city', y='pressure', data=average_data_pressure, ax=ax[1])
ax[1].set_title('Average Pressure (hPa)')
ax[1].set_ylabel('Pressure (hPa)')

sns.barplot(x='city', y='humidity', data=average_data_humidity, ax=ax[2])
ax[2].set_title('Average Humidity (%)')
ax[2].set_ylabel('Humidity (%)')

plt.suptitle('Comparative Weather Analysis')
plt.show()
# %%
