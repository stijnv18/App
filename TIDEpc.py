import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel
from darts.metrics import mae
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle
# Load the data
df = pd.read_csv('customer_36.csv')

# Calculate the number of data points
num_data_points = len(df)

# Create a 'DateTime' column
df['DateTime'] = pd.date_range(start='2010-07-01', periods=num_data_points, freq='H')

# Create a TimeSeries instance
series = TimeSeries.from_dataframe(df, 'DateTime', 'consumption')
# Create a TimeSeries instance for the covariates
covar = TimeSeries.from_dataframe(df, 'DateTime', ['cloud_cover'], ['sunshine_duration'])
# Split the data into a training set and a validation set
train, val = series.split_after(pd.Timestamp('2013-06-23'))
common_model_args = {
    "input_chunk_length": 48,  # lookback window````
    "output_chunk_length": 14,  # forecast/lookahead window
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
}
# Create a TIDe model
model = TiDEModel(**common_model_args)
# Train the model
model.fit(train, epochs=20, verbose=True)
# Generate a 7-day forecast
forecast = model.predict(7*24)  # 7 days, 24 hours per day, 2 data points per hour
# Slice the actual series to the last week
last_week = series[-7*24:]  # 7 days, 24 hours per day, 2 data points per hour
# Calculate the MAE of the forecast
error = mae(forecast, last_week)
print('MAE: ', error)

# Generate a 3-day forecast
forecast = model.predict(3*24)  # 3 days, 24 hours per day, 2 data points per hour
# Slice the actual series to the last 3 days
last_3_days = series[-3*24:]  # 3 days, 24 hours per day, 2 data points per hour
#print('Forecasted values: ', forecast.values())
# Flatten the list of lists
forecast_flat = [item for sublist in forecast.values() for item in sublist]
#print('Forecasted values: ', forecast_flat)
# Convert the flattened list to a NumPy array
forecast_np = np.array(forecast_flat)
# Convert to NumPy arrays
#forecast_np = np.array(forecast)
print('Forecasted values: ', last_3_days.values())
# Flatten the list of lists
last_3_days_flat = [item[0] for item in last_3_days.values()]
print('Forecasted values: ', last_3_days_flat)
# Convert the flattened list to a NumPy array
last_3_days_np = np.array(last_3_days_flat)

# Convert the forecast TimeSeries to a DataFrame
forecast_df = forecast.pd_dataframe()
# If forecast is a DataFrame or Series
if isinstance(forecast, (pd.DataFrame, pd.Series)):
    forecast_np = forecast.values
# If forecast is a list of lists
elif isinstance(forecast, list) and isinstance(forecast[0], list):
    forecast_np = np.array([item for sublist in forecast for item in sublist])
else:
    print("Unable to convert forecast to a NumPy array.")
# Assuming `forecast_np` are your predicted values and `last_3_days_np` are the actual values
mae = mean_absolute_error(last_3_days_np, forecast_np)
# Replace negative values with 0
forecast_np = np.where(forecast_np < 0, 0, forecast_np)
print(f"Mean Absolute Error: {mae}")
# Create a figure and axis
fig, ax = plt.subplots()

# Plot the actual data
ax.plot(last_3_days_np, label='Actual')

# Plot the forecasted data
ax.plot(forecast_np, label='Forecast')

# Add a legend
ax.legend()

# Show the plot
plt.show()
# Save the model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)