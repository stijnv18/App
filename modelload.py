import pandas as pd
from darts.models import TiDEModel
from joblib import load
from darts import TimeSeries
import matplotlib.pyplot as plt
import numpy as np

# Load the model from the file
model = TiDEModel.load('my_model.pt')
print(model)
if model is None:
    print("Failed to load the model.")
# Load the data from the CSV file
df = pd.read_csv('customer_36v2.csv')

# Convert the 'timestamp' column to datetime if not already done
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split the dataframe
df_train = df[df['timestamp'] < '2011-06-30']
df_test = df[df['timestamp'] >= '2011-06-30']

# Create a TimeSeries object from the train dataframe
series_train = TimeSeries.from_dataframe(df_train, 'timestamp', 'consumption', freq='h', fill_missing_dates=True)

# Predict the next 72 points
prediction = model.predict(n=72, series=series_train)

# Get the values of the prediction
prediction_values = prediction.values()

print(f"Prediction for the next 3 days each hour: {prediction_values}")

# Get the timestamps of the prediction
prediction_timestamps = prediction.time_index

# Clip the predicted values at 0
prediction_values = np.clip(prediction.values(), 0, None)

# Create a new TimeSeries object with the clipped values
prediction = TimeSeries.from_times_and_values(prediction.time_index, prediction_values)

# Print the prediction values along with their corresponding timestamps
for timestamp, value in zip(prediction_timestamps, prediction_values):
	print(f"Prediction for {timestamp}: {value}")

# Now, for each new hour in the test set, make a prediction
for index, row in df_test.iterrows():
    # Create a new TimeSeries object for the new data
    new_data = TimeSeries.from_times_and_values(pd.date_range(start=row['timestamp'], periods=1, freq='h'), [row['consumption']])
    
    # Append the new data to the series
    series_train = series_train.append(new_data)
    
    # Make a prediction for the next hour
    prediction = model.predict(n=1, series=series_train)
    
    # Print the prediction
    print(f"Prediction for {row['timestamp'] + pd.Timedelta(hours=1)}: {prediction.values()[0]}")