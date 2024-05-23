import pandas as pd
from darts.models import TiDEModel
from joblib import load
from darts import TimeSeries

# Load the model from the file
model = TiDEModel.load('my_model.pt')
print(model)
if model is None:
    print("Failed to load the model.")
# Load the data from the CSV file
df = pd.read_csv('customer_36v2.csv')

# Convert the 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create a TimeSeries object from the entire dataframe
series = TimeSeries.from_dataframe(df, 'timestamp', 'consumption', freq='h', fill_missing_dates=True)

# Predict the next 72 points
prediction = model.predict(n=72, series=series)

# Get the values of the prediction
prediction_values = prediction.values()

print(f"Prediction for the next 3 days each hour: {prediction_values}")

# Get the timestamps of the prediction
prediction_timestamps = prediction.time_index

# Print the prediction values along with their corresponding timestamps
for timestamp, value in zip(prediction_timestamps, prediction_values):
    print(f"Prediction for {timestamp}: {value}")