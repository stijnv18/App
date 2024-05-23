import pandas as pd
from joblib import load
from darts import TimeSeries

# Load the model from the file
model = load('my_model.joblib')
print(model)
if model is None:
    print("Failed to load the model.")
# Load the data from the CSV file
df = pd.read_csv('customer_36v2.csv')

# Convert the 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get predictions one at a time
for index in range(47, len(df)):
    # Create a TimeSeries object
    series = TimeSeries.from_dataframe(df.iloc[:index+1], 'timestamp', 'consumption', freq='H', fill_missing_dates=True)
    
    # Predict the next point
    prediction = model.predict(n=1, series=series)
    
    print(f"Prediction for row {index}: {prediction}")