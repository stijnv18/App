import pandas as pd
import pickle
from darts import TimeSeries
from joblib import load

# Load the data
df = pd.read_csv('customer_36.csv')

# Create a 'DateTime' column
df['DateTime'] = pd.date_range(start='2010-07-01', periods=len(df), freq='h')

# Create a TimeSeries instance
series = TimeSeries.from_dataframe(df, 'DateTime', 'consumption')

# Load the pre-trained model
with open('my_model.joblib', 'rb') as f:
    model = load('my_model.joblib')

# Check if the model is loaded correctly
if model is None:
    print('Failed to load the model')
else:
    # Generate a 7-day forecast
    forecast = model.predict(series)

    # Replace negative values with 0
    forecast.values()[forecast.values() < 0] = 0

    # Print the forecasted values
    print('Forecasted values: ', forecast.values())