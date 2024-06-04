import pandas as pd
from river import time_series
import time
from river import metrics
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, Dialect

# Global variable to store the latest prediction
latest_prediction = []
training_running = None
actual_values = []
prediction_length = 1
error = []
prev_prediction_length = 0

# Connect to the InfluxDB server
host = 'http://localhost:8086'
token = "QaRtTYtoGsLFHzFTMbkx5DbYp9kERjxsVVNJ3oyLYHJRPqOehKsfuf16jhcE6SN-i4pozXIoCKW41gbM9cdiSg=="
org = "beheerder"
bucket = "dataset"
client = InfluxDBClient(url=host, token=token)

print("Running the model training snarimax...")

# Define the start time and end time
start_time = datetime.strptime("2011-11-23T09:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
end_time = datetime.strptime("2014-02-28T00:00:00Z", '%Y-%m-%dT%H:%M:%SZ')

# Initialize current_time to start_time
current_time = start_time

model_without_exog = time_series.SNARIMAX(p=1,d=0,q=1,sp=0,sd=1,sq=1,m=24)
mae_without_exog = metrics.MAE()

while current_time <= end_time:
    # Define the stop time for this month
    stop_time = current_time + timedelta(days=30)  # Approximate a month as 30 days

    # Format times as strings
    start_str = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    stop_str = stop_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"Training from {start_str} to {stop_str}")
    # Define the query for this month
    query = f"""from(bucket: "dataset")
    |> range(start: {start_str}, stop: {stop_str})
    |> filter(fn: (r) => r["_measurement"] == "measurement")
    |> filter(fn: (r) => r["_field"] == "MeanEnergyConsumption")"""

    # Fetch the data for this month
    tables = client.query_api().query(query, org=org)
    i = 0
    # Process the data for this month
    for table in tables:
        for record in table.records:
            y = record.get_value()
            model_without_exog.learn_one(y)
            if i > 0:  # Skip the first observation
                forecast = model_without_exog.forecast(horizon=prediction_length)  # forecast 1 step ahead
            i+=1

    # Move to the next month
    current_time = stop_time

training_running = 0