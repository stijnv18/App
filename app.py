# Standard library imports
import time

Starttime = time.time()	

from datetime import datetime, timedelta
from threading import Thread
import json
import logging


# Third-party libraries
from darts import TimeSeries
from darts.models import TiDEModel
from flask import Flask, render_template, jsonify, abort, request
from influxdb_client import InfluxDBClient
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from river import metrics, time_series

stoptime = time.time()
print("Elapsed time loading libs: ", stoptime - Starttime)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)

# Global variable to store the latest prediction
latest_prediction = []
training_running = None
actual_values = []
prediction_length = 1
error = []
prev_prediction_length = 0

dbconnecttime = time.time()

# Connect to the InfluxDB server
host = 'http://192.168.2.201:8086'
token = "QaRtTYtoGsLFHzFTMbkx5DbYp9kERjxsVVNJ3oyLYHJRPqOehKsfuf16jhcE6SN-i4pozXIoCKW41gbM9cdiSg=="
org = "beheerder"
bucket = "dataset"
client = InfluxDBClient(url=host, token=token, org=org)
dbconnecstop = time.time()
print("Elapsed time connecting to db: ", dbconnecstop - dbconnecttime)

# Query the data from your bucket
query = """from(bucket: "dataset")
  |> range(start: 2010-07-01T00:00:00Z, stop: 2013-06-30T23:00:00Z)
  |> filter(fn: (r) => r["_measurement"] == "Solar")
  |> filter(fn: (r) => r["_field"] == "consumption" or r["_field"] == "sunshine_duration" or r["_field"] == "cloud_cover")"""

tables = client.query_api().query(query, org=org)

data = {'consumption': [], 'cloud_cover': [], 'sunshine_duration': []}
for table in tables:
    for record in table.records:
        field = record.get_field()
        if field in data:
            data[field].append((record.get_time(), record.get_value()))

dfs = {field: pd.DataFrame(values, columns=['timestamp', field]) for field, values in data.items()}

# Convert the data to a pandas DataFrame
dfT = dfs['consumption']
for field in ['cloud_cover', 'sunshine_duration']:
    dfT = dfT.merge(dfs[field], on='timestamp', how='outer')
dfT['timestamp'] = dfT['timestamp'].dt.tz_convert(None)

requesttime = time.time()
print("Elapsed time querying data: ", requesttime - dbconnecstop)

def create_graph(dates, values):
	# Generate your graph
	graph = go.Figure(
		data=[go.Scatter(x=dates, y=values)]
	)
	# Convert the figures to JSON
	graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

@app.route('/')
def index():
    # Fetch the latest prediction
    prediction = get_latest_prediction().get_json()

    # Check if there is a prediction
    if not prediction['prediction_dates'] or not prediction['prediction_values'] or not prediction['actual_dates'] or not prediction['actual_values']:
        # No prediction or actual values available, return a default page
        return render_template('index2.html', graphJSON='null')

    # Unpack the dates and values from the prediction
    prediction_dates, prediction_values = prediction["prediction_dates"],prediction["prediction_values"]
    actual_dates, actual_values = prediction["actual_dates"], prediction["actual_values"]

    # Call the function to create the graph
    graphJSON = create_graph(prediction_dates, prediction_values, actual_dates, actual_values)
	

    return render_template('index.html', graphJSON=graphJSON)

@app.route('/latest_prediction', methods=['GET'])
def get_latest_prediction():
	global latest_prediction
	global actual_values
 
	if not latest_prediction or not actual_values:
		return jsonify({'prediction_dates': [], 'prediction_values': [], 'actual_dates': [], 'actual_values': []})

	prediction_dates, prediction_values = zip(*latest_prediction)
	prediction_dates = [date.isoformat() for date in prediction_dates]

	# Check if actual_values is a list before trying to unpack it
	if isinstance(actual_values, list):
		actual_date, actual_value = zip(*actual_values)
		actual_date = [date.isoformat() for date in actual_date]
	else:
		actual_date = []
		actual_value = []
	return jsonify({
		'prediction_dates': prediction_dates, 
		'prediction_values': prediction_values,
		'actual_dates': actual_date, 
		'actual_values': actual_value
	})
@app.route('/latest_error', methods=['GET'])
def get_error():
    global error
    if not error:
        return jsonify({'error_dates': [], 'error_values': []})

    if isinstance(error, (list, tuple)) and len(error) > 0 and isinstance(error[0], (list, tuple)):
        error_dates, error_values = zip(*error)
    else:
        # Handle the case where error is not iterable
        error_dates, error_values = [], []

    error_dates = [date.isoformat() for date in error_dates]

    return jsonify({'error_dates': error_dates, 'error_values': error_values})

@app.route('/start_training', methods=['POST'])
def handle_start_training():
	global training_running
	global prediction_length
	global prev_prediction_length
	# Get the model and prediction_length from the request body
	data = request.get_json()
	model_name = data.get('model')
	prediction_length = int(data.get('prediction_length', 1))
 
	if model_name == 'SNARIMAX' and (training_running != 1 or prev_prediction_length != prediction_length):
		training_running = 0
		time.sleep(1)
		Thread(target=run_SNARIMAX).start()
		prev_prediction_length = prediction_length
	elif model_name == 'Holtwinters' and (training_running != 2 or prev_prediction_length != prediction_length):
		training_running = 0
		time.sleep(1)
		Thread(target=run_Holtwinters).start()
		prev_prediction_length = prediction_length
	elif model_name == 'TIDE' and (training_running != 3 or prev_prediction_length != prediction_length):
		training_running = 0
		time.sleep(1)
		Thread(target=run_TIDE).start()
		prev_prediction_length = prediction_length
	else:
		abort(400, 'model already running or invalid model selected')
	return jsonify({'message': 'Training started'})

def run_SNARIMAX():
	global training_running
	global latest_prediction
	global actual_values
	global prediction_length
	global error
	global df
	training_running = 1
 
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
    
	print("Running the model training snarimax...")
	model_without_exog = (time_series.SNARIMAX(p=1,d=0,q=1,sp=0,sd=1,sq=1,m=24))
	start_time = datetime.strptime("2011-11-23T09:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	end_time = datetime.strptime("2014-02-28T00:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	
	current_time = start_time
	while current_time <= end_time and training_running == 1:
		stop_time = current_time + timedelta(days=30)
		start_str = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
		stop_str = stop_time.strftime('%Y-%m-%dT%H:%M:%SZ')
		print(f"Training from {start_str} to {stop_str}")
  
		query = f"""from(bucket: "dataset")
		|> range(start: {start_str}, stop: {stop_str})
		|> filter(fn: (r) => r["_measurement"] == "measurement")
		|> filter(fn: (r) => r["_field"] == "MeanEnergyConsumption")"""
  
		tables = client.query_api().query(query, org=org)
		i = 0
		# Process the data for this month
		for table in tables:
			for record in table.records:
				if training_running != 1:
					break
				start_time_sleep = time.time()
				y = record.get_value()
				time_of_observation = record.get_time()
				model_without_exog.learn_one(y)
				if i > 0:  # Skip the first observation
					forecast = model_without_exog.forecast(horizon=prediction_length)
					actual_values.append((time_of_observation, y))
					actual_values = actual_values[-168:]
					latest_prediction.append((time_of_observation + timedelta(hours=prediction_length), forecast[prediction_length-1]))
					latest_prediction = latest_prediction[-168-prediction_length:]
				i+=1
				end_time_sleep = time.time()  # End time of the loop
				loop_time = end_time_sleep - start_time_sleep  # Time taken by the loop
				sleep_time = max(0.1 - loop_time, 0)  # Sleep time (100ms - loop time), but not less than 0
				time.sleep(sleep_time)
		current_time = stop_time
	training_running = 0

def run_Holtwinters():
	global training_running
	global latest_prediction
	global actual_values
	global df
	training_running = 2
	
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
    
	print("Running the model training holt...")


	model = time_series.HoltWinters(alpha=0.3,beta=0.1,gamma=0.5,seasonality=24,multiplicative=True,)
	metric = metrics.MAE()
 
	start_time = datetime.strptime("2011-11-23T09:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	end_time = datetime.strptime("2014-02-28T00:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	current_time = start_time
 
	while current_time <= end_time and training_running == 2:
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
				if training_running != 2:
					break
				start_time_sleep = time.time()
				y = record.get_value()
				time_of_observation = record.get_time()
				model.learn_one(y)
				if i >= model.seasonality:
					prediction = model.forecast(horizon=prediction_length)
					actual_values.append((time_of_observation, y))
					latest_prediction.append((time_of_observation, prediction[prediction_length-1]))
					actual_values = actual_values[-168:]
					latest_prediction = latest_prediction[-168-prediction_length:]
				i += 1
				end_time_sleep = time.time()  # End time of the loop
				loop_time = end_time_sleep - start_time_sleep  # Time taken by the loop
				sleep_time = max(0.1 - loop_time, 0)  # Sleep time (100ms - loop time), but not less than 0
				time.sleep(sleep_time)
		current_time = stop_time

	training_running = 0
def run_TIDE():
	global training_running
	global latest_prediction
	global actual_values
	global dfT
	global prediction_length
	training_running = 3
 
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
 
	print("Running the model TIDE...")
	# Load the model from the file
	model = TiDEModel.load('my_model.pt')
	if model is None:
		print("Failed to load the model.")

	start_time = datetime.strptime("2010-07-01T00:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	end_time = datetime.strptime("2013-06-30T23:00:00Z", '%Y-%m-%dT%H:%M:%SZ')
	
	current_time = start_time
	while current_time <= end_time and training_running == 3:
		stop_time = current_time + timedelta(days=48)
		start_str = current_time.strftime('%Y-%m-%dT%H:%M:%SZ')
		stop_str = stop_time.strftime('%Y-%m-%dT%H:%M:%SZ')
		print(f"Training from {start_str} to {stop_str}")
		query = f"""from(bucket: "poggers")
		|> range(start: {start_str}, stop: {stop_str})
		|> filter(fn: (r) => r["_measurement"] == "Solar")
		|> filter(fn: (r) => r["_field"] == "consumption" or r["_field"] == "sunshine_duration" or r["_field"] == "cloud_cover")"""

		tables = client.query_api().query(query, org=org)
		i = 0
		data = {'consumption': [], 'cloud_cover': [], 'sunshine_duration': []}
		# Process the data for this month
		for table in tables:
			for record in table.records:
				if training_running != 3:
					break
				y = record.get_field()
				if y in data:
					data[y].append((record.get_time(), record.get_value()))
		dfs = {field: pd.DataFrame(values, columns=['timestamp', field]) for field, values in data.items()}

		# Convert the data to a pandas DataFrame
		dfT = dfs['consumption']
		for field in ['cloud_cover', 'sunshine_duration']:
			dfT = dfT.merge(dfs[field], on='timestamp', how='outer')
		dfT['timestamp'] = dfT['timestamp'].dt.tz_convert(None)
		# Convert the DataFrame to a TimeSeries
		dfT.set_index('timestamp', inplace=True)
		series = TimeSeries.from_dataframe(dfT, fill_missing_dates=True, freq='h')
		if i > 0:
			print("test")
			# Now make a prediction with the updated series
			prediction = model.predict(n=prediction_length, series=series)
			# Get the values of the prediction
			prediction_values = prediction.values()
			# Get the timestamps of the prediction
			prediction_timestamps = prediction.time_index
			# Clip the predicted values at 0
			prediction_values = np.clip(prediction.values(), 0, None)
			# Create a new TimeSeries object with the clipped values
			prediction = TimeSeries.from_times_and_values(prediction.time_index, prediction_values)
			# Append the actual value and its timestamp to actual_values
			actual_values.append((time_of_observation, y))
			actual_values = actual_values[-168:]
			# Append the prediction and its timestamp to latest_prediction
			latest_prediction.append((time_of_observation + pd.Timedelta(hours=1), float(prediction.values()[0].item())))
			latest_prediction = latest_prediction[-168:]
		i+=1
		time.sleep(0.1)
		current_time = stop_time
	training_running = 0
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8888)