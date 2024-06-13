# Standard library imports
import time

Starttime = time.time()	

from datetime import datetime, timedelta
from threading import Thread
import json
import logging
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

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
from pandas import Timestamp
from homeassistant_api import Client,State

stoptime = time.time()
print("Elapsed time loading libs: ", stoptime - Starttime)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

# Global variable to store the latest prediction
latest_prediction = []
training_running = None
actual_values = []
prediction_length = 1
error = []
prev_prediction_length = 0

dbconnecttime = time.time()

homeass = "http://homeassistant:8123/api/"
homeass = "http://localhost:8123/api/"
homeass_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIzODM2ODY2OWNhNTQ0MGRlODk5ODA5NGJjZmJiMjMyNiIsImlhdCI6MTcxODA5NjQ2OSwiZXhwIjoyMDMzNDU2NDY5fQ.V_3CZP7ZcR5eHNHAJmUOJlh-9wOQoF8ZzXxmSd5Mz_8"
client_homeass = Client(homeass, homeass_token)

# Connect to the InfluxDB server
host = 'http://influxdb:8086'
host =  'http://localhost:8086'
token = "A8N8FW0T9zLcF5Rx7hwZfAs10ADmNxqQtqi9t3c_L6s59RjeXbcZRXC2nqgb8RgmSBQzwcMJJxS7EenDP3-P1Q=="
org = "beheerder"
client = InfluxDBClient(url=host, token=token, org=org)

dbconnecstop = time.time()
homeass = "http://homeassistant.home:8123/"
homeass_token = "Bearer" + " " + "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiI2ZjQwMzQwYzQwZjg0ZjQwYjQwZjQwZjQwZjQwZjQwZjQiLCJpYXQiOjE2MjYwNjYwNzYsImV4cCI6MTY1NjYwNjA3Nn0.1Z6"
print("Elapsed time connecting to db: ", dbconnecstop - dbconnecttime)

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
        return render_template('index.html', graphJSON='null')

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
					latest_prediction.append((time_of_observation + timedelta(hours=prediction_length), prediction[prediction_length-1]))
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
		query = f"""from(bucket: "dataset")
		|> range(start: {start_str}, stop: {stop_str})
		|> filter(fn: (r) => r["_measurement"] == "Solar")
		|> filter(fn: (r) => r["_field"] == "consumption")"""

		tables = client.query_api().query(query, org=org)
	
		data = []
		# Process the data for this month
		for table in tables:
			for record in table.records:
				if training_running != 3: 	
					break
				start_time_sleep = time.time()
				y = record.get_value()
				time_of_observation = record.get_time()
				data.append((time_of_observation, y))
				df = pd.DataFrame(data, columns=['timestamp', 'consumption'])	
				df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
				
				series = TimeSeries.from_dataframe(df, 'timestamp','consumption',fill_missing_dates=True, freq='h')
				if len(series) >= 48:
					prediction = model.predict(n=prediction_length, series=series, verbose=False, show_warnings=False)
					prediction_values = prediction.values()
					
					prediction_values = np.clip(prediction.values(), 0, None)
					prediction_values = np.array(prediction_values).flatten()
					latest_prediction.append((time_of_observation + timedelta(hours=prediction_length), prediction_values[prediction_length-1]))
					latest_prediction = latest_prediction[-prediction_length-168:]
					actual_values.append((time_of_observation, y))
					actual_values = actual_values[-168:]
					#prediction = TimeSeries.from_times_and_values(prediction.time_index, prediction_values)
					#Get all prediction times and values

					#Get the latest acutal time and value
					actual_time_limit = series.time_index[-1]


					latest_prediction_limit = [Timestamp(item[0]) for item in latest_prediction]
					latest_prediction_limit = [item.tz_localize(None) for item in latest_prediction_limit]

		

					print(actual_time_limit in latest_prediction_limit)
					if len(latest_prediction) >0:

						if len(latest_prediction) > prediction_length:
							print(latest_prediction[-prediction_length:][0][1])		

						if len(latest_prediction) > prediction_length + 1 and latest_prediction[-prediction_length:][0][1] > 1 and (actual_time_limit in latest_prediction_limit):
	
							print("Switching on the light")	
							try:
								new_state = client_homeass.set_state(State(entity_id='input_boolean.test_switch', state='on'))
							except:
								print("no homeassistant connection")
						else:
							#print("Switching off the light")
							try:
								new_state = client_homeass.set_state(State(entity_id='input_boolean.test_switch', state='off'))
							except:
								print("no homeassistant connection")


				end_time_sleep = time.time()  # End time of the loop
				loop_time = end_time_sleep - start_time_sleep  # Time taken by the loop
				sleep_time = max(0.1 - loop_time, 0)  # Sleep time (100ms - loop time), but not less than 0
				time.sleep(sleep_time)
		current_time = stop_time
	training_running = 0
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8888)