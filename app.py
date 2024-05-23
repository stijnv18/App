from flask import Flask, render_template, jsonify, abort, request
from threading import Thread
import plotly
import plotly.graph_objs as go
import json
import pandas as pd
from river import time_series
import time
from river import metrics
from datetime import datetime, timedelta
import logging
from darts.models import TiDEModel
from joblib import load
from darts import TimeSeries
import numpy as np

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
		'actual_values': actual_value,
	})
@app.route('/latest_error', methods=['GET'])
def get_error():
    global error
    print(error)
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

 
	if model_name == 'SNARIMAX' and training_running != 1 or prev_prediction_length != prediction_length:
		training_running = 0
		time.sleep(0.1)
		Thread(target=run_SNARIMAX).start()
		prev_prediction_length = prediction_length
	elif model_name == 'Holtwinters' and training_running != 2 or prev_prediction_length != prediction_length:
		training_running = 0
		time.sleep(0.1)
		Thread(target=run_Holtwinters).start()
		prev_prediction_length = prediction_length
	elif model_name == 'TIDE' and training_running != 3 or prev_prediction_length != prediction_length:
		training_running = 0
		time.sleep(0.1)
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
	training_running = 1
 
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
    
	print("Running the model training snarimax...")
	df = pd.read_csv('merged_data.csv')
	df['DateTime'] = pd.to_datetime(df['DateTime'])

	X_train = df.drop('MeanEnergyConsumption', axis=1)
	y_train = df['MeanEnergyConsumption']

	# Get month and day of the week from the date time column
	X_train['Month'] = X_train['DateTime'].dt.month
	X_train['DayOfWeek'] = X_train['DateTime'].dt.dayofweek

	# Convert the training set back to DataFrame for the model training
	train_df = pd.concat([X_train, y_train], axis=1)

	model_without_exog = (time_series.SNARIMAX(p=1,d=0,q=1,sp=0,sd=1,sq=1,m=24))
 
	mae_without_exog = metrics.MAE()
	for i, (_, row) in enumerate(train_df.iterrows()):
		y = row['MeanEnergyConsumption']
		model_without_exog.learn_one(y)
		if i > 0:  # Skip the first observation
			forecast = model_without_exog.forecast(horizon=prediction_length)  # forecast 1 step ahead
			mae_without_exog.update(y, forecast[prediction_length-1])
			actual_values.append((row['DateTime'], y))
			actual_values = actual_values[-168:]
			# Save the latest prediction
			latest_prediction.append((row['DateTime'] + timedelta(hours=prediction_length), forecast[prediction_length-1]))
			latest_prediction = latest_prediction[-168-prediction_length:]
			error.append(mae_without_exog.get())
		if training_running != 1:
			break
		time.sleep(0.1)
	training_running = 0

def run_Holtwinters():
	global training_running
	global latest_prediction
	global actual_values
	training_running = 2
	
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
    
	print("Running the model training holt...")
	df = pd.read_csv('merged_data.csv')
	df['DateTime'] = pd.to_datetime(df['DateTime'])
	df = df.dropna()

	df['day_of_week'] = df['DateTime'].dt.dayofweek
	df['hour_of_day'] = df['DateTime'].dt.hour
	df['month'] = df['DateTime'].dt.month

	stream = iter(df.itertuples(index=False))
	stream = iter([(x._asdict(), y) for x, y in zip(df.drop('MeanEnergyConsumption', axis=1).itertuples(index=False), df['MeanEnergyConsumption'])])
	print(next(stream))

	model = time_series.HoltWinters(
		alpha=0.3,
		beta=0.1,
		gamma=0.5,
		seasonality=24,
		multiplicative=True,
	)

	metric = metrics.MAE()
	# Assuming 'df' is your DataFrame and 'MeanEnergyConsumption' is what you want to predict
	for i, (_, row) in enumerate(df.iterrows()):
	
		y = row['MeanEnergyConsumption']
		model.learn_one(y)

		# Predict the next point only after the model has been trained on 'seasonality' number of data points
		if i >= model.seasonality:
			prediction = model.forecast(horizon=prediction_length)
			actual_values.append((row['DateTime'], y))
			# Update the metric
			metric.update(y, prediction[prediction_length-1])
			# Save the latest prediction
			latest_prediction.append((row['DateTime'], prediction[prediction_length-1]))
			actual_values = actual_values[-168:]
			latest_prediction = latest_prediction[-168-prediction_length:]

			if training_running != 2:
				break
			time.sleep(0.1)
	training_running = 0
def run_TIDE():
	global training_running
	global latest_prediction
	global actual_values
	training_running = 3
	# Reset latest_prediction and actual_values
	latest_prediction = []
	actual_values = []
	print("Running the model TIDE...")
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

	# Now, for each new hour in the test set, make a prediction
	for index, row in df_test.iterrows():
		# Create a new TimeSeries object for the new data
		new_data = TimeSeries.from_times_and_values(pd.date_range(start=row['timestamp'], periods=1, freq='h'), [row['consumption']])
		
		# Append the new data to the series
		series_train = series_train.append(new_data)
		
		# Append the actual value and its timestamp to actual_values
		actual_values.append((row['timestamp'], row['consumption']))
		actual_values = actual_values[-168:]
		
		# Append the prediction and its timestamp to latest_prediction
		latest_prediction.append((row['timestamp'] + pd.Timedelta(hours=1), prediction.values()[0]))
		latest_prediction = latest_prediction[-168:]

	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8888)