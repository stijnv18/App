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
					
				y = record.get_value()
				time_of_observation = record.get_time()
				print(f"Observation {time_of_observation}: {y}")
				data.append((time_of_observation, y))
				df = pd.DataFrame(data, columns=['timestamp', 'consumption'])	
				df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
				
				series = TimeSeries.from_dataframe(df, 'timestamp','consumption',fill_missing_dates=True, freq='h')
				if len(series) >= 48:
					prediction = model.predict(n=prediction_length, series=series)
					prediction_values = prediction.values()
					
					prediction_values = np.clip(prediction.values(), 0, None)
					prediction_values = np.array(prediction_values).flatten()
					#prediction = TimeSeries.from_times_and_values(prediction.time_index, prediction_values)


					latest_prediction.append((time_of_observation + timedelta(hours=prediction_length), prediction_values[prediction_length-1]))

					actual_values.append((time_of_observation, y))
					actual_values = actual_values[-168:]
					print(actual_values)

				time.sleep(0.1)
		current_time = stop_time
	training_running = 0