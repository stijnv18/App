if (graphs) {
	Plotly.newPlot("graph", graphs.data, graphs.layout);
}

const training_url = "/start_training";

const eBtn_start_model_snarimax = document.querySelector("#start_model_snarimax");
const eBtn_start_model_holt = document.querySelector("#start_model_holt");
const eBtn_start_model_tide = document.querySelector("#start_model_tide");

const startModel = (url, model) => {
	const predictionLength = document.querySelector("#prediction_length").value;
	Plotly.newPlot("graph", [], {}); // Clear the existing graph
	fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json"
		},
		body: JSON.stringify({ "model": model, "prediction_length": predictionLength }),
	})
		.then(response => response.json())
			.then(data => console.log(data))
			.catch(error => console.error("Error:", error));

	setInterval(() => { // Start updating the graph every second
		fetch("/latest_prediction")
			.then(response => response.json())
				.then(data => {
					console.log(data); // Log the data
					let predictionData = [{
						x: data.prediction_dates,
						y: data.prediction_values,
						name: "Predictions",
						line: { color: "blue" }
					}];
					let actualData = [{
					x: data.actual_dates,
					y: data.actual_values,
					name: "Actual values",
					line: { color: "red" }
					}];
					let newData = predictionData.concat(actualData);
					if (graphs) {
						Plotly.newPlot("graph", newData, graphs.layout);
					} else {
						Plotly.newPlot("graph", newData, { yaxis: { range: [0, Math.max(...data.prediction_values, ...data.actual_values) + 0.1] } });
					}
			})
			.catch(error => console.error("Error:", error));
	}, 100);

	setInterval(() => {
		fetch("/latest_error")
			.then(response => response.json())
				.then(data => {
					console.log(data);  // Log the data
					let errorData = [{
						x: data.error_dates,
						y: data.error_values,
						name: "Error",
						line: { color: "red" }
					}];
					Plotly.newPlot("error_graph", errorData, { yaxis: { range: [0, Math.max(...data.error_values) + 0.1] } });
				})
				.catch(error => console.error("Error:", error));
	}, 1000);
};

eBtn_start_model_snarimax.addEventListener("click", () => startModel(training_url, "SNARIMAX"));
eBtn_start_model_holt.addEventListener("click", () => startModel(training_url, "Holtwinters"));
eBtn_start_model_tide.addEventListener("click", () => startModel(training_url, "TIDE"));
