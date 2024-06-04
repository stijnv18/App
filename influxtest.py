from influxdb_client import InfluxDBClient, Point, Dialect
from influxdb_client.client.flux_table import FluxTable
from influxdb_client.client.write_api import SYNCHRONOUS

# Connect to the InfluxDB server
host = '172.24.4.130:8086'
token = "BKLgxb15c4FA6bE9TOBdyzqdJmD9gVbzJwWEco_el-wXuIdoFhGVs80LBoWbGSG6o5cv6yb4FVQ-BbLK_NmGeg=="
org = "beheerder"
bucket = "dataset"
client = InfluxDBClient(url=host, token=token, org=org)

# Query the data from your bucket
query = """from(bucket: "dataset")
|> range(start: 2011-11-23T09:00:00Z, stop: 2014-02-28T00:00:00Z)
|> filter(fn: (r) => r["_measurement"] == "measurement")
|> filter(fn: (r) => r["_field"] == "MeanEnergyConsumption")"""

tables = client.query_api().query(query, org=org)

# Extract the data from the FluxTable objects
data = []
for table in tables:
	for record in table.records:
		print((record.get_time(), record.get_value()))
