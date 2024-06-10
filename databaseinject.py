import pandas as pd
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "A8N8FW0T9zLcF5Rx7hwZfAs10ADmNxqQtqi9t3c_L6s59RjeXbcZRXC2nqgb8RgmSBQzwcMJJxS7EenDP3-P1Q=="
org = "beheerder"
url = "http://localhost:8086"

write_client = InfluxDBClient(url=url, token=token, org=org)
write_api = write_client.write_api(write_options=SYNCHRONOUS)

# Read the CSV file
#df = pd.read_csv('merged_data.csv')

# Convert the timestamp column to datetime
#df['DateTime'] = pd.to_datetime(df['DateTime'])

# Write each row of the CSV file to the database
#for index, row in df.iterrows():
#    point = Point("measurement").time(time=row['DateTime'], write_precision=WritePrecision.NS)
#    for column in df.columns:
#        if column != 'DateTime':
#            point.field(column, row[column])
#    write_api.write(bucket="dataset", org=org, record=point)###

#write_api.__del__()

#query_api = write_client.query_api()
#query = """from(bucket: "dataset")
#|> range(start: -10y)
#|> filter(fn: (r) => r._measurement == "measurement")"""
#tables = query_api.query(query, org=org)

#for table in tables:
#    for record in table.records:
#        print(record)

import pandas as pd
#import influxdb_client
#from influxdb_client import InfluxDBClient, Point, WritePrecision
#from influxdb_client.client.write_api import SYNCHRONOUS

#token = "BKLgxb15c4FA6bE9TOBdyzqdJmD9gVbzJwWEco_el-wXuIdoFhGVs80LBoWbGSG6o5cv6yb4FVQ-BbLK_NmGeg=="
#org = "beheerder"
#url = "172.24.4.129:8086"

write_client = InfluxDBClient(url=url, token=token, org=org)
write_api = write_client.write_api(write_options=SYNCHRONOUS)

#Read the CSV file
df = pd.read_csv('customer_36v2.csv')

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Write each row of the CSV file to the database
for index, row in df.iterrows():
    point = Point("Solar").time(time=row['timestamp'], write_precision=WritePrecision.NS)
    for column in df.columns:
        if column != 'timestamp':
            point.field(column, row[column])
    write_api.write(bucket="dataset", org = org, record=point)

write_api.__del__()
    
query_api = write_client.query_api()
query = """from(bucket: "dataset")
# |> range(start: -15y)
# |> filter(fn: (r) => r._measurement == "Solar")"""
tables = query_api.query(query, org)

for table in tables:
    for record in table.records:
        print(record)