# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# spark.stop()

# %% [markdown]
# # Configure environment

# %% [markdown]
# ## Connect to warehouse

# %%
import base64 as b64
import json
import time
import re
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

import pwd
import numpy as np
import sys
import pandas as pd

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F
#np.bool = np.bool_

import trino
from contextlib import closing
from urllib.parse import urlparse
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication

groupName = 'U1'

def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left

username, validity_h = getUsername()
hadoopFS = os.environ.get('HADOOP_FS')
namespace = 'iceberg.' + username
sharedNS = 'iceberg.com490_iceberg'

if not re.search('[A-Z][0-9]', groupName):
    raise Exception('Invalid group name {groupName}')

print(f"you are: {username}")
print(f"credentials validity: {validity_h} hours left.")
print(f"shared namespace is: {sharedNS}")
print(f"your namespace is: {namespace}")
print(f"your group is: {groupName}")

trinoAuth = JWTAuthentication(os.environ.get('EPFL_COM490_TOKEN'))
trinoUrl  = urlparse(os.environ.get('TRINO_URL'))
Query=[]

print(f"Warehouse URL: {trinoUrl.scheme}://{trinoUrl.hostname}:{trinoUrl.port}/")

conn = connect(
    host=trinoUrl.hostname,
    port=trinoUrl.port,
    auth=trinoAuth,
    http_scheme=trinoUrl.scheme,
    verify=True
)

print('Connected!')

# %% [markdown]
# ## Start a Spark session

# %%
username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

spark = SparkSession\
            .builder\
            .appName(pwd.getpwuid(os.getuid()).pw_name)\
            .config('spark.ui.port', randrange(4040, 4440, 5))\
            .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path)) \
            .config('spark.jars', f'{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar')\
            .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')\
            .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')\
            .config('spark.sql.catalog.iceberg.type', 'hadoop')\
            .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/iceberg/')\
            .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')\
            .config('spark.sql.catalog.spark_catalog.type', 'hadoop')\
            .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')\
            .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')\
            .config('spark.eventLog.gcMetrics.youngGenerationGarbageCollectors', 'G1 Young Generation')\
            .config("spark.executor.memory", "6g")\
            .config("spark.executor.cores", "4")\
            .config("spark.executor.instances", "4")\
            .master('yarn')\
            .getOrCreate()

# %%
spark.sparkContext

# %%
# spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
# table = pd.read_sql(f"""SHOW TABLES IN {sharedNS}""", conn)
# table

# %%
# pd.read_sql(f"""DESCRIBE {sharedNS}.sbb_istdaten""", conn)

# %%
# # ! pip install geopandas

# %% [markdown]
# ### Import custom script

# %%
#IMPORT custom scripts file

import importlib.util
import os

# Define path to the file
file_path = os.path.abspath('../scripts/preprocessing.py')

# Load the module
spec = importlib.util.spec_from_file_location("utils", file_path)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)
print(preprocess.test())


# %%
# from pyspark.sql.functions import col
# df = spark.read\
#         .option("header", True) \
#         .parquet(f"{hadoopFS}/user/com-490/group/U1/schedule_v3")
# preprocess.edges(spark, df).show(10)

# %%
from pyspark.sql.functions import variance, avg, col, exp, stddev, coalesce,lit
import math
from scipy.stats import norm

def get_confidence_df(spark):
    df =  spark.read.parquet(f"{hadoopFS}/user/com-490/group/U1/multiclass_model.parquet")

    # Add expected delay column
    df = df.withColumn(
        "expected_delay",
        col("prob_small") * 120 +
        col("prob_medium") * 420 +
        col("prob_big") * 720
    )
    
    grouped_df = df.groupBy("stop_id", "dow", "hour").agg(
        avg("expected_delay").alias("mean_delay"),
        stddev("expected_delay").alias("std_dev_delay")
    )

    # Replace null std_dev_delay with 0
    grouped_df = grouped_df.withColumn(
        "std_dev_delay",
        coalesce(col("std_dev_delay"), lit(0))
    )
    
    grouped_df.orderBy(['dow','hour'])
    return grouped_df

def prob_delay_less_than(threshold, mean, std_dev):
    """
    Computes log(P(delay ≤ threshold)) assuming a normal distribution.
    Handles degenerate case where std_dev = 0.
    """
    if std_dev == 0:
        # Degenerate distribution: P(X ≤ threshold) = 1 if threshold ≥ mean, else 0
        return 0.0 if threshold >= mean else -math.inf  # log(1) = 0, log(0) = -inf
    
    z = (threshold - mean) / std_dev
    return math.log(norm.cdf(z))



# %%
timetable = spark.read \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/edges")
df =  get_confidence_df(spark)

# timetable.show(5)
df.filter((col('std_dev_delay').isNull())).show(5)

# %%
from pyspark.sql.functions import hour

df_joined = df.alias('b').join(
    timetable.alias('a'),
    on=(
        (col("a.to_stop") == col("b.stop_id")) &
        (col("a.dow") == col("b.dow")) &
        (hour(col("a.t_arrival")) == col("b.hour"))
    ),
    how='left'
).filter((col('std_dev_delay').isNull())).count()

print(df_joined)


# %% [markdown]
# ### Get weather data (daily avg. precipitation)

# %%
# # %%time
# daily_precip = preprocess.get_daily_precip(spark, month=None)

# %%
# # %%time
# hourly_weather = preprocess.get_hourly_weather(spark, month=None)

# %%
# weather_stations_with_region = preprocess.weather_stations_with_region(spark)

# %% [markdown]
# ### Get stops in a uuid and test

# %%
# # # %%time
# stops_geo_df = preprocess.get_stops_by_UUID(spark, uuid_arr)

# %% [markdown]
# ### Get stops schedule and delays, merge with weather
#
# Provided array of UUIDs create a table with delays and weather conditions.

# %%
# # %%time
# # ==================== Test ======================#
# uuid_arr = ['fa8f5deb-65a9-45cf-9510-3bc84a782842',
#             'a7a21b73-6ffe-4fbf-a635-6e2b961f3072',
#             'e168fd57-f57a-4075-a350-0dcfbb55147f'] #Extended Lausanne region
# df_final =  preprocess.get_delays_with_weather(spark, uuid_arr)

# %%
# #reading the final reasults
# df_final = spark.read \
#     .option("header", True) \
#     .parquet(f"{hadoopFS}/user/{username}/final/data/delays_with_weather").limit(10)
# df_final_pd = df_final.toPandas()
# df_final_pd.head()

# %% [markdown]
# ### Select the regions of interest and press run to get the historical data and trip data (takes 3-4 min.)

# %%
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go

# Load data
df_geo = pd.read_sql(f"""
    SELECT name, region, wkb_geometry, uuid
    FROM {sharedNS}.geo
""", conn)

search_box = widgets.Text(
    placeholder="Search city...",
    description="Search:",
    layout=widgets.Layout(width="100%")
)

city_checkboxes_container = widgets.VBox(layout=widgets.Layout(width="100%", height="150px", overflow="auto"))
uuid_checkboxes_container = widgets.VBox(layout=widgets.Layout(width="100%", height="150px", overflow="auto"))

run_button = widgets.Button(description="Run", button_style="success")
map_button = widgets.Button(description="Display Map", button_style="info")

output = widgets.Output()
city_checkbox_widgets = []
have_run = False
saveas_hist = "delays_with_weather_extended"
# saveas_trips = "trips_with_sequence"
saveas_trips = "schedule_v3"

#update uuid checkboxes when cities are selected 
def update_uuid_checkboxes():
    selected_cities = [
        cb.description for cb in city_checkbox_widgets
        if cb.value
    ]
    uuid_widgets = []

    for city in selected_cities:
        city_uuids = df_geo[df_geo["name"] == city]["uuid"].drop_duplicates()
        uuid_widgets.append(widgets.Label(f"UUIDs for {city}:"))
        uuid_widgets.extend([
            widgets.Checkbox(value=True, description=str(u)) for u in city_uuids
        ])

    uuid_checkboxes_container.children = uuid_widgets


# Used some help from ChatGPT for this part
# Global state to persist selections
selected_cities_state = {}

def city_checkbox_callback(change):
    if change["name"] == "value":
        selected_cities_state[change.owner.description] = change["new"]
        update_uuid_checkboxes()

def populate_city_checkboxes(search_text=""):
    global city_checkbox_widgets
    all_names = sorted(df_geo["name"].unique())
    selected = [n for n in all_names if selected_cities_state.get(n, False)]
    matching = [n for n in all_names 
                if n.lower().find(search_text.lower()) >= 0 and n not in selected]
    city_checkbox_widgets = []
    for name in selected + matching:
        cb = widgets.Checkbox(
            value=selected_cities_state.get(name, False),
            description=name
        )
        cb.observe(city_checkbox_callback, names="value")
        city_checkbox_widgets.append(cb)

    city_checkboxes_container.children = city_checkbox_widgets
    update_uuid_checkboxes()

# Search box change callback 
def on_search_change(change):
    populate_city_checkboxes(change["new"])

search_box.observe(on_search_change, names="value")

def on_run_clicked(b):
    with output:
        clear_output()

        selected_uuids = [
            cb.description for cb in uuid_checkboxes_container.children
            if isinstance(cb, widgets.Checkbox) and cb.value
        ]

        if not selected_uuids:
            print("No UUIDs selected.")
            return

        print("Running get_delays_with_weather with UUIDs:")
        print(selected_uuids)

        try:
            global df_final
            stops_geo_df = preprocess.get_stops_by_UUID(spark, selected_uuids)
            df_final = preprocess.get_delays_with_weather_ext(spark, saveas=saveas_hist)
            print(f" Historical data saved at {hadoopFS}/user/com-490/group/U1/{saveas_hist}: ")
            display(df_final.limit(5).show())
            
            df_trips = preprocess.get_trips_v3(spark, saveas=saveas_trips)
            print(f" Schedule data saved at {hadoopFS}/user/com-490/group/U1/{saveas_trips}: ")
            display(df_trips.limit(5).show())

            df_edges = preprocess.edges(spark, df_trips)
            print(f" Edges saved at {hadoopFS}/user/com-490/group/U1/edges")
            display(df_edges.limit(5).show())

            have_run = True
        
        except Exception as e:
            print("Error:", e)

# --- Map button logic ---
def on_map_clicked(b):
    with output:
        clear_output()
        try:
            if have_run: # if you have run the calculation just read the output
                stops_geo_df = spark.read.option("header", True)\
                    .parquet(f"{hadoopFS}/user/com-490/group/U1/stops")
            if not have_run: #otherwise calculate stops
                selected_uuids = [
                    cb.description for cb in uuid_checkboxes_container.children
                    if isinstance(cb, widgets.Checkbox) and cb.value
                ]
        
                if not selected_uuids:
                    print("No UUIDs selected.")
                    return
                    
                print("Finding stops")
                stops_geo_df = preprocess.get_stops_by_UUID(spark, selected_uuids).distinct()
                
            stops_geo = stops_geo_df.toPandas()
            stops_geo["stop_lon"] = pd.to_numeric(stops_geo['stop_lon'], errors="coerce")
            stops_geo["stop_lat"] = pd.to_numeric(stops_geo['stop_lat'], errors="coerce")

            max_lat = stops_geo['stop_lat'].max()
            min_lat = stops_geo['stop_lat'].min()
            max_lon = stops_geo['stop_lon'].max()
            min_lon = stops_geo['stop_lon'].min()

            center_lat = (max_lat + min_lat) / 2
            center_lon = (max_lon + min_lon) / 2

            hover_text = (
                "<b>Stop Name:</b> " + stops_geo['stop_name'] + "<br>" +
                "<b>Stop ID:</b> " + stops_geo['stop_id'].astype(str) + "<br>" +
                "<b>Latitude:</b> " + stops_geo['stop_lat'].round(6).astype(str) + "<br>" +
                "<b>Longitude:</b> " + stops_geo['stop_lon'].round(6).astype(str) + "<br>"
            )

            fig = go.Figure(go.Scattermapbox(
                lat=stops_geo['stop_lat'],
                lon=stops_geo['stop_lon'],
                mode='markers',
                marker=dict(size=6, color='red'),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ))

            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=10.5
                ),
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                showlegend=False
            )

            fig.show()
        except Exception as e:
            print("Error displaying map:", e)

# Attach button callbacks
run_button.on_click(on_run_clicked)
map_button.on_click(on_map_clicked)

# Initial render
populate_city_checkboxes()

# Layout
ui = widgets.HBox([
    widgets.VBox([search_box, city_checkboxes_container], layout=widgets.Layout(width="45%")),
    uuid_checkboxes_container
])

display(ui, widgets.HBox([run_button, map_button]), output)


# %%
# # Example reading of tables
# spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops") # read stops with coordinates
# df = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/delays_with_weather_extended").select("trip_id", "stop_id", "stop_name", "dow", col("arr_time").alias("planned_arr"), col("dep_time").alias("planned_dep"),col("avg_precip").alias("mean_delay"),col("avg_wspd").alias("std_delay"))#.show(10, truncate=False) # read historical data
# spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/delays_with_weather_extended").limit(5).show()
# spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/schedule").filter((col('trip_id')=='85:151:1004')& (col('dow')==4)).show(100,truncate=False) # read trip information yellow
# preprocess.get_trips_v2(spark)

# %%
## TODO - import
from pyspark.sql.functions import udf, from_unixtime, dayofmonth, hour, minute, dayofweek
from pyspark.sql.types import StringType
import pytz #timezones
from datetime import datetime
from pyspark.sql.functions import col, explode, year, month, when, avg, first
from pyspark.sql.functions import sum as spark_sum, max as spark_max

import math
from pyspark.sql.functions import array, explode, lit, col, struct, when
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lower, trim, regexp_replace

import geopandas as gpd
from shapely.geometry import Point
import shapely.wkb


from pyspark.sql.window import Window
from pyspark.sql.functions import lag, unix_timestamp, when
from pyspark.sql.functions import year, month, dayofmonth, date_format
from pyspark.sql.functions import row_number, array, explode, lit, col, struct

# from pyspark.sql.window import Window

# %%
df = spark.read.option("header", True)\
                .parquet(f"{hadoopFS}/user/com-490/group/U1/schedule_v3")

# Define window spec to order rows within duplicates
window = Window.partitionBy('route_id','arrival_time','departure_time','stop_id','stop_sequence','dow').orderBy("trip_id")

# Add row number
df_with_rank = df.withColumn("rn", F.row_number().over(window))

# Keep only the first occurrence
df= df_with_rank.filter("rn = 1").drop("rn")

# df = df.dropDuplicates(['route_id','arrival_time','departure_time','stop_id','stop_sequence','dow'])
    # Add "instance" to account for repeating trip_id

df.orderBy(['arrival_time', 'stop_sequence'])\
.filter((col('dow')==3))\
.filter(col('stop_id')== '8595939') \
.orderBy(['stop_sequence', 'arrival_time'])\
.filter(hour(col('arrival_time'))>12)\
.show(10,truncate=False)

# %%
from pyspark.sql import functions as F
from pyspark.sql import Window

window_spec1 = Window.partitionBy("route_id", "stop_sequence") \
                    .orderBy("arrival_time")

# Use rank() to order them chronologically with smallest = 1
df_with_instance = df.withColumn("instance", F.rank().over(window_spec1))
df = df_with_instance

# df.filter(col("from_stop") == '8595939') \
#     .filter(col("to_stop") == '8595937') \
#     .filter(col("dow") == 3) \
#     .orderBy("t_arrival").show(300,truncate=False)

 # %%
 # Calculate pairs
window_spec = Window.partitionBy("trip_id", "instance").orderBy("stop_sequence")

# Add lagged stop and arrival time
df_with_lag = df.withColumn("prev_stop", lag("stop_id").over(window_spec)) \
                .withColumn("prev_time", lag("arrival_time").over(window_spec)) \
                .withColumn("prev_sequence", lag("stop_sequence").over(window_spec))

# Filter: ensure stop_name != prev_stop and stop_sequence != prev_sequence
df_filtered = df_with_lag.filter((col("stop_id") != col("prev_stop")) & (col("stop_sequence") != col("prev_sequence")))

result = df_filtered.withColumn(
    "travel_time",
    unix_timestamp("arrival_time", "HH:mm:ss") - unix_timestamp("prev_time", "HH:mm:ss")
).select(
    "route_id",
    "trip_id",
    col("prev_stop").alias("from_stop"),
    col("stop_id").alias("to_stop"),
    col("travel_time").alias("T_nominal"),
    col("arrival_time").alias("t_arrival"),
    "dow",
).withColumn(
    "T_nominal",
    when(col("T_nominal") == 0, 40).otherwise(col("T_nominal")) # default for travel time < 0
).filter(
    (F.col("from_stop").isNotNull()) &
    (F.col("travel_time") >= 0) 
)
# result=result.dropDuplicates(['route_id','from_stop','to_stop','T_nominal','t_arrival','dow'])
result.show(10, truncate=False)

# %%
# spark.stop()
