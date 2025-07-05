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

# %% [markdown]
# ---
# # Final Assignment: Robust Journey Planning
#
# In this notebook, we will use temporal and weather information about sbb transports to discover delay trends and suggest optimal route.

# %% [markdown]
# ---
# ## Problem Motivation
# Imagine you are a regular user of the public transport system, and you are checking the operator's schedule to meet your friends for a class reunion. The choices are:
#
# You could leave in 10mins, and arrive with enough time to spare for gossips before the reunion starts.
#
# You could leave now on a different route and arrive just in time for the reunion.
#
# Undoubtedly, if this is the only information available, most of us will opt for option 1.
#
# If we now tell you that option 1 carries a fifty percent chance of missing a connection and be late for the reunion. Whereas, option 2 is almost guaranteed to take you there on time. Would you still consider option 1?
#
# Probably not. However, most public transport applications will insist on the first option. This is because they are programmed to plan routes that offer the shortest travel times, without considering the risk factors.

# %%
groupName='U1'

# %% [markdown]
# ## Configure environment

# %%
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

import pwd
import numpy as np
import sys

from pyspark.sql import SparkSession
from random import randrange
import pyspark.sql.functions as F
#np.bool = np.bool_


username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'U1'

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

# %%
import base64 as b64
import json
import time
import re

def getUsername():
    payload = os.environ.get('EPFL_COM490_TOKEN').split('.')[1]
    payload=payload+'=' * (4 - len(payload) % 4)
    obj = json.loads(b64.urlsafe_b64decode(payload))
    if (time.time() > int(obj.get('exp')) - 3600):
        raise Exception('Your credentials have expired, please restart your Jupyter Hub server:'
                        'File>Hub Control Panel, Stop My Server, Start My Server.')
    time_left = int((obj.get('exp') - time.time())/3600)
    return obj.get('sub'), time_left


# %% [markdown]
# ## Connect to warehouse

# %%
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

# %%
import trino
from contextlib import closing
from urllib.parse import urlparse
from trino.dbapi import connect
from trino.auth import BasicAuthentication, JWTAuthentication

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

# %%
import pandas as pd

table = pd.read_sql(f"""SHOW TABLES IN {sharedNS}""", conn)

# %% [markdown]
# ## Start a Spark session

# %%
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

# %% [markdown]
# ### Get weather data (daily avg. precipitation)

# %%
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

stops_df = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops") # read stops with coordinates

def find_shortest_paths(start_stop_id, end_stop_id, confidence): # TODO: Replace with PATHFINDING ALGO
    start_stop = stops_df.filter(stops_df["stop_id"] == start_stop_id).select("stop_name").first()['stop_name']
    end_stop = stops_df.filter(stops_df["stop_id"] == end_stop_id).select("stop_name").first()['stop_name']
    print("Start: ", start_stop_id, " ", start_stop)
    print("End: ", end_stop_id, " ", end_stop)
    print("Confidence: ", confidence)
    
# Convert Spark DataFrame to Pandas
stops_pd = stops_df.select("stop_id", "stop_name").distinct().toPandas()
stops_pd = stops_pd.dropna().drop_duplicates(subset=["stop_id"])

# Create name-to-ID mapping
stop_map = dict(zip(stops_pd["stop_name"], stops_pd["stop_id"]))
stop_names_sorted = sorted(stop_map.keys())

# Start and End stop dropdowns
start_dropdown = widgets.Dropdown(
    options=stop_names_sorted,
    description="Start:",
    layout=widgets.Layout(width="100%")
)

end_dropdown = widgets.Dropdown(
    options=stop_names_sorted,
    description="End:",
    layout=widgets.Layout(width="100%")
)

# Confidence slider
confidence_slider = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Confidence:",
    readout_format=".2f",
    layout=widgets.Layout(width="100%")
)

# Button and Output
run_button = widgets.Button(description="Find Path", button_style="primary")
output = widgets.Output()

# Callback
def on_run_clicked(b):
    with output:
        clear_output()

        start_stop_name = start_dropdown.value
        end_stop_name = end_dropdown.value
        confidence = confidence_slider.value

        if start_stop_name == end_stop_name:
            print("Start and end stops must be different.")
            return

        start_stop_id = stop_map[start_stop_name]
        end_stop_id = stop_map[end_stop_name]

        print(f"Finding path from '{start_stop_name}' to '{end_stop_name}' with confidence {confidence:.2f}...")
        try:
            result = find_shortest_paths(start_stop_id, end_stop_id, confidence=confidence)
            display(result)
        except Exception as e:
            print("Error:", e)

# Attach callback
run_button.on_click(on_run_clicked)

# Display UI
ui = widgets.VBox([
    start_dropdown,
    end_dropdown,
    confidence_slider,
    run_button,
    output
])

display(ui)

# %%
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load stops from Spark
stops_df = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops")

def find_shortest_paths(start_stop_id, end_stop_id, confidence, rain, desired_time):
    start_stop = stops_df.filter(stops_df["stop_id"] == start_stop_id).select("stop_name").first()['stop_name']
    end_stop = stops_df.filter(stops_df["stop_id"] == end_stop_id).select("stop_name").first()['stop_name']
    print("Start:", start_stop_id, start_stop)
    print("End:", end_stop_id, end_stop)
    print("Confidence:", confidence)
    print("Rain:", rain)
    print("Desired Arrival Time:", desired_time)
    # TODO: Integrate actual pathfinding logic using parameters
    return None

# Convert Spark DataFrame to Pandas for UI
stops_pd = stops_df.select("stop_id", "stop_name").distinct().toPandas()
stops_pd = stops_pd.dropna().drop_duplicates(subset=["stop_id"])
stop_map = dict(zip(stops_pd["stop_name"], stops_pd["stop_id"]))
stop_names_sorted = sorted(stop_map.keys())

# UI widgets
start_dropdown = widgets.Dropdown(
    options=stop_names_sorted,
    description="Start:",
    layout=widgets.Layout(width="100%")
)

end_dropdown = widgets.Dropdown(
    options=stop_names_sorted,
    description="End:",
    layout=widgets.Layout(width="100%")
)

confidence_slider = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Confidence:",
    readout_format=".2f",
    layout=widgets.Layout(width="100%")
)

weather_toggle = widgets.ToggleButtons(
    options=[('No Rain', False), ('Rain', True)],
    description='Weather:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width="100%")
)

arrival_time_input = widgets.Text(
    value='08:00',
    description='Arrival Time:',
    placeholder='HH:mm',
    layout=widgets.Layout(width="100%")
)

run_button = widgets.Button(description="Find Path", button_style="primary")
output = widgets.Output()

# Callback function
def on_run_clicked(b):
    with output:
        clear_output()

        start_stop_name = start_dropdown.value
        end_stop_name = end_dropdown.value
        confidence = confidence_slider.value
        rain = weather_toggle.value
        desired_time = arrival_time_input.value

        if start_stop_name == end_stop_name:
            print("Start and end stops must be different.")
            return

        start_stop_id = stop_map[start_stop_name]
        end_stop_id = stop_map[end_stop_name]

        print(f"Finding path from '{start_stop_name}' to '{end_stop_name}'")
        print(f"Confidence: {confidence:.2f}, Rain: {rain}, Desired Arrival Time: {desired_time}")

        try:
            result = find_shortest_paths(
                start_stop_id,
                end_stop_id,
                confidence=confidence,
                rain=rain,
                desired_time=desired_time
            )
            if result is not None:
                display(result)
        except Exception as e:
            print("Error:", e)

# Link callback to button
run_button.on_click(on_run_clicked)

# Assemble UI
ui = widgets.VBox([
    start_dropdown,
    end_dropdown,
    confidence_slider,
    weather_toggle,
    arrival_time_input,
    run_button,
    output
])

# Display UI
display(ui)


# %% [markdown]
# # That's all, folks!

# %%
spark.stop()
