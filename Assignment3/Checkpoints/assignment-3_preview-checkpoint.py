# -*- coding: utf-8 -*-
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
# # DSLab Homework3 - Uncovering Public Transport Conditions using SBB data
#
# In this notebook, we will use temporal information about sbb transports to discover delay trends. 
#
# ## Hand-in Instructions:
#
# - __Due: *.05.2025~ 23:59:59 CET__
# - your project must be private
# - add necessary comments and discussion to make your codes readable
# - make sure that your code is runnable

# %% [markdown]
# ---
# <div style="font-size: 100%" class="alert alert-block alert-info">
#     <b>ℹ️  Fair Cluster Usage:</b> As there are many of you working with the cluster, we encourage you to:
#     <ul>
#         <li>Whenever possible, prototype your queries on small data samples or partitions before running them on whole datasets</li>
#         <li>Save intermediate data in your HDFS home folder <b>f"/user/{username}/..."</b></li>
#         <li>Convert the data to an efficient storage format when this is an option</li>
#         <li>Use spark <em>cache()</em> and <em>persist()</em> methods wisely to reuse intermediate results</li>
#     </ul>
# </div>
#
# For instance:
#
# ```python
#     # Read a subset of the original dataset into a spark DataFrame
#     df_sample = spark.read.csv(f'/data/com-490/csv/{table}', header=True).sample(0.01)
#     
#     # Save DataFrame sample
#     df_sample.write.parquet(f'/user/{username}/assignment-3/{sample_table}.parquet', mode='overwrite')
#
#     # ...
#     df_sample = spark.read.parquet(f'/user/{username}/assignment-3/{sample_table}.parquet')
# ```
#
# Note however, that due to Spark partitioning, and parallel writing, the original order may not be preserved when saving to files.

# %% [markdown]
# ---
# ## Start a spark Session environment

# %% [markdown]
# We provide the `username` and `hadoopFS` as Python variables accessible in both environments. You can use them to enhance the portability of your code, as demonstrated in the following Spark SQL command. Additionally, it's worth noting that you can execute Iceberg SQL commands directly from Spark on the Iceberg data.

# %%
import os
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
# Be nice to others - remember to add a cell `spark.stop()` at the end of your notebook.

# %% [markdown]
# ---
# For your convenience, the Spark sessions is configured to use a default _spark_catalog_ and our _iceberg_ catalog where the SBB data is located.
#
# Execute the code below to create your schema _spark_catalog.{username}_ and set it as your default, and verify the presence of the Iceberg SBB tables.

# %%
# %%time
spark.sql(f'CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}')

# %%
# %%time
spark.sql(f'USE spark_catalog.{username}')

# %%
# %%time
spark.sql(f'SHOW CATALOGS').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW SCHEMAS IN iceberg').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
# %%time
spark.sql(f'SHOW TABLES IN iceberg.geo').show(truncate=False)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.stop_times LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.trips LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.routes LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.calendar LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.calendar_dates LIMIT 1').show(truncate=False,vertical=True)

# %%
# %%time
spark.sql(f'SELECT * FROM iceberg.sbb.istdaten LIMIT 1').show(truncate=False,vertical=True)

# %% [markdown]
# 💡 Notes:
# - Do not hesitate to create temporary views out of tabular, data stored on file or from SQL queries. That will make your code reusable, and easier to read
# - You can convert spark DataFrames to pandas DataFrames inside the spark driver to process the results. But Only do this for small result sets, otherwise your spark driver will run OOM.

# %%
# %%time
spark.read.options(header=True).csv(f'/data/com-490/csv/weather_stations').withColumns({
      'lat': F.col('lat').cast('double'),
      'lon': F.col('lon').cast('double'),
    }).createOrReplaceTempView("weather_stations")

# %%
spark.sql(f'SELECT * FROM weather_stations').printSchema()

# %%
# %%time
# Note, that this would also works: SHOW TABLES IN global_temp
spark.sql(f'SHOW TABLES').show(truncate=False)

# %%
spark.table("weather_stations").show()

# %%
spark.sql(f'SELECT * FROM weather_stations LIMIT 5').toPandas()

# %%
# spark.sql(f'DROP VIEW weather_stations')

# %% [markdown]
# # OVERALL COMMENTS ABOUT OUR CODE

# %% [markdown]
# **Caching**:
# 	most of part2 is intermediate steps df are used again in the next step, (+ materialization/visualization)
# 	most of the intermediate steps are pointless to cache.
# 	The once we do have already gone through filtering f.ex, part c where we look for a specific tripID. This is a much lighter subset, and it is defendable to cache actual, predicted and pivoted_df. Specially pivoted that is going to be used more 	in part3. 
#
#
# **Hadoop**:
# 	Running the entire notebook can take about half-hour (longer when busy server). 
# 	For convenience, most of the time/resource-consuming data frames are precomputed and saved as parquet files on HDFS under the namespace of "umansky". The provided preview notebook allows to load necessary df avoiding expensive queries and 	visualise results.
# 	
#
# **Feature Engineering Strategies**:
#
# Combine weather with delay dataframes with joining over day. This allows us to trace delays any day and attribute them to the weather. 
# Given months of interest get_daily_precip_month_pd computed daily averages
# Using pivoted_delay_df (in the code we use the "good" tripId, but this can in principle be extended to any tripID with control queries to check consistency of stops). delay_flags_df uses pivoted_df and cleans up NaNs and assign binary system to delays. 
# For months of interest compute daily average Precipitation (using get_daily_precip_month_pd) and inner join with Pivoted delay over the  
# Using this 2 a merge can be created and feature extraction is possible
#
# The Delay balance (for the tripID) is 300 (>5min) to 4.7k total binaries. This is after NaN hunting. Feature selection needs to be carefull when trying to classify a minority of large delays. 

# %% [markdown]
# ---
# ## PART I: First Steps with Spark DataFrames using Weather Data (20 points)
#
# We copied several years of historical weather data downloaded from [Wunderground](https://www.wunderground.com).
#
# We made this weather data available because of its strong impact on our daily lives, including in areas such as transportation.
#
# Let's see if we can see any trends in this data.

# %% [markdown]
# ### I.a Restructure the weather history - 2/20
#
# Load the JSON data from HDFS _/data/com-490/json/weather_history/_ into a Spark DataFrame using the appropriate method from the SparkSession.
#
# Restructure the data so that the schema matches this output, where the field _observation_ is a **single** record of weather meaasurements at a given point in time:
# ```
# root
#  |-- metadata: struct (nullable = true)
#  |    |-- expire_time_gmt: long (nullable = true)
#  |    |-- language: string (nullable = true)
#  |    |-- location_id: string (nullable = true)
#  |    |-- status_code: long (nullable = true)
#  |    |-- transaction_id: string (nullable = true)
#  |    |-- units: string (nullable = true)
#  |    |-- version: string (nullable = true)
#  |-- site: string (nullable = true)
#  |-- year: integer (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- observation: struct (nullable = true)
#  |    |-- blunt_phrase: string (nullable = true)
#  |    |-- class: string (nullable = true)
#  |    |-- clds: string (nullable = true)
#  |    |-- ...
# ```
#
# 💡 Notes:
# - The JSON data is multilines
# - Use functions learned during the exercises.
# - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html

# %%
## Imports
from pyspark.sql.functions import col, explode, year, month

# %%
# %%time
## Read weather history data and convert to structure as described above

# Read the JSON
df_weather_raw = spark.read.option("multiline", True).json(f"{hadoopFS}/data/com-490/json/weather_history/")

# %%
# Explode observations
df_weather_exploded = df_weather_raw.withColumn("observation", explode(col("observations")))

# %%
# Select fields
json_df = df_weather_exploded.select(
    col("metadata"),
    col("site"),
    year((F.col("observation.valid_time_gmt") * 1000).cast("timestamp")).alias("year"),
    month((F.col("observation.valid_time_gmt") * 1000).cast("timestamp")).alias("month"),
    col("observation")
)

# %%
# Create temp view
json_df.createOrReplaceTempView("weather_history")

# %%
# Check
json_df.printSchema()

# %% [markdown]
# ---

# %% [markdown]
# __User-defined and builtin functions__
#
# In Spark Dataframes you can create your own user defined functions for your SQL commands.
#
# So, for example, if we wanted to make a user-defined python function that returns a string value in lowercase, we could do something like this:

# %%
import pyspark.sql.functions as F


# %%
@F.udf
def lowercase(text):
    """Convert text to lowercase"""
    return text.lower()


# %% [markdown]
# The `@F.udf` is a "decorator" -- and in this case is equivalent to:
#
# ```python
# def lowercase(text):
#     return text.lower()
#     
# lowercase = F.udf(lowercase)
# ```
#
# It basically takes our function and adds to its functionality. In this case, it registers our function as a pyspark dataframe user-defined function (UDF).
#
# Using these UDFs is very straightforward and analogous to other Spark dataframe operations. For example:

# %%
# %%time
json_df.select(json_df.site,lowercase(json_df.site).alias('lowercase_site')).show(n=5)

# %% [markdown]
# The DataFrame API already includes many built-in functions, including the function for converting strings to lowercase.
# Other handy built-in dataframe functions include functions for transforming date and time fields.
#
# Note that the functions can be combined. Consider the following dataframe and its transformation:
#
# ```
# from pyspark.sql import Row
#
# # create a sample dataframe with one column "degrees" going from 0 to 180
# test_df = spark.createDataFrame(spark.sparkContext.range(180).map(lambda x: Row(degrees=x)), ['degrees'])
#
# # define a function "sin_rad" that first converts degrees to radians and then takes the sine using built-in functions
# sin_rad = F.sin(F.radians(test_df.degrees))
#
# # show the result
# test_df.select(sin_rad).show()
# ```
#
# Refs:
# - [Dataframe API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)
# - [GroupedData API](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.html)

# %% [markdown]
# ---
# ### I.b Processing timestamps - 2/20

# %% [markdown]
# Use UDF to organize the weather data by their timestamps.
#
# Check out the [Spark python API documentation](https://spark.apache.org/docs/latest/api/python/index.html). Look for the `sql` section and find the listing of `sql.functions`. Using either Spark built-in functions or their equivalent SQL expressions, convert the GMT _observation.valid_time_gmt_ from a string format to a date format _YYYY-mm-dd HH:MM:SS_, and extract the year, month, day, hour and minute components.
#
# A sample of the output should be similar to the one shown below:
#
# ```
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |site|observation.valid_time_gmt|observation_time_tz|year|month|dayofmonth|hour|minute|
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |LSGC|1672528800                |2023-01-01 00:20:00|2023|1    |1         |0   |20    |
# |LSGC|1672530600                |2023-01-01 00:50:00|2023|1    |1         |0   |50    |
# |LSGC|1672532400                |2023-01-01 01:20:00|2023|1    |1         |1   |20    |
# |LSGC|1672534200                |2023-01-01 01:50:00|2023|1    |1         |1   |50    |
# |LSGC|1672536000                |2023-01-01 02:20:00|2023|1    |1         |2   |20    |
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# ```
#
# ⚠️ When working with dates and timestamps, always be mindful of timezones and Daylight Saving Times (DST). Verify that the time information is consistent for the first few hours of January 1st and DST changes. Note that the weather's year, month, and day fields are based on the local timezone, i.e. _'Europe/Zurich'_. Timestamps represent the number of seconds since _1970-01-01 00:00:00 UTC_. However, Spark may interpret timezones differently depending on the function used and the local timezone of the spark cluster, which can lead to inconsistencies, i.e. you may instead end up with the wrong values, like:
#
# ```
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |site|observation.valid_time_gmt|observation_time_tz|year|month|dayofmonth|hour|minute|
# +----+--------------------------+-------------------+----+-----+----------+----+------+
# |LSGC|1672528800                |2022-12-31 23:20:00|2023|1    |31        |23  |20    |
# ```

# %%
## Import
from pyspark.sql.functions import udf, from_unixtime, dayofmonth, hour, minute
from pyspark.sql.types import StringType
import pytz #timezones
from datetime import datetime


# %%
# %%time
## Write code to convert to structure as described above
# Define UDF
@udf(StringType())
def gmt_to_local_time(unix_seconds):
    if unix_seconds is None:
        return None
    utc_dt = datetime.utcfromtimestamp(unix_seconds).replace(tzinfo=pytz.utc)
    zurich_dt = utc_dt.astimezone(pytz.timezone('Europe/Zurich'))
    return zurich_dt.strftime('%Y-%m-%d %H:%M:%S')


# %%
df_weather_time = df_weather_exploded.select(
    col("site"),
    col("observation.valid_time_gmt"),
    gmt_to_local_time(col("observation.valid_time_gmt")).alias("observation_time_tz")
)

df_weather_time = df_weather_time.withColumn("year", year(col("observation_time_tz").cast("timestamp"))) \
                                 .withColumn("month", month(col("observation_time_tz").cast("timestamp"))) \
                                 .withColumn("dayofmonth", dayofmonth(col("observation_time_tz").cast("timestamp"))) \
                                 .withColumn("hour", hour(col("observation_time_tz").cast("timestamp"))) \
                                 .withColumn("minute", minute(col("observation_time_tz").cast("timestamp")))

# %%
# Show
df_weather_time.show(5, truncate=False)

# %% [markdown]
# ### I.c Transform the data - 4/20
#
# Modify the DataFrame to add the weather measurements column and save the transformation into a _weather_df_ table: 
#
# The Spark Dataframe weather_df must includes the columns _month_, _dayofmonth_, _hour_ and _minutes_, calculated from _observation.valid_time_gmt_ as before, and:
#
# - It contains all (and only) the data from a full year of data, that is if there is only data in the second part of _2022_ then you shouldn't consider any data from _2022_. However, few missing values and gaps in the data are acceptable.
# - It contains a subset of weather information columns from the original data as show in the example below
# - A row should be similar to:
#
# ```
#  site                       | LSGC                
#  observation.valid_time_gmt | 1672528800          
#  observation_time_tz        | 2023-01-01 00:20:00 
#  valid_time_gmt             | 1672528800          
#  clds                       | CLR                 
#  day_ind                    | N                   
#  dewPt                      | 6                   
#  feels_like                 | 13                  
#  gust                       | 48                  
#  heat_index                 | 13                  
#  obs_name                   | La Chaux-De-Fonds   
#  precip_hrly                | NULL                
#  precip_total               | NULL                
#  pressure                   | 904.36              
#  rh                         | 63                  
#  temp                       | 13                  
#  uv_desc                    | Low                 
#  uv_index                   | 0                   
#  vis                        | 9.0                 
#  wc                         | 13                  
#  wdir                       | 240                 
#  wdir_cardinal              | WSW                 
#  wspd                       | 30                  
#  wx_phrase                  | Fair                
#  year                       | 2023                
#  month                      | 1                   
#  dayofmonth                 | 1                   
#  hour                       | 0                   
#  minute                     | 20                  
# ```
#
# __Note:__ 
# - [pyspark.sql.DataFrame](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html)

# %%
# Apply timezone UDF
df_weather_with_time = df_weather_exploded.select(
    col("site"),
    col("observation.valid_time_gmt"),
    gmt_to_local_time(col("observation.valid_time_gmt")).alias("observation_time_tz"),
    col("observation.clds"),
    col("observation.day_ind"),
    col("observation.dewPt"),
    col("observation.feels_like"),
    col("observation.gust"),
    col("observation.heat_index"),
    col("observation.obs_name"),
    col("observation.precip_hrly"),
    col("observation.precip_total"),
    col("observation.pressure"),
    col("observation.rh"),
    col("observation.temp"),
    col("observation.uv_desc"),
    col("observation.uv_index"),
    col("observation.vis"),
    col("observation.wc"),
    col("observation.wdir"),
    col("observation.wdir_cardinal"),
    col("observation.wspd"),
    col("observation.wx_phrase")
)

# %%
# %%time
# Add extracted fields: year, month, dayofmonth, hour, minute
df_weather_with_time = df_weather_with_time.withColumn("year", year(col("observation_time_tz").cast("timestamp"))) \
                                           .withColumn("month", month(col("observation_time_tz").cast("timestamp"))) \
                                           .withColumn("dayofmonth", dayofmonth(col("observation_time_tz").cast("timestamp"))) \
                                           .withColumn("hour", hour(col("observation_time_tz").cast("timestamp"))) \
                                           .withColumn("minute", minute(col("observation_time_tz").cast("timestamp")))

# %%
weather_df = df_weather_with_time.filter(col("year") == 2023)

# %%
# Select a row (for example the first one)
row = weather_df.limit(1).collect()[0]

# Print nicely
for field in weather_df.columns:
    print(f"{field:<28} | {row[field]}")


# %%


# %% [markdown]
# ### I.d Top average monthly precipitation per site - 4/20
#
# We will now use the Spark DataFrame group by aggregations to compute monthly aggregations.
#
# The _Spark.DataFrame.groupBy_ does not return another DataFrame, but a _GroupedData_ object instead. This object extends the DataFrame with methods that allow you to do various transformations and aggregations on the data in each group of rows. 
#
# Conceptually the procedure is a lot like this:
# ![groupby](./figs/sgCn1.jpg)
#
#
# The column set used for the _groupBy_ is the _key_ - and it can be a list of column keys, such as _groupby('key1','key2',...)_ - all rows in a _GroupedData_ have the same key, and various aggregation functions can be applied on them to generate a transformed DataFrame. In the above example, the aggregation function is a simple `sum`.

# %% [markdown]
# **Question:**
#
# Apply a group by on the _weather_df_ created earlier to compute the monthly precipitation on (site,month of year). Find the sites and months that have the highest total precipitation: sort the site and month in decreasing order of monthly precipitation and show the 10 top ones.
#
# Name the spark DataFrame _avg_monthly_precip_df_.
#
# The schema of the table is, at a minimum:
# ```
# root
#  |-- site: string (nullable = true)
#  |-- month: integer (nullable = true)
#  |-- avg_total_precip: double (nullable = true)
# ```
#
# Note:
# * A site may report multiple hourly precipitation measurements (precip_hrly) within a single hour. To prevent adding up hourly measurement for the same hour, you should compute an aggregated values observed at each site within the same hour.
# * Some weather stations do not report the hourly  precipitation, they will be shown as _(null)_

# %%
# imports
from pyspark.sql.functions import sum as spark_sum, max as spark_max

# %%
# Aggregate per hour to prevent multiple values within the same hour
hourly_precip_df = weather_df.groupBy("site", "year", "month", "dayofmonth", "hour")\
                             .agg(spark_max("precip_hrly").alias("max_precip_hrly"))

# Group by (site, month) to compute monthly totals
avg_monthly_precip_df = hourly_precip_df.groupBy("site", "month").agg(spark_sum("max_precip_hrly").alias("avg_total_precip"))

# Sort in descending order of total precipitation and take top 10
avg_monthly_precip_df = avg_monthly_precip_df.orderBy(col("avg_total_precip").desc())

# Print schema to confirm structure
avg_monthly_precip_df.printSchema()

# %% [markdown]
# Convert the _avg_monthly_precip_df_ Spark DataFrames to a Pandas DataFrame and **visualize the results** in the notebook.
#
# We are not looking for perfection, we just want to verify that your results are generally accurate. However, feel free to unleash your creativity and come up with a visualization that you find insightful.
#
# 💡 Do not hesitate to take advantage of the _weather_station_ table if you would like to include geospation information in your analysis. The data is available in _/data/com-490/csv/weather_stations_, see also examples at the beginning of this notebook. Additional details about the stations can be found [here](https://metar-taf.com/?c=464582.65369.10).

# %%
avg_monthly_precip_pd = avg_monthly_precip_df.toPandas()

# Sort by precipitation (just to be sure)
avg_monthly_precip_pd = avg_monthly_precip_pd.sort_values("avg_total_precip", ascending=False).dropna(subset=["avg_total_precip"])
avg_monthly_precip_pd.head(10)


# %%
def get_daily_precip_month_pd(weather = weather_df, site = "LSGL", month = 1):
    daily_precip_df = weather.groupBy("site", "month", "dayofmonth") \
                            .agg(spark_sum("precip_hrly").alias("total_daily_precip"))
    return daily_precip_df.filter(col("month") == month).filter(col("site")==site)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=avg_monthly_precip_pd,
    x="site",
    y="avg_total_precip",
    hue="month",
    palette="viridis"
)
plt.title("(Site, Month) Pairs by Average Total Precipitation")
plt.xticks(rotation=45)
plt.show()

# %%
import folium
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd

# Load and join
weather_stations = spark.table("weather_stations").toPandas()
weather_stations.rename(columns={"Name": "site"}, inplace=True)

avg_monthly_precip_geo = avg_monthly_precip_pd.merge(weather_stations, on="site", how="left")

# Slider widget (outside the function!)
month_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=12,
    step=1,
    description='Month:',
    continuous_update=False
)

# Function to update map
def plot_precip_map(selected_month):
    # Filter by selected month
    df_filtered = avg_monthly_precip_geo[avg_monthly_precip_geo["month"] == selected_month]

    # Create map
    m = folium.Map(location=[46.8, 8.2], zoom_start=7)

    # Plot each station with a constant circle size
    for _, row in df_filtered.iterrows():
        if not pd.isnull(row['lat']) and not pd.isnull(row['lon']):
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,  # uniform size
                popup=f"{row['site']} ({row['month']}): {row['avg_total_precip']:.1f} mm",
                color='blue',
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

    clear_output(wait=True)
    display(m)

# Link the slider with the function
widgets.interact(plot_precip_map, selected_month=month_slider)
print("Lausanne code: LSGL")

# %% [markdown]
# ### I.e Spark Windows  - 4/20
#
# In the previous question, we calculated the total average monthly precipitation for each site.
#
# Now, let's shift focus: suppose we want to determine, for each day, which site reported the highest temperature.
#
# This is a more complex task—it can't be done with simple aggregations alone. Instead, it requires _windowing_ our data, a powerful technique that allows us to perform calculations across sets of rows that are related to the current row, without collapsing the data as a regular group-by would.
#
#
# We recommend reading this [window functions article](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html), the [spark.sql.Window](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)  and optionally the [Spark SQL](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html) documentation to get acquainted with the idea. You can think of a window function as a fine-grained and more flexible _groupBy_. 
#
# To use window functions in Spark, we need to define two key aspects:
# 1. Window Specifications: This includes defining the columns for partitioning, the order in which the rows should be arranged, and the grouping criteria for the window.
# 2. Aggregation Logic: This specifies the aggregation or computation (such as max, avg, etc.) to be performed on each windowed group
#
# Define a window function, _hourly_window_, that partitions the data by the columns (_year, month, dayofmonth, hour_). Within each partition, order the rows by hourly temperature in descending order. Then, apply the _rank_ function over this window to assign rankings to the sites based on their temperatures (_temp_). Finally, filter the results to keep only the top _N_ ranked sites.
#
# Despite the complexity of the operation, it can be accomplished efficiently in just a few lines of code!

# %%
from pyspark.sql import Window

# %% [markdown]
# First, define a 'tumbling' (fixed-size, non-overlapping) _pyspark.sql.window.WindowSpec_ to specify the partitioning and ordering of the window. This window definition partitions the data by the columns (_year, month, dayofmonth, hour_) and orders the rows (i.e., the site measurements) within each partition by temperature (_temp_), ordered in descending order. As outlined in the previous section, the pattern should follow:
#
# ```
# Window.partitionBy(...).orderBy(...) ...
# ```

# %%
# create the window specifications
from pyspark.sql.functions import rank, col

# Define the window
hourly_window = Window.partitionBy("year", "month", "dayofmonth", "hour").orderBy(col("temp").desc())

# %% [markdown]
# Next, define the computation for the _hourly_window_. This is a window aggregation of type _pyspark.sql.column.Column_, which allows you to perform calculations (such as ranking or aggregation) within the defined window.
#
# Use this _hourly_window_ to calculate the hourly ranking of temperatures. Use the helpful built-in F.rank() _spark.sql.function_, and call its _over_ method to apply it over the _hourly_window_, and name the resulting column (alias) _rank_.

# %%
# TODO - create the hourly ranking logics that will be applied on hourly window
hourly_rank = rank().over(hourly_window).alias("rank")

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `RANK() OVER (PARTITION BY year, month, dayofmonth, hour ORDER BY temp DESC NULLS LAST ...) AS rank`. This _window function_ assigns a rank to each record within the partitions and based on the ordering criteria of _hourly_window_.

# %%
print(hourly_rank)

# %% [markdown]
# Finally, apply the _hourly_rank_ window computation to the _weather_df_ DataFrame computed earlier.
#
# Filter the results to show all and only the sites with the 5 highest temperature per hour (if multiple sites have the same temperature, they count as one), then order the hourly measurements in chronological order, showing the top ranked sites in their ranking order.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +----+----+-----+----------+----+----+----+
# |site|year|month|dayofmonth|hour|temp|rank|
# +----+----+-----+----------+----+----+----+
# +----+----+-----+----------+----+----+----+
# |site|year|month|dayofmonth|hour|temp|rank|
# +----+----+-----+----------+----+----+----+
# |LSZE|2023|1    |1         |0   |16  |1   |
# |LSZT|2023|1    |1         |0   |14  |2   |
# |LSPH|2023|1    |1         |0   |14  |2   |
# |....|....|.    |.         |.   |..  |.   |
# +----+----+-----+----------+----+----+----+
# ```

# %%
# TODO -- apply the window logic to create the additional column rank, and display the results as shown above
weather_ranked_df = weather_df.withColumn("rank", hourly_rank)

# Filter: keep only top 5 ranked temperatures per hour
top5_weather_per_hour_df = weather_ranked_df.filter(col("rank") <= 5)

# Sort results chronologically and by rank
top5_weather_per_hour_df = top5_weather_per_hour_df.orderBy("year", "month", "dayofmonth", "hour", "rank")

# Show the result
top5_weather_per_hour_df.select("site", "year", "month", "dayofmonth", "hour", "temp", "rank").show(20, truncate=False)

# %% [markdown]
# ### I.f Sliding Spark Windows - 4/20

# %% [markdown]
# In the previous question, we computed the rank over a tumbling window, where the windows are of fixed size (hourly) and do not overlap.
#
# With window functions, you can also compute aggregate functions over a _sliding window_, where the window moves across the data, potentially overlapping with previous intervals.
#
# **Question:** For each site, calculate the hourly average temperature computed over the past 3 hours of data.
#
# The process follows a similar pattern to the previous steps, with a few distinctions:
#
# * Rows are processed independently for each site.
# * The window slides over the timestamps (_valid_time_gmt_, in seconds) in chronological order, spanning intervals going back to 2 hours and 59 minutes (10740 seconds) **before** the current row's timestamp up to the current row's timestamp."
#
# 💡 Hints:
# * [spark.sql.Window(Spec)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html)
# * Times are in minutes intervals

# %% [markdown]
# First, as before, define a _pyspark.sql.window.WindowSpec_ to specify the window partition, the row ordering inside the partition, and its _range_.

# %%
sliding_3hour_window = Window.partitionBy("site").orderBy(col("valid_time_gmt")).rangeBetween(-10740, 0)

# %% [markdown]
# Next, define a computation on the window: calculate the average ranking of temperatures. Use the helpful built-in F.avg() spark.sql.function, apply it _over_ the window, and name the result (_alias_) _avg_temp_.

# %%
sliding_3hour_avg = F.avg(col("temp")).over(sliding_3hour_window).alias("avg_temp")

# %% [markdown]
# **Checkpoint:** the resulting object is analogous to the SQL expression `avg(temp) OVER (PARTITION BY site ORDER BY valid_time_gmt ASC NULLS FIRST RANGE BETWEEN ... AND CURRENT ROW) AS avg_temp`

# %%
print(sliding_3hour_avg)

# %% [markdown]
# Finally, apply _sliding_3hour_avg_ to the _weather_df_ DataFrame computed earlier, and order chronologically. _Then_ filter the output to show the outpout of sites 'LSTO', 'LSZH and 'LSGL'.
#
# **Checkpoint:** The output should ressemble:
#
# ```
# +--------------+----+-----+----------+----+----+----+------------------+
# |valid_time_gmt|year|month|dayofmonth|hour|site|temp|          avg_temp|
# +--------------+----+-----+----------+----+----+----+------------------+
# |    1640991600|2022|    1|         1|   0|LSGL|   7|               7.0|
# |    1640991600|2022|    1|         1|   0|LSTO|  10|              10.0|
# |    1640992800|2022|    1|         1|   0|LSZH|   2|               2.0|
# |    1640994600|2022|    1|         1|   0|LSZH|   3|               2.5|
# |    1640995200|2022|    1|         1|   1|LSGL|   6|               6.5|
# |    1640995200|2022|    1|         1|   1|LSTO|  10|              10.0|
# |    1640996400|2022|    1|         1|   1|LSZH|   3|2.6666666666666665|
# |    1640998200|2022|    1|         1|   1|LSZH|   3|              2.75|
# (...)
# ```

# %%
# TODO -- apply the sliding_3hour_avg logic, showing ony the results for the sites
weather_with_avg = weather_df.withColumn("avg_temp", sliding_3hour_avg)
# Filter for specific sites
filtered_weather = weather_with_avg.filter(col("site").isin("LSTO", "LSZH", "LSGL")).orderBy("valid_time_gmt")

# %%
# %%time
filtered_weather.select(
    "valid_time_gmt",
    "year",
    "month",
    "dayofmonth",
    "hour",
    "site",
    "temp",
    "avg_temp"
).show(8, truncate=False)

# %% [markdown]
# __Note:__ The code block under is a sanity check against the example in the question.

# %%
# Test against the example
weather_df_test = df_weather_with_time.filter(col("year") == 2022)
sliding_window = Window.partitionBy("site").orderBy(col("valid_time_gmt")).rangeBetween(-10740, 0)
sliding_3hour_avg = F.avg(col("temp")).over(sliding_window).alias("avg_temp")
weather_with_avg = weather_df_test.withColumn("avg_temp", sliding_3hour_avg)
filtered_weather = weather_with_avg.filter(col("site").isin("LSTO", "LSZH", "LSGL"))
filtered_weather = filtered_weather.orderBy("valid_time_gmt")
filtered_weather.select("valid_time_gmt", "year", "month", "dayofmonth", "hour", "site", "temp", "avg_temp").show(8, truncate=False)

# %% [markdown]
# Adapt the _hourly_rank_ and combine it with the window to show the weather stations with the 5 top temperatures averaged over the 3h sliding window.

# %%
# %%time
sliding_hourly_rank=Window.partitionBy('year','month','dayofmonth','hour').orderBy(F.desc('avg_temp'))

weather_df.select(
    'valid_time_gmt', 
    'year', 
    'month', 
    'dayofmonth',
    'hour', 
    'site', 
    'temp', 
    sliding_3hour_avg
).select('valid_time_gmt',
         'year',
         'month',
         'dayofmonth',
         'hour',
         'site', 
         col('avg_temp').alias('temp'),
         hourly_rank
        ).filter('rank <= 5').sort('valid_time_gmt','rank').show(8, truncate=False)

# %%

# %% [markdown]
# ---
# ## PART II: SBB Network - Vehicle Journey Trajectories (20 points)

# %% [markdown]
# ### II.a Filter trips from SBB Istdaten - 4/20
#

# %% [markdown]
# In this part, you will reconstruct public transport journey trajectories from the available transport data, as illustrated below. The example displays the historical data extracted from istdaten for a single trip ID, collected over the course of a year. The horizontal axis represents the sequence of stops along the trip, while the vertical axis shows the timeline of arrivals and departures of each trip. This type of chart offers valuable insights into where delays typically occur along the journey.
#
# ![./figs/journeys.png](./figs/journeys.png)
#
# There are several ways to compute this table in Spark, each with its own trade-offs. In the next question, you'll explore one such method using window and table pivot functions.
#
# ⚠️ The question in this section can be computationally demanding if you are not careful, therefore:
#
# - It is advisable to begin by experimenting with smaller datasets first. Starting with smaller datasets enables faster iterations and helps to understand the computational requirements before moving on to larger datasets.
# - It is advisable to use the DataFrame _cache()_ for the most expensive computation. _Istdaten_trips_df_ is a good candidate for that, because it can take several tens of seconds to generate this relatively small table.
#

# %% [markdown]
# ---
# First, create the DataFrame _istdaten_trips_df_ from the _iceberg.sbb.istdaten_ table.
#
# The table must:
# - Only include the ids of _distinct_ trip that appear on at least 200 different days in _isdaten_ in _2024_
# - Only trips from the Transport Lausanne (TL) operator.
# - Onyl trip ids that serve stops in the Lausanne region.
#     - Use the data available in _/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet_)
#     - Or use Trino to create your own list of stops in the greater Lausanne region (Lausanne and Ouest Lausannois).
#
# 💡 Note:
# - You may assume that the SBB timetables (_stops_, _stop_times_ etc), are valid for the full year in which they are published.
# - Filtering the trips based on both the TL operator and the presence of at least one stop in the only region served by this operator might seem redundant in this case. However, in a more general context, this approach allows us to reuse the same query for nation wide operators.
#

# %%
# %time
## Or use your own
lausanne_stops_df = spark.sql('SELECT DISTINCT * FROM parquet.`/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet`')
lausanne_stops_df.createOrReplaceTempView('lausanne_stops_df')

# %%
lausanne_stops_df.printSchema()

# %%
lausanne_stops_df.show(1)

# %%
spark.sql(f'SHOW TABLES IN iceberg.sbb').show(truncate=False)

# %%
# %%time
trips_TL_2024_df = spark.read.parquet(f"{hadoopFS}/user/{username}/assignment-3/data/trips_TL_2024_df")

# %%
# %%time
# distinct count (check if all trip ids are unique)
trips_TL_2024_df.select("trip_id").distinct().count()

# %%
# %%time
trips_TL_2024_df.show(10, truncate=False)

# %% [markdown]
# ### II.b Transform the data - 8/20
#
# Next, use the _istdaten_trips_df_ table computed earlier to create a Spark Dataframe _istdaten_df_ that contains a subset of _sbb.istdaten_ containing only trips that are listed into _istdaten_trips_df_.
#
# The table must:
# - Include the _istdaten_ details for the full year 2024 of all the trips that appear in _istdaten_trips_df_.
# - Not include _failed_ or _unplanned_ trips.
# - Include all stops in the Lausanne area and stops that are not listed in the Lausanne area, but are connected via at least one trip to stops in the Lausanne area.
#
# The table should be similar to the one shown below when properly ordered (showing only one trip ID on a given operating day, it can have additional columns if you want):
#
# ```
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# |operating_day|trip_id                      |bpuic  |arr_time           |arr_actual         |dep_time           |dep_actual         |
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# |2024-01-03   |85:151:TL013-4506262507243798|8579253|NULL               |NULL               |2024-01-03 11:51:00|2024-01-03 11:51:30|
# |2024-01-03   |85:151:TL013-4506262507243798|8579254|2024-01-03 11:52:00|2024-01-03 11:52:14|2024-01-03 11:52:00|2024-01-03 11:52:50|
# |2024-01-03   |85:151:TL013-4506262507243798|8591991|2024-01-03 11:53:00|2024-01-03 11:54:19|2024-01-03 11:53:00|2024-01-03 11:54:45|
# |2024-01-03   |85:151:TL013-4506262507243798|8592074|2024-01-03 11:56:00|2024-01-03 11:56:04|2024-01-03 11:56:00|2024-01-03 11:56:10|
# |2024-01-03   |85:151:TL013-4506262507243798|8592009|2024-01-03 11:57:00|2024-01-03 11:56:57|2024-01-03 11:57:00|2024-01-03 11:57:25|
# |2024-01-03   |85:151:TL013-4506262507243798|8592083|2024-01-03 11:58:00|2024-01-03 11:57:50|2024-01-03 11:58:00|2024-01-03 11:58:17|
# |2024-01-03   |85:151:TL013-4506262507243798|8592045|2024-01-03 11:59:00|2024-01-03 11:58:42|2024-01-03 11:59:00|2024-01-03 11:58:42|
# |2024-01-03   |85:151:TL013-4506262507243798|8592129|2024-01-03 12:00:00|2024-01-03 11:59:04|NULL               |NULL               |
# +-------------+-----------------------------+-------+-------------------+-------------------+-------------------+-------------------+
# ```

# %%
# %%time
filtered_istdaten = spark.read.parquet(f"{hadoopFS}/user/{username}/assignment-3/data/filtered_istdaten")

# %%
# %%time
filtered_istdaten.printSchema()

# %% [markdown]
# Validate for one trip ID:

# %% [markdown]
# ### II.c Compute Journey Trajectories - 8/20
#
# Create a windows operator as seen before to work on _operating_day, trip_id_, _ordered_ by _arr_time_ (expected arrival times, and actual arrival times to break ties if expected arrival times are equal). Use the window to create the Spark DataFrame _trip_sequences_df_. In each window, compute:
# - _start_time_: the **first non-null** (ignore nulls) expected _dep_time_ in the window, with respect to the window's ordering.
# - _sequence_: the order of the _bpuic_ in the trip journey, according to the windows' ordering.
# - _arr_time_rel_: the interval _(arr_time - start_time)_, or NULL if _arr_time_ is NULL
# - _dep_time_rel_: the interval _(dep_time - start_time)_, or NULL if _dep_time_ is NULL
# - _arr_actual_rel_: the interval _(arr_actual - start_time)_, or NULL if _arr_actual_ is NULL
# - _dep_actual_rel_: the interval _(dep_actual - start_time)_, or NULL if _dep_actual_ is NULL
#
# The results for a given _operating_day, trip_id_ should look like this, feel free to add additional columns, such as _line_text_ etc.:
# ```
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# |operating_day|             trip_id|  bpuic|sequence|         start_time|        arr_time_rel|      arr_actual_rel|        dep_time_rel|      dep_actual_rel|
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# |   2024-01-03|85:151:TL013-4506...|8579253|       1|2024-01-03 11:51:00|                NULL|                NULL|INTERVAL '0 00:00...|INTERVAL '0 00:00...|
# |   2024-01-03|85:151:TL013-4506...|8579254|       2|2024-01-03 11:51:00|INTERVAL '0 00:01...|INTERVAL '0 00:01...|INTERVAL '0 00:01...|INTERVAL '0 00:01...|
# |   2024-01-03|85:151:TL013-4506...|8591991|       3|2024-01-03 11:51:00|INTERVAL '0 00:02...|INTERVAL '0 00:03...|INTERVAL '0 00:02...|INTERVAL '0 00:03...|
# |   2024-01-03|85:151:TL013-4506...|8592074|       4|2024-01-03 11:51:00|INTERVAL '0 00:05...|INTERVAL '0 00:05...|INTERVAL '0 00:05...|INTERVAL '0 00:05...|
# |   2024-01-03|85:151:TL013-4506...|8592009|       5|2024-01-03 11:51:00|INTERVAL '0 00:06...|INTERVAL '0 00:05...|INTERVAL '0 00:06...|INTERVAL '0 00:06...|
# |   2024-01-03|85:151:TL013-4506...|8592083|       6|2024-01-03 11:51:00|INTERVAL '0 00:07...|INTERVAL '0 00:06...|INTERVAL '0 00:07...|INTERVAL '0 00:07...|
# |   2024-01-03|85:151:TL013-4506...|8592045|       7|2024-01-03 11:51:00|INTERVAL '0 00:08...|INTERVAL '0 00:07...|INTERVAL '0 00:08...|INTERVAL '0 00:07...|
# |   2024-01-03|85:151:TL013-4506...|8592129|       8|2024-01-03 11:51:00|INTERVAL '0 00:09...|INTERVAL '0 00:08...|                NULL|                NULL|
# +-------------+--------------------+-------+--------+-------------------+--------------------+--------------------+--------------------+--------------------+
# ```
#
# And the schema (minimum column set):
# ```
# root
#  |-- operating_day: date (nullable = true)
#  |-- trip_id: string (nullable = true)
#  |-- bpuic: integer (nullable = true)
#  |-- sequence: integer (nullable = false)
#  |-- start_time: timestamp_ntz (nullable = true)
#  |-- arr_time_rel: interval day to second (nullable = true)
#  |-- arr_actual_rel: interval day to second (nullable = true)
#  |-- dep_time_rel: interval day to second (nullable = true)
#  |-- dep_actual_rel: interval day to second (nullable = true)
# ```
#
# 💡 Hints:
# - The times are of type _timestamp_ntz_. You can easily compute a time interval with expressions like _F.col('t2')-F.col('t1')_ or _F.expr('t2 - t1')_.
# - Use Windows aggregation logics AUDF (as previously seen) to compute the start time and the sequence number over the _operating_day, trip_id_ windows.
# - Use _F.row_number()_ to get the row number in a window (according to the window's ordering).
# - In Spark ordering, NULL timestamps come first.

# %%
## TODO - create the window specifications

# %%
## TODO - create the logics you want to apply on the window

# %%
# %%time
trip_sequences_df = spark.read.parquet(f"{hadoopFS}/user/{username}/assignment-3/data/trip_sequences_df")

# %%
trip_sequences_df.printSchema()

# %%
# %%time
trip_sequences_df.count()

# %%
# %%time
trip_sequences_df.filter(
    """trip_id LIKE '85:151:TL013-4506262507243798'"""
).select(
    'operating_day',
    'trip_id',
    'bpuic',
    'sequence',
    'start_time',
    'arr_time_rel',
    'arr_actual_rel',
    'dep_time_rel',
    'dep_actual_rel'
).show(8,truncate=True)

# %%
trip_sequences_df.createOrReplaceTempView('trip_sequences_df')

# %% [markdown]
# ---
# Use the _trip_sequence_df_ dataframe to trace the journey of each trip, displaying the travel time from the first stop to all subsequent stops sequence along the route. The _x-axis_ of the graph represents the stops in the journey, while the _y-axis_ represents the travel time from the first stop.
#
# Note that the dataframe contains many invalid traces that should be ignored
#
# You can verify the presence of invalid traces by running queries on the dataframe, such as:
#
# ```
# SELECT trip_id,
#        bpuic,
#        collect_set(sequence)
#        FROM trip_sequences_df
#              GROUP BY trip_id,bpuic
# ```
# Or
# ```
# SELECT trip_id,
#        num_stops,
#        COUNT(*) as freq
#        FROM (
#              SELECT operating_day,
#                     trip_id,
#                     MAX(sequence) as num_stops
#                  FROM trip_sequences_df
#                  GROUP BY operating_day,trip_id
#        ) GROUP BY trip_id,num_stops ORDER BY trip_id
# ```
#
# These queries reveal inconsistencies in stop ordering, or inconsistent number of stops per trip across the year. Together, these queries (and similar ones) are useful for identifying day-to-day variations or anomalies in the stop sequences of scheduled trips.
#
# We did this analysis for you and we suggest that you to focus your analysis on valid and reliable trip patterns, such as _trip_id='85:151:TL031-4506262507505612'_ for the rest of this study.

# %%
# Time-Order spesific trip
trip_data = trip_sequences_df.filter(
    F.col("trip_id") == "85:151:TL031-4506262507505612"
)

# Convert to Pandas for visualization
trip_pd = trip_data.toPandas()


# Ensure it's sorted by sequence
trip_pd = trip_pd.sort_values(by='sequence')

# Find first valid arrival time (some trips may start with NaT)
first_arrival = trip_pd['arr_actual_rel'].dropna().iloc[0]

# Compute cumulative travel time in minutes
trip_pd['cumulative_travel_time_min'] = (
    trip_pd['arr_actual_rel'] - first_arrival
).dt.total_seconds() / 60

# %%
trip_pd.head(10)

# %%
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import plotly.express as px

# Extract just the date string for clearer legends
trip_pd['operating_day_str'] = trip_pd['operating_day'].astype(str)

fig = px.line(
    trip_pd,
    x="sequence",
    y="cumulative_travel_time_min",
    color="operating_day_str",             
    line_group="operating_day_str",
    markers=False,
    title='Cumulative Travel Time \n Trip ID: 85:151:TL031-4506262507505612',
    labels={
        "sequence": "Stop Sequence",
        "cumulative_travel_time_min": "Cumulative Travel Time (min)",
        "operating_day_str": "Operating Day"
    },
    width=1000,                            # Smaller window
    height=500
)

fig.update_traces(opacity=0.5)            # Make lines more subtle
fig.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='sequence', y='cumulative_travel_time_min', data=trip_pd)
plt.xlabel('Stop Sequence')
plt.ylabel('Cumulative Travel Time (min)')
plt.title('Distribution of Cumulative Travel Time \n Trip ID: 85:151:TL031-4506262507505612')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Consider only _trip_id='85:151:TL031-4506262507505612'_.
#
# Create the table _trips_pivoted_df_, based on the following conditions:
# - It only include the trip with the specified trip_id.
# - It only include trips on operating days where the first actual departure time (at sequence id 1) is within 5 minutes (before or after) of the expected departure time.
# - The table should consists of the following columns:
#     - _bpuic_: a stop on the _trip_id_ line
#     - _sequence_: The sequence number of the bpuic, _1_ is the first stop.
#     - _evt_type_: indicates if the row corresponds to an arrival or a departure time of the trip at the given stop.
#     - _{trip_id}_: Contains the expected departure and arrival times (in chronological order) of the the selected _trip_id_.
#     - _{trip_id}-{operating_day}_: Contains the actual departure and arrival times (in chronological order) of the selected _trip_id_, on the operating_day.
#
# The table schema should look like:
#
# ```
# root
#  |-- bpuic: integer (nullable = true)
#  |-- sequence: integer (nullable = false)
#  |-- evt_type: string (nullable = false)
#  |-- 85:151:TL031-4506262507505612: long (nullable = true)              # Column of expected times
#  |-- 85:151:TL031-4506262507505612_2024-01-03: long (nullable = true)   # Actual times on 2024-01-03
#  |-- 85:151:TL031-4506262507505612_2024-01-04: long (nullable = true)   # Actual times on 2024-01-04
#  |-- 85:151:TL031-4506262507505612_2024-01-05: long (nullable = true)   # ...
#  |-- 85:151:TL031-4506262507505612_2024-01-08: long (nullable = true)
#  |-- ...
# ```
#
# And below is sample of how the table should look like. Column (4) are the expected times, columns (5) and above are 
# actual times observed on the given days.
#
# ```
#    (1)      (2)       (3)         (4)                 (5)                        (...)
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# |  bpuic|sequence|evt_type|85:151:TL031-...|85:151:TL031-..._2024-01-03|85:151:TL031-..._2024-01-04|
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# |8588983|       1|     arr|            NULL|                       NULL|                       NULL|
# |8588983|       1|     dep|               0|                         12|                          6|
# |8593869|       2|     arr|              60|                        149|                        129|
# |8593869|       2|     dep|              60|                        158|                        129|
# |8591933|       3|     arr|             180|                        217|                        180|
# |8591933|       3|     dep|             180|                        238|                        220|
# |8593868|       4|     arr|             240|                        253|                        220|
# |8593868|       4|     dep|             240|                        256|                        220|
# |8507227|       5|     arr|             360|                        321|                        280|
# |8507227|       5|     dep|             360|                        336|                        280|
# |8593867|       6|     arr|             420|                        357|                        326|
# |8593867|       6|     dep|             420|                        376|                        343|
# |8594986|       7|     arr|             480|                        433|                        403|
# |8594986|       7|     dep|             480|                        457|                        423|
# +-------+--------+--------+----------------+---------------------------+---------------------------+
# ```
#
# 💡 Hints:
# - It will be easier for you to convert the time intervals to seconds, e.g. using the UDF _F.col('interval').cast("long")_, or the Spark DataFrame _filter("CAST(interval AS long)")_

# %% [markdown]
# There are many ways to accomplish this task. The steps outlined below are just one possible approach. Feel free to experiment and try out your own method.
#
# **First**, compute the DataFrame _trip_filter_list_df_ that contains ony the _operating days_ on which the actual departure time of the considered _trip_id_ at _sequence=1_ is no more than 5mins (300 seconds) before or after the expected departure time.

# %%
# %%time
trip_data.printSchema()

# %%
# %time
## TODO - create the trip_filter_list_df as indicated above
# Starting with trip_data with tripID filtered

trip_filter_list_df = (trip_data.filter(F.col("sequence") == 1)  
    # Filer @ first stop with actual dep - dep < 5min
    .withColumn(
        "time_diff",
        F.abs(F.col("dep_actual_rel").cast("long") - F.col("dep_time_rel").cast("long"))
    ).filter(F.col("time_diff") <= 300)
    # Select only the operating_day column and drop duplicates
    .select("operating_day").distinct()
)

# %%
# %%time
trip_filter_list_df.show()

# %% [markdown]
# **Second**, create _trip_filter_sequence_df_, a subset of _trip_sequence_df_ computed earlier that contains only the trips of interest happening on the days computed in _trip_filter_list_df_

# %%
# %%time
## TODO - create the subset of trip_filter_sequence_df as indicated above
trip_filter_sequences_df = (trip_sequences_df.filter(F.col("trip_id") == "85:151:TL031-4506262507505612")
    # Inner Join with valid operating days 
    .join(trip_filter_list_df, on="operating_day", how="inner")
)

# %%
# %%time
trip_filter_sequences_df.printSchema()
# Used step1 to Filter the full trip_sequences_df to retain the stops info,
# ,but only on dep actual-dep requeirement is satisfied 

# %% [markdown]
# Next, create two DataFrames, _planned_df_ and _actual_df_.
#
# For _planned_df_, The schema should include the following columns:
# - _trip_id_: The trip identifier, e.g. _85:151:TL031-4506262507505612_
# - _bpuic_: A stop ID (this column is informative only, for verification purpose).
# - _sequence_: The sequence number of the stop within the specified _trip_id_.
# - _evt_type_: Use the function _F.explode()_ in a _withColumn_ operation to duplicate each row into two: one with _evt_type_ set to "arr" and the other with _evt_type_ set to "dep".
# - _evt_time_:
#     - _F.col('arr_time_rel')_ **when** _evt_type = "arr"_
#     - _F.col('dep_time_rel')_ **when** _evt_type = "dep"_
#
# For _actual_df_:
#     This DataFrame will have the same schema as _planned_df_, but the values for _evt_time_ will be based on _arr_actual_rel_ and _dep_actual_rel_, instead of the planned _arr_time_rel_ and _dep_time_rel_. The values in the column _trip_id_ should be changed to the append the _operating_day_ to the _trip_id_, e.g. 85:151:TL031-4506262507505612_2024-01-03, 85:151:TL031-4506262507505612_2024-01-04, ...
#
# 💡 Hints:
# - We recommend that you convert _evt_time_ to seconds, e.g. using _F.col("evt_time").cast(long)_

# %%
## TODO - any imports here

# %%
# %%time
## TODO - create the planned_df table
planned_df = (trip_filter_sequences_df
   .withColumn("evt_type", F.explode(F.array(F.lit("arr"), F.lit("dep"))))   # Duplicate each row into "arr" and "dep" events
    # Set evt_time based on evt_type
    .withColumn("evt_time",
        F.when(F.col("evt_type") == "arr", F.col("arr_time_rel").cast("long"))
          .otherwise(F.col("dep_time_rel").cast("long"))
    )
     
    # Select required columns
    .select(
        F.col("trip_id").alias("tripID_date"),
        "bpuic",
        "sequence",
        "evt_type",
        "evt_time"
    )
)
planned_df.cache();

# %%
# %%time
## TODO - create the actual_df table
actual_df = (trip_filter_sequences_df
    .withColumn("evt_type", F.explode(F.array(F.lit("arr"), F.lit("dep"))))
    
    # Set evt_time based on evt_type - ONLY USE ACTUAL TIMES
    .withColumn("evt_time",
        F.when(F.col("evt_type") == "arr", F.col("arr_actual_rel").cast("long"))
          .otherwise(F.col("dep_actual_rel").cast("long"))  
    )
    
    # Modify tripID to include operating_day
    .withColumn(
        "tripID_date",
        F.concat(F.col("trip_id"), F.lit("_"), F.col("operating_day"))
    )
    
    # Select and rename columns
    .select(
        "tripID_date",
        "bpuic",
        "sequence",
        "evt_type",
        "evt_time"
    )
)
actual_df.cache();

# %% [markdown]
# Finally, create the the union of the _actual_df_ and the _planned_df_ DataFrames (append the rows) and execute a _pivot_ operation on the union.
#
# Pivoting in Spark is a technique used to transform data from a long format, where rows in a single column may combine values from multiple entities (i.e. different operating_day) into a wide format, where each unique value of a column in the long format becomes its own column in the wide format. Essentially, it "spreads" out the data from one column across multiple columns based on a grouping rules. Note that pivot is a _GroupedData_ operation, it requires a _groupBy_.
#
# For example:
# - We are pivoting on the _trip_id_ column, which means we want each unique _trip_id_ to become its own column in the resulting dataframe.
# - For each group, defined by _bpuic_, _evt_type_ (arrival or departure) and _sequence_, we want to select the first _evt_time_ for each unique combination of _bpuic_, _evt_type_ and _sequence_ and copy it to the corresponding _trip_id_ column.
#
# So, pivoting reorganizes the data, turning a single column's unique values into new columns, making it easier to analyze and compare data across different entities (like trips in this case).
#
# See:
# * [Spark pivot (SQL)](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-pivot.html)
# * [Spark pivot (python)](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.pivot.html#pyspark.sql.GroupedData.pivot)

# %%
# %%time
## TODO - execute the pivot as explained above
trips_pivoted_df = (planned_df.union(actual_df)
    .groupBy("bpuic", "sequence", "evt_type")
    .pivot("tripID_date")
    .agg(F.first("evt_time"))  # Take the first time for each group
    .orderBy("sequence", "evt_type")  # sort by stop sequence and then event type
)
trips_pivoted_df.cache();

# %%
trips_pivoted_df.printSchema()

# %%
trips_pivoted_df.select('bpuic','sequence','evt_type', "85:151:TL031-4506262507505612", "85:151:TL031-4506262507505612_2024-01-03", "85:151:TL031-4506262507505612_2024-01-04").show()

# %%
planned_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
actual_df.filter("sequence=1").groupBy("evt_type").agg(F.col("evt_type"),F.min("evt_time"),F.max("evt_time")).show(truncate=False)

# %%
trips_pivoted_df.toPandas().drop(columns=['bpuic','evt_type']).plot(x='sequence',legend=False)

# %%
# %%time
plot_data = trips_pivoted_df.toPandas().drop(columns=['bpuic', 'evt_type'])
plot_data.update(plot_data.drop(columns='sequence') / 60) #sec -> min
predict_ID = '85:151:TL031-4506262507505612'

fig, ax = plt.subplots(figsize=(10, 7))

plot_data.plot(
    ax=ax,
    x='sequence',
    alpha=0.5,
    linewidth=1,
    legend=False
)

plot_data.plot(
    ax=ax,
    x='sequence',
    y = predict_ID,
    color='black',
    alpha=1,
    linewidth=1.3,
    label='Predicted Trip'
) 

ax.set_title("Cumulative Travel Time: Prediction & Real Time \n Trip ID: 85:151:TL031-4506262507505612")
ax.set_xlabel("Stop Sequence")
ax.set_ylabel("Cumulative Travel Time (min)")

# %%
# %%time
fig, ax = plt.subplots(figsize=(10, 7))
melted = plot_data.melt(id_vars='sequence', var_name='trip_id', value_name='cumulative_time')

melted.boxplot(column='cumulative_time', by='sequence', grid=False, ax=ax)

plot_data.plot(
    ax=ax,
    x='sequence',
    y = predict_ID,
    color='black',
    alpha=1,
    linewidth=1,
    label='Predicted Trip'
) 

ax.set_title("Cumulative Travel Time Prediced & Real Time Average \n Trip ID: 85:151:TL031-4506262507505612")
plt.suptitle("")  # Remove automatic title from the boxplot
ax.set_xlabel("Stop Sequence")
ax.set_ylabel("Cumulative Travel Time (min)")

# %% [markdown]
# ---
# ## PART III: SBB Delay Model building (20 points)
#
# In the final segment of this assignment, your task is to tackle the prediction of SBB delays within the Lausanne region.
#
# To maintain simplicity, we've narrowed down the scope to building and validating a model capable of predicting delays exceeding 5 minutes. The model will classify delays as 0 if they're less than 5 minutes, and 1 otherwise. That said, you're encouraged to explore regression models if you'd like to predict delays as continuous values for more granular insights.
#
# This problem offers ample room for creativity, allowing for multiple valid solutions. We provide a structured sequence of steps to guide you through the process, but beyond that, you'll navigate independently. By this stage, you should be adept in utilizing the Spark API, enabling you to explore the Spark documentation and gather all necessary information.
#
# Feel free to employ innovative approaches and leverage methods and data acquired in earlier sections of the assignment. This open-ended problem encourages exploration and experimentation.

# %% [markdown]
# ### III.a Feature Engineering - 8/20
#
#
# Construct a feature vector for training and testing your model.
#
# Best practices include:
#
# * Data Source Selection and Exploration:
#   - Do not hesitate to reuse the data from Lausanne created in assignment 2. Query the data directly from files into Spark DataFrames.
#   - Explore the data to understand its structure, identifying relevant features and potential issues such as missing or null values.
#
# * Data Sanitization:
#   - Clean up null values and handle any inconsistencies or outliers in the data.
#
# * Historical Delay Computation:
#   - Utilize the SBB historical istdaten to compute historical delays, incorporating this information into your feature vector.
#   - Experiment with different ways to represent historical delays, such as aggregating delays over different time periods or considering average delays for specific routes or stations.
#
# * Incorporating Additional Data Sources:
#   - Integrate other relevant data sources, **at a minimum, integrate weather data history** from the previous questions into your feature vector.
#   - Explore how these additional features contribute to the predictive power of your model and how they interact with the primary dataset.
#
# * Feature Vector Construction using Spark MLlib:
#   - Utilize [`Spark MLlib`](https://spark.apache.org/docs/latest/ml-features.html). methods to construct the feature vector for your model.
#   - Consider techniques such as feature scaling, transformation, and selection to enhance the predictive performance of your model.
#

# %% [markdown]
# ### Method 1: Joining Delay (from trips_pivoted_df) with Daily Weather 
# First we do a few visualization to track the data sanitization and to get a feel for 1-0 delay inbalance

# %%
delay_flags_df = spark.read.parquet(f"{hadoopFS}/user/{username}/assignment-3/data/delay_flags_df")

# %%
delay_flags_df.printSchema()

# %%
# %%time
trips_pivoted_df \
    .select(
        'bpuic',
        'sequence', 
        'evt_type',
        F.col("85:151:TL031-4506262507505612").alias("scheduled"),
        F.col("85:151:TL031-4506262507505612_2024-01-03").alias("actual_2024-01-03"),
        F.col("85:151:TL031-4506262507505612_2024-01-04").alias("actual_2024-01-04")
    ) \
    .withColumn("delay_2024-01-03", 
        (F.col("actual_2024-01-03") - F.col("scheduled"))) \
    .withColumn("delay_2024-01-04", 
        (F.col("actual_2024-01-04") - F.col("scheduled"))) \
    .orderBy("sequence", "evt_type") \
    .show(n=10, truncate=False, vertical=True)

# %%
feature_vector_df = spark.read.parquet(f"{hadoopFS}/user/{username}/assignment-3/data/feature_vector_df")

# %%
feature_vector_df.show(truncate=False)

# %%
# TODO ...
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# %% [markdown]
# ### Tables from assignment 2

# %%
stops = spark.read.parquet(f"hdfs://iccluster059.iccluster.epfl.ch:9000/user/{username}/assignment-3/stops_lausanne_region.parquet")

print("Schema:"); stops.printSchema()
print("Row count:", stops.count())
stops.show(3)

# %%
walking_pair = spark.read.parquet(f"hdfs://iccluster059.iccluster.epfl.ch:9000/user/{username}/assignment-3/lausanne_trips.parquet")
print("Schema:"); walking_pair.printSchema()
print("Row count:", walking_pair.count())
walking_pair.show(3)

# %%
[v for v in locals().keys() if v.endswith('_df')]

# %%
# Convert INTERVAL columns to seconds
ts_df = (
    trip_sequences_df
    .withColumn("arr_plan_sec",  F.col("arr_time_rel").cast("long"))
    .withColumn("arr_act_sec",   F.col("arr_actual_rel").cast("long"))
    .withColumn("dep_plan_sec",  F.col("dep_time_rel").cast("long"))
    .withColumn("dep_act_sec",   F.col("dep_actual_rel").cast("long"))
)

# Delay (take whichever is defined)
ts_df = (
    ts_df
    .withColumn("delay_sec",
        F.greatest(
            F.when(F.col("arr_act_sec").isNotNull() & F.col("arr_plan_sec").isNotNull(),
                   F.col("arr_act_sec") - F.col("arr_plan_sec")).otherwise(F.lit(0)),
            F.when(F.col("dep_act_sec").isNotNull() & F.col("dep_plan_sec").isNotNull(),
                   F.col("dep_act_sec") - F.col("dep_plan_sec")).otherwise(F.lit(0))
        )
    )
    .withColumn("delay_gt5min", (F.col("delay_sec") > 300).cast("int"))
)

# %%
# %%time
# Historical istdaten
hist_df = spark.sql("SELECT operating_day, trip_id, line_id, bpuic, arr_time, arr_actual, dep_time, dep_actual FROM iceberg.sbb.istdaten")

# delays between each timestamp (seconds)
hist_df = hist_df.withColumn("arr_delay_sec",F.unix_timestamp("arr_actual") - F.unix_timestamp("arr_time")).withColumn("dep_delay_sec",F.unix_timestamp("dep_actual") - F.unix_timestamp("dep_time"))

# sanity check
hist_df.select("arr_time", "arr_actual", "arr_delay_sec").show(5, truncate=False)

# %% [markdown]
# ### **Method 1: Monthly average delay per daily stop "recent trend" (bpuic)**

# %%
# %%time
# daily summary
daily_stop = (
  hist_df
    .groupBy("bpuic","operating_day")
    .agg(
      F.avg("arr_delay_sec").alias("daily_avg_arr"),
      F.avg("dep_delay_sec").alias("daily_avg_dep")
    )
)

# 30-day rolling over those days (today + previous 29)
win30 = Window.partitionBy("bpuic").orderBy("operating_day").rowsBetween(-29,0)
daily_roll = daily_stop \
  .withColumn("stop_30d_roll_avg_arr", F.avg("daily_avg_arr").over(win30)) \
  .withColumn("stop_30d_roll_avg_dep", F.avg("daily_avg_dep").over(win30))

daily_roll \
  .filter(F.col("bpuic").isNotNull()) \
  .orderBy("bpuic","operating_day") \
  .show(5, truncate=False)

### DATA SANITIZATION
daily_roll = daily_roll.filter(F.col("bpuic").isNotNull())


# %% [markdown]
# ### **Method 2: Monthly average delay per route "performance on calendar month basis" (line_id)**

# %%
# %%time

# Extract year-month bucket from the date
by_month = hist_df.withColumn("year_month", F.date_format("operating_day", "yyyy-MM"))


# aggregate by each route in each month, compute average arrival/departure delays
route_monthly_features = (
    by_month
      .groupBy("line_id", "year_month")
      .agg(
        F.avg("arr_delay_sec").alias("route_monthly_avg_arr_delay"),
        F.avg("dep_delay_sec").alias("route_monthly_avg_dep_delay"),
        F.count("*").alias("route_monthly_event_count")
      )
)


# %%
# %%time
### QUICK INSPECTION OF FEATURE TABLES
route_monthly_features.show(5)

# %% [markdown]
# ### Join all features into one singular dataframe and data sanitization

# %%
# %%time
#  base table with delay target and join keys
features_df = (
    hist_df
      .withColumn("label", F.unix_timestamp("arr_actual") - F.unix_timestamp("arr_time"))
      # keys for joins:
      .withColumn("operating_day", F.to_date("operating_day"))
      .withColumn("year_month",      F.date_format("operating_day","yyyy-MM"))
      .withColumn("ts_hour",         F.date_trunc("hour", "arr_actual"))
      .select("trip_id","line_id","bpuic","operating_day","year_month","ts_hour","label"))

# join in the 30-day rolling stop delays

features_df = features_df.join(daily_roll.select("bpuic","operating_day",
              "stop_30d_roll_avg_arr","stop_30d_roll_avg_dep"),on=["bpuic","operating_day"],how="left")

# join in the monthly route averages

features_df = features_df.join(route_monthly_features,on=["line_id","year_month"],how="left")


# join in the **weather** features, hourly bucket
features_df = features_df.join(
    weather_df,      
    on="ts_hour", how="left"
).fillna({
    # fill any missing weather or rolling stats with reasonable defaults
    "stop_30d_roll_avg_arr":   0.0,
    "stop_30d_roll_avg_dep":   0.0,
    "route_monthly_avg_arr_delay": 0.0,
    "route_monthly_avg_dep_delay": 0.0,
    "route_monthly_event_count":   0,
    "temp":       15.0,
    "rh":         50.0,
    "precip_hrly": 0.0,
    "wx_phrase":  "Unknown"
})

# %% [markdown]
# ### Build MLlib pipeline.

# %%
# 1. Index columns (by category)
# 2. One-hot encoding
# 3. Assemble all features into a single vector
# 4. Scale numeric part
# 5. Build pipeline
# 6. Fit and transform to get feature vector + label

# %%
## small sample:
sample_df = features_df.sample(withReplacement=False, fraction=0.01, seed=42)
sample_df.cache() # keep it in memory
sample_df.count() # materialize the cache

# %%
# step 1. index
si_stop  = StringIndexer(inputCol="bpuic",    outputCol="stop_idx", handleInvalid="keep")
si_route = StringIndexer(inputCol="line_id",  outputCol="route_idx",handleInvalid="keep")
si_wx    = StringIndexer(inputCol="wx_phrase",outputCol="wx_idx",    handleInvalid="keep")

# step 2. one hot encoding
ohe = OneHotEncoder(inputCols=["stop_idx","route_idx","wx_idx"],outputCols=["stop_vec","route_vec","wx_vec"])

# step 3. assemble
assembler = VectorAssembler(
    inputCols=[
      "stop_vec","route_vec","wx_vec",
      "stop_30d_roll_avg_arr","stop_30d_roll_avg_dep",
      "route_monthly_avg_arr_delay","route_monthly_avg_dep_delay",
      "route_monthly_event_count",
      "temp","rh","precip_hrly"],outputCol="raw_features")

# step 4. scale
scaler = StandardScaler(inputCol="raw_features",outputCol="features",withMean=True,withStd=True)

#step 5. build pipeline

pipeline = Pipeline(stages=[si_stop, si_route, si_wx,ohe, assembler, scaler])

# step 6. fit and transform
model = pipeline.fit(sample_df)
model_features = model.transform(sample_df)
model_features.select("features", "label").show(5)


# %%
model_features.select("features", "label").show(10)

# %%
# Calculate delays (actual - scheduled) for each observation
# Create binary delay indicators (0/1)
delay_flags_df = trips_pivoted_df.select(
    "bpuic",
    "sequence",
    "evt_type",
    F.col("85:151:TL031-4506262507505612").alias("scheduled_time"),
    *[
        # Binary flag: 1 if |actual - scheduled| > 300s (5 mins), else 0
        F.when(
            F.abs(F.col(col) - F.col("85:151:TL031-4506262507505612")) > 300, 1
        ).otherwise(0).alias(f"delay_flag_{col.split('_')[-1]}")
        for col in trips_pivoted_df.columns 
        if '_' in col  # Only actual date columns
    ]
)

# Filter out nulls
delay_flags_df = delay_flags_df.na.drop(subset=["scheduled_time"])

# %%
delay_flags_df.printSchema()
delay_flags_df.select('bpuic','sequence','evt_type', "scheduled_time", "delay_flag_type", "delay_flag_2024-01-03", "delay_flag_2024-02-05", "delay_flag_2024-03-05", "delay_flag_2024-10-07", "delay_flag_2024-11-01").show(100)

# %% [markdown]
# ### Historical Delay Computation

# %%
# Over the last 30 days:
# hourly timestamp on the trip side
ts_df = ts_df.withColumn("ts_hour",F.date_trunc("hour", F.col("start_time")))

# hourly timestampsame on the weather side
weather_df = (weather_df.withColumn("ts_hour",F.to_timestamp(F.concat_ws(" ",F.concat_ws("-", "year", "month", "dayofmonth"),F.format_string("%02d:00:00", F.col("hour"))),"yyyy-M-d HH:mm:ss")))

# picking a random lausanne station LSGZ
wx_site = "LSGZ"
wx_sel  = weather_df.filter(F.col("site") == wx_site).select("ts_hour", "temp", "rh", "precip_hrly", "wx_phrase")
full_df = ts_df.join(wx_sel, on="ts_hour", how="left")

# Minimal exploration
trip_sequences_df.printSchema()
trip_sequences_df.groupBy("trip_id").count().show(5, truncate=False)

# null handling
full_df = (full_df.fillna({"temp": 15.0, "rh": 50.0, "precip_hrly": 0.0}).where("delay_sec <= 8*3600"))

# second historical window (7 days)
hist7_w = Window.partitionBy("trip_id","bpuic").orderBy("operating_day").rowsBetween(-7, -1)
full_df = full_df.withColumn("hist7_rate",F.avg("delay_gt5min").over(hist7_w))

# numerical columns for assembler
numeric_cols = ["hist_delay_rate", "hist_obs","temp", "rh", "precip_hrly", "hour"]
numeric_cols += ["hist7_rate"] 

# %% [markdown]
# ### III.b Model building - 6/20
#
# Utilizing the features generated in section III.a), your objective is to construct a model capable of predicting delays within the Lausanne region.
#
# To accomplish this task effectively:
#
# * Feature Integration:
#         - Incorporate the features created in section III.a) into your modeling pipeline.
#
# * Model Selection and Training:
#         - Explore various machine learning algorithms available in Spark MLlib to identify the most suitable model for predicting delays.
#         - Train the selected model using the feature vectors constructed from the provided data.

# %%
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# %% [markdown]
# ### **After reading through the Spark MLlib, we use the GBT Regressor**

# %%
lausanne_stops.printSchema()
lausanne_stops.show(5, truncate=False)

# %%
lausanne_stops = spark.read.parquet(
    "/data/com-490/labs/assignment-3/sbb_stops_lausanne_region.parquet"
)

stops_bpuic = (
    lausanne_stops
      .select(col("stop_id").alias("bpuic"))
      .distinct()
)
data = model_features.join(stops_bpuic, on="bpuic", how="inner")
data.select("bpuic").distinct().show(5)


# %%
# Split into train and test set
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)

# %%
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=50,          # number of trees
    maxDepth=5,          # depth of each tree
    stepSize=0.1,        # learning rate
    seed=42
)

# %%
# train model
gbt_model = gbt.fit(train_df)

# %% [markdown]
# ### III.c Model evaluation - 6/20
#
# * Evaluate the performance of your model
#     * Usie appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
#     * Utilize techniques such as cross-validation to ensure robustness and generalizability of your model.
#
# * Interpretation and Iteration:
#     * Interpret the results of your model to gain insights into the factors influencing delays within the Lausanne region.
#     * Iterate III.a)on your model by fine-tuning hyperparameters, exploring additional feature engineering techniques, or experimenting with different algorithms to improve predictive performance.
#

# %%
# evaluation metrics

preds = gbt_model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
print(f"Test RMSE = {evaluator.evaluate(preds):.2f} seconds")

# %%
# binary label for delay
data_cls = data.withColumn(
    "delay_label",
    (F.col("label") > 300).cast("integer")
)

#  GBT classification stage from precomupted feature vector
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="delay_label",
    maxIter=20,    # initial guess
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline_cls = Pipeline(stages=[gbt])

# small grid and 5-fold CV for robustness
paramGrid = (ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3,5,7])
    .addGrid(gbt.maxIter,  [10,20,30])
    .build())

binary_evaluator = BinaryClassificationEvaluator(labelCol="delay_label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC")

cv = CrossValidator(estimator=pipeline_cls,estimatorParamMaps=paramGrid,
                    evaluator=binary_evaluator,numFolds=5,parallelism=2)

# train the CV model
cvModel = cv.fit(train_df)

# evaluate on the hold‐out test set
preds = cvModel.transform(test_df)

# AUC ROC
auc = binary_evaluator.evaluate(preds)
print(f"Test AUC-ROC = {auc:.3f}")

# Accuracy / Precision / Recall / F1
multiEval = MulticlassClassificationEvaluator(
    labelCol="delay_label",
    predictionCol="prediction"
)

acc  = multiEval.setMetricName("accuracy").evaluate(preds)
prec = multiEval.setMetricName("weightedPrecision").evaluate(preds)
rec  = multiEval.setMetricName("weightedRecall").evaluate(preds)
f1   = multiEval.setMetricName("f1").evaluate(preds)

print(f"Accuracy = {acc:.2f}")
print(f"Precision = {prec:.2f}")
print(f"Recall = {rec:.2f}")
print(f"F1-score = {f1:.2f}")

# CHeck feature importances
bestModel = cvModel.bestModel.stages[-1]  # GBTClassificationModel
importances = bestModel.featureImportances
assembler = pipeline.stages[-2]  # VectorAssembler
inputCols = assembler.getInputCols()

# Zip and sort top 10
featImps = sorted(
    zip(inputCols, importances.toArray()),
    key=lambda x: x[1],
    reverse=True
)[:10]
print("Top 10 important features!:")
for feat,imp in featImps:
    print(f"   {feat:30s} → {imp:.4f}")

# %%

# %%

# %% [markdown]
# # That's all, folks!
#
# Be nice to other, do not forget to close your spark session. Ж)

# %%
spark.stop()

# %%
