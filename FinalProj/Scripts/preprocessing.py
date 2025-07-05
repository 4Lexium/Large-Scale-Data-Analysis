# +
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
import base64 as b64
import json
import time
import re

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'U1'

WEATHER_DATA_PATH = f"{hadoopFS}/data/com-490/json/weather_history/"


# -

def test():
    return "Imported successfully!"


# +
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

# +
@udf(StringType())
def gmt_to_local_time(unix_seconds):
    if unix_seconds is None:
        return None
    utc_dt = datetime.utcfromtimestamp(unix_seconds).replace(tzinfo=pytz.utc)
    zurich_dt = utc_dt.astimezone(pytz.timezone('Europe/Zurich'))
    return zurich_dt.strftime('%Y-%m-%d %H:%M:%S')

@udf(DoubleType())
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat2), math.radians(lat1)
    delta_phi = math.radians(lat1 - lat2)
    delta_lambda = math.radians(lon1 - lon2)
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Convert each list[int] to bytes
def decode_wkb(wkb_array):
    return shapely.wkb.loads(bytes(wkb_array))


# -

def get_weather_df(spark_obj, year_val=2024):
    df_weather_raw = spark_obj.read.option("multiline", True).json(WEATHER_DATA_PATH)
    df_weather_raw = df_weather_raw.withColumn("observation", explode(col("observations")))

    # Apply timezone UDF
    df_weather_with_time = df_weather_raw.select(
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
    
    # Add extracted fields: year, month, dayofmonth, hour, minute
    weather_df = df_weather_with_time.withColumn("year", year(col("observation_time_tz").cast("timestamp"))) \
                                               .withColumn("month", month(col("observation_time_tz").cast("timestamp"))) \
                                               .withColumn("dayofmonth", dayofmonth(col("observation_time_tz").cast("timestamp"))) \
                                               .withColumn("hour", hour(col("observation_time_tz").cast("timestamp"))) \
                                               .withColumn("minute", minute(col("observation_time_tz").cast("timestamp"))) \
                                               .withColumn("dow", dayofweek(col("observation_time_tz").cast("timestamp")))\
                                               .filter(
                                                   (col("year") == year_val) &
                                                   (col("hour").between(6, 20)) &
                                                   (col("dow").between(2, 6))  # Monday–Friday
                                                )
    # spark_df.createOrReplaceTempView("weather_df")
    return weather_df


def weather_stations_with_region(spark, saveas = 'weather_stations_with_region'):
    
    stations_pd = spark.read.options(header=True).csv(f'/data/com-490/csv/weather_stations').withColumns({
      'lat': F.col('lat').cast('double'),
      'lon': F.col('lon').cast('double'),
    }).toPandas()
    
    # Load polygons from Spark
    geo_df = spark.read.table("iceberg.geo.shapes").select("uuid", "wkb_geometry")
    geo_pd = geo_df.toPandas()
    
    geo_pd["geometry"] = geo_pd["wkb_geometry"].apply(decode_wkb)
    gdf_regions = gpd.GeoDataFrame(geo_pd[["uuid", "geometry"]], geometry="geometry", crs="EPSG:4326")
    
    # Step 1: Load weather stations from Spark
    # stations_pd = spark.table("weather_stations").toPandas()
    
    stations_pd["geometry"] = stations_pd.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    gdf_stations = gpd.GeoDataFrame(stations_pd, geometry="geometry", crs="EPSG:4326")
    gdf_with_region = gpd.sjoin(gdf_stations, gdf_regions, how="left", predicate="within")
    final_df = gdf_with_region.drop(columns=["geometry", "index_right"]).dropna()
    spark_df = spark.createDataFrame(final_df)
    spark_df.createOrReplaceTempView(saveas)

    spark_df.write.mode("overwrite").option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")
    
    return spark_df


def get_daily_precip(spark_obj, year=2024, month = None, precip_threshold=0.1):
    weather = get_weather_df(spark_obj, year_val=year)
    daily_precip_df = weather.groupBy("site", "year", "month", "dayofmonth", "dow") \
                            .agg(spark_sum("precip_hrly").alias("total_daily_precip"))\
                            .withColumn("rained", when(col("total_daily_precip") > precip_threshold, True).otherwise(False)).dropna()
    if month is not None:
        daily_precip_df = daily_precip_df.filter((col("month") == month))
    daily_precip_df.write.mode("overwrite").option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/daily_precip")
    
    return daily_precip_df.dropna()


def get_hourly_weather(spark_obj, year=2024, month = None, precip_threshold=0.1):
    weather = get_weather_df(spark_obj, year_val=year)
    hourly_weather_df = weather.select("site", "year", "month", "dayofmonth", "dow", "hour", "precip_hrly", "temp", "wspd")\
                        .groupBy("site", "year", "month", "dayofmonth", "dow", "hour")\
                        .agg(avg("precip_hrly").alias("avg_precip"),
                             avg("temp").alias("avg_temp"),
                             avg("wspd").alias("avg_wspd"))\
                            .withColumn("rained", when(col("avg_precip") > precip_threshold, True).otherwise(False)).dropna()

    if month is not None:
        hourly_weather_df = daily_precip_df.filter((col("month") == month))

    hourly_weather_df.write.mode("overwrite").option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/get_hourly_weather")
    
    return hourly_weather_df


def get_stops_by_UUID(spark, uuid_arr, saveas='stops'):
    '''
        Given an array with uuid, finds all the stops 
        in the area, saves it to hdfs and returns the 
        dataframe.
    '''
    geo_df = spark.read.table("iceberg.geo.shapes") \
        .filter(lower(col("uuid")).isin(uuid_arr)) \
        .select("uuid","wkb_geometry")
    geo_pd = geo_df.toPandas()

    geo_pd["geometry"] = geo_pd["wkb_geometry"].apply(decode_wkb)
    gdf_regions = gpd.GeoDataFrame(geo_pd, geometry="geometry", crs="EPSG:4326")
    
    stops_df = spark.read.table("iceberg.sbb.stops") \
        .select("stop_id", "stop_name", "stop_lon", "stop_lat").distinct()
    stops_df= stops_df.withColumn(
        "stop_id",
        regexp_replace(col("stop_id"), "^Parent", "")
    )
    stops_pd = stops_df.toPandas().drop_duplicates(["stop_id"])
    stops_pd["geometry"] = stops_pd.apply(lambda row: Point(row["stop_lon"], row["stop_lat"]), axis=1)
    gdf_stops = gpd.GeoDataFrame(stops_pd, geometry="geometry", crs="EPSG:4326")
    
    region_union = gdf_regions.geometry.union_all()
    # gdf_result = gdf_stops[gdf_stops.within(region_union)]
    gdf_result = gpd.sjoin(gdf_stops, gdf_regions[["uuid", "geometry"]], how="inner", predicate="within")
    gdf_result = gdf_result.drop(columns=["geometry", "index_right"])
    filtered_spark_df = spark.createDataFrame(gdf_result).distinct()
    filtered_spark_df.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")
    return filtered_spark_df.drop_duplicates(['stop_id'])


def get_delays_with_weather(spark, saveas="delays_with_weather"):
    '''
        Given an the three datafiles exist, finds all the stops in the area 
        and creates a table with delay information and weather conditions,
        saves it to HDFS and returns the dataframe.
    '''
    df_stops = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops")
    df_stations = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/weather_stations_with_region").distinct()
    df_rain = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/daily_precip").distinct()
    
    df_istdaten = spark.read.table("iceberg.sbb.istdaten")
    df_stop_times = spark.read.table("iceberg.sbb.stop_times")
    
    df_delays = df_istdaten.join(
        df_stops,
        on=col("bpuic").cast("string") == col("stop_id")
    ).filter(
        (col("operating_day") >= "2024-01-01") &
        (col("operating_day") <= "2024-12-31")
    ).select(
        "operating_day", "trip_id", col("bpuic").alias("stop_id"), "stop_name", col("product_id").alias("type"),
        "arr_time", "arr_actual", "dep_time", "dep_actual",
        (unix_timestamp("arr_actual") - unix_timestamp("arr_time")).alias("arr_delay_sec"),
        (unix_timestamp("dep_actual") - unix_timestamp("dep_time")).alias("dep_delay_sec"),
        col("uuid").alias("stops_uuid")
    ).join(
        df_stations,
        on=col("stops_uuid") == col("uuid")
    ).select(
        "operating_day",
        "trip_id", "stop_id", "stop_name", "type",
        "arr_time", "arr_actual", "dep_time", "dep_actual",
        "arr_delay_sec", "dep_delay_sec",
        "Name"
    ).filter(
        (dayofweek(col("operating_day")).between(2, 6)) & # Monday=2 to Friday=6
        (hour(col("arr_time")).between(6, 20)) 
    )

    df_final = df_delays.join(
        df_rain,
        on=(
            (col("Name") == col("site")) &
            (year(col("operating_day")) == col("year")) &
            (month(col("operating_day")) == col("month")) &
            (dayofmonth(col("operating_day")) == col("dayofmonth"))
        ),
        how="left"
    ).select(
        "operating_day", dayofweek(col("operating_day")).alias("dow"),
        "trip_id", "stop_id", "stop_name", "type",
        "arr_time", "arr_actual", "dep_time", "dep_actual",
        "arr_delay_sec", "dep_delay_sec",
        "site", "total_daily_precip", "rained"
    ).dropna().distinct()

    df_final.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")

    return df_final


def get_delays_with_weather_ext(spark, saveas="delays_with_weather_extended"):
    '''
        Given an the three datafiles exist, finds all the stops in the area 
        and creates a table with delay information and weather conditions,
        saves it to HDFS and returns the dataframe.
    '''
    df_stops = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops").select(
        "stop_id", col("stop_name").alias("sn"), "stop_lon", "stop_lat", "uuid"
    )
    df_stations = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/weather_stations_with_region").distinct()
    df_rain = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/get_hourly_weather").distinct()
    
    df_istdaten = spark.read.table("iceberg.sbb.istdaten")
    df_stop_times = spark.read.table("iceberg.sbb.stop_times")
    
    df_delays = df_istdaten.join(
        df_stops,
        on=col("bpuic").cast("string") == col("stop_id")
    ).filter(
        (col("operating_day") >= "2024-01-01") &
        (col("operating_day") <= "2024-12-31")
    ).select(
        "operating_day", "trip_id", col("bpuic").alias("stop_id"), "stop_name",
        col("product_id").alias("type"), "arr_time", "arr_actual", "dep_time", "dep_actual",
        (unix_timestamp("arr_actual") - unix_timestamp("arr_time")).alias("arr_delay_sec"),
        (unix_timestamp("dep_actual") - unix_timestamp("dep_time")).alias("dep_delay_sec"),
        col("uuid").alias("stops_uuid")
    ).join(
        df_stations,
        on=col("stops_uuid") == col("uuid")
    ).select(
        "operating_day",
        "trip_id", "stop_id", "stop_name", "type",
        "arr_time", "arr_actual", "dep_time", "dep_actual",
        "arr_delay_sec", "dep_delay_sec",
        "Name"
    ).filter(
        (dayofweek(col("operating_day")).between(2, 6)) & # Monday=2 to Friday=6
        (hour(col("arr_time")).between(6, 20)) 
    )

    df_final = df_delays.join(
        df_rain,
        on=(
            (col("Name") == col("site")) &
            (year(col("operating_day")) == col("year")) &
            (month(col("operating_day")) == col("month")) &
            (dayofmonth(col("operating_day")) == col("dayofmonth")) &
            (hour(col("arr_time")) == col("hour"))
        ),
        how="left"
    ).select(
        "operating_day", dayofweek(col("operating_day")).alias("dow"), 
        "trip_id", "stop_id","stop_name", "type",
        "arr_time", "arr_actual", "dep_time", "dep_actual",
        "arr_delay_sec", "dep_delay_sec",
        "site", "avg_wspd", "avg_temp", "avg_precip", "rained"
    ).dropna().distinct()

    df_final.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")

    return df_final


def get_trips(spark, saveas="trips_with_sequence"):
    stops_df = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/stops").select("stop_id", "stop_name")

    
    #only keep service running on weekdays
    calendar_df = spark.read.table("iceberg.sbb.calendar").filter( 
        (col("pub_date") >= "2024-01-01") &
        (col("pub_date") <= "2024-12-31")
    ).filter(
        (col("monday") == 1) |
        (col("tuesday") == 1) |
        (col("wednesday") == 1) |
        (col("thursday") == 1) |
        (col("friday") == 1)
    ).select(
        col("service_id").alias("c_service_id"),
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday"
    ).distinct()
    
    
    trips_df = spark.read.table("iceberg.sbb.trips").filter(
        (col("pub_date") >= "2024-01-01") &
        (col("pub_date") <= "2024-12-31")
    ).select(
        "trip_id", "route_id", "service_id"
    ).join( # drop trips only running on weekends
        calendar_df,
        on= col("service_id")==col("c_service_id"),
        how= "inner"
    ).select(
        "trip_id", "route_id",        
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday"
    )
    
    
    stop_times_df = spark.read.table("iceberg.sbb.stop_times")\
    .filter(
        (col("pub_date") >= "2024-01-01") &
        (col("pub_date") <= "2024-12-31") &
        (hour(col("arrival_time")).between(6, 20))
    ).select(
        col("trip_id").alias("st_trip_id"), "arrival_time", "departure_time", col("stop_id").alias("st_stop_id"), "stop_sequence"
    ).join(
        stops_df,
        on = col("stop_id")==col("st_stop_id"), # only keep the stips in the required region
        how = "inner"
    ).select(
        "st_trip_id", "arrival_time", "departure_time", "stop_id", "stop_name", "stop_sequence"
    ).distinct()
    
    df_final = stop_times_df.join(
        trips_df,
        on=col("trip_id")==col("st_trip_id"),
        how= "inner"
        ).select(
            "route_id", "trip_id", "arrival_time", 
            "departure_time", "stop_id", "stop_name", 
            "stop_sequence",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday"
        ).distinct()

    df_final.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")

    return df_final


def get_trips_v2(spark, saveas="schedule"):

    df = spark.read.option("header", True).parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")\
                                          .select("trip_id", "stop_id", "stop_name", "dow",
                                                  col("arr_time").alias("planned_arr"),
                                                  col("dep_time").alias("planned_dep"))# read historical data
    
    df_with_time = df.withColumn("arr_time", date_format("planned_arr", "HH:mm:ss")) \
                     .withColumn("dep_time", date_format("planned_dep", "HH:mm:ss")) \
                     .select("trip_id", "stop_id", "stop_name", "dow", "arr_time", "dep_time") \
                     .drop_duplicates(['trip_id', 'dow', 'arr_time'])
    
    window_spec = Window.partitionBy("trip_id", "dow").orderBy("arr_time")

    # Compute stop sequence
    df_final = df_with_time.withColumn("stop_sequence", row_number().over(window_spec))
    df_final.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")
    return df_final


def get_trips_v3(spark, saveas="schedule_v3"):

    df = get_trips(spark)
    
    days = array(
            struct(lit(2).alias("dow"), col("monday").alias("active")),     # Monday
            struct(lit(3).alias("dow"), col("tuesday").alias("active")),    # Tuesday
            struct(lit(4).alias("dow"), col("wednesday").alias("active")),  # Wednesday
            struct(lit(5).alias("dow"), col("thursday").alias("active")),   # Thursday
            struct(lit(6).alias("dow"), col("friday").alias("active"))      # Friday
        )
        
    # Add exploded day column
    df_expanded = df.withColumn("day_struct", explode(days)) \
                    .withColumn("dow", col("day_struct.dow")) \
                    .withColumn("active", col("day_struct.active")) \
                    .drop("day_struct")
    
    # Filter only rows where active is true
    df_result = df_expanded.filter(col("active") == True).drop("monday", "tuesday", "wednesday", "thursday", "friday", "active").drop_duplicates()

    df_result.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")
    
    return df_result


def edges(spark, df, saveas="edges"):

    #remove duplicates keep the trip_id relationships
    # Define window spec to order rows within duplicates
    window = Window.partitionBy('route_id','arrival_time','departure_time','stop_id','stop_sequence','dow').orderBy("trip_id")
    
    # Add row number
    df_with_rank = df.withColumn("rn", F.row_number().over(window))
    
    # Keep only the first occurrence
    df= df_with_rank.filter("rn = 1").drop("rn")
    
    
    # Add "instance" to account for repeating trip_id
    # Add "instance" to account for repeating trip_id
    window_spec1 = Window.partitionBy("trip_id", "stop_sequence") \
                        .orderBy("arrival_time")
    
    # Use rank() to order them chronologically with smallest = 1
    df_with_instance = df.withColumn("instance", F.rank().over(window_spec1))
    df = df_with_instance

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

    #drop duplicate trips
    result=result.dropDuplicates(['route_id','from_stop','to_stop','T_nominal','t_arrival','dow'])
    
    result.write \
        .mode("overwrite") \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/{saveas}")
    return result
