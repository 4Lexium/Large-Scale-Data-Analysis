# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # DSLab Assignment 2 - Data Wrangling with Hadoop
# ---
#
# ## Hand-in Instructions
#
# - __Due: **.04.2025 23h59 CET__
# - Create a fork of this repository under your group name, if you do not yet have a group, you can fork it under your username.
# - `git push` your final verion to the master branch of your group's repository before the due date.
# - Set the group name variable below, e.g. group_name='Z9'
# - Add necessary comments and discussion to make your codes readable.
# - Let us know if you need us to install additional python packages.
#
# ## Useful references
#
# * [Trino documentation](https://trino.io/docs/471)
# * [Enclosed or Unenclosed](https://github.com/Esri/spatial-framework-for-hadoop/wiki/JSON-Formats)

groupName='U1'

# ---
# ⚠️ **Note**: all the data used in this homework is described in the [FINAL-PREVIEW](./final-preview.md) document, which can be found in this repository. The document describes the final project due for the end of this semester.
#
# For this notebook you are free to use the following tables, which can all be found under the _iceberg.com490_iceberg_ namespace shared by the class (you may use the sharedNS variable).
# - You can list the tables with the command `f"SHOW TABLES IN {sharedNS}"`.
# - You can see the details of each table with the command `f"DESCRIBE {sharedNS}.{table_name}"`.
#
# ---
# For your convenience we also define useful python variables:
#
# * _hadoop_fs_
#     * The HDFS server, in case you need it for hdfs, pandas or pyarrow commands.
# * _username_:
#     * Your user id (EPFL gaspar id), use it as your personal namespace for your private tables.
# * _sharedNS_:
#     * The namespace of the tables shared by the class. **DO NOT** modify or drop tables in this namespace, or drop the namespace.
# * _namespace_:
#     * Your personal namespace.

# <div style="font-size: 100%" class="alert alert-block alert-warning">
#     <b>Fair cluster Usage:</b>
#     <br>
#     As there are many of you working with the cluster, we encourage you to prototype your queries on small data samples before running them on whole datasets. Do not hesitate to partion your tables, and LIMIT the output of your queries to a few rows to begin with. You are also free to test your queries using alternative solutions such as <i>DuckDB</i>.
#     <br><br>
#     You may lose your session if you remain idle for too long or if you interrupt a query. If that happens you will not lose your tables, but you may need to reconnect to the warehouse.
#     <br><br>
#     <b>Try to use as much SQL as possible and avoid using pandas operations.</b>
# </div>

# +
import os
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")

# +
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


# +
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
# -

# ---

# +
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

# +
import pandas as pd

table = pd.read_sql(f"""SHOW TABLES IN {sharedNS}""", conn)
table


# -

# ## Part I. 10 Points

# ### a) Declare an SQL result generator - 2/10
#
# Complete the Python generator below to execute a single query or a list of queries, returning the results row by row.
#
# The generator should implement an out-of-core approach, meaning it should limit memory usage by fetching results incrementally, rather than loading all data into memory at once.

def sql_fetch(queries, conn, batch_size=100):
    if isinstance(queries, str): # Wrap single query string in a list to standardize processing
        queries = [queries]
    with closing(conn.cursor()) as cur:
        cursor = conn.cursor()
    
        for query in queries:
            cursor.execute(query)
    
            while (rows := cursor.fetchmany(batch_size)):
                for row in rows:
                    yield row # Stream one-by-one row to the caller


# ### b) Explore SBB data - 3/10
#
# Explore the _{sharedNS}.sbb_istdaten_, _{sharedNS}.sbb_stops_, and _{sharedNS}.sbb_stop_times_ tables.
#
# Identify the field(s) used across all three tables to represent stop locations. Analyze their value ranges, format patterns, null and invalid values, and identify any years when null or invalid values are more prevalent. Use this information to implement the necessary transformations for reliably joining the tables on these stop locations.

# Create namespace where we will save tables
with closing(conn.cursor()) as cur:
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS hive.{username}')
    cur.fetchone()
    cur.execute(f"USE hive.{username}")
    cur.fetchone()

# 1. **Explore the _{sharedNS}.sbb_istdaten_, _{sharedNS}.sbb_stops_, and _{sharedNS}.sbb_stop_times_ tables.**

pd.read_sql(f"""DESCRIBE {sharedNS}.sbb_istdaten""", conn)

pd.read_sql(f"""DESCRIBE {sharedNS}.sbb_stops""", conn)

pd.read_sql(f"""DESCRIBE {sharedNS}.sbb_stop_times""", conn)

# 2. **Identify the field(s) used across all three tables to represent stop locations. Analyze their value ranges, format patterns.**

# - From the _DESCRIBE_ query, we found fields that may match: `trip_id`, `bpuic`, `stop_name`, `stop_id`

# Query to compare range values, format patterns
q_joint = f"""
SELECT 
    stops.stop_id AS stops_stop_id,
    times.stop_id AS stoptimes_stop_id,
    ist.bpuic AS istdaten_bpuic,
    ist.trip_id AS istdaten_trip_id,
    times.trip_id AS stoptimes_trip_id,
    ist.stop_name AS istdaten_stop_name,
    stops.stop_name AS stops_stop_name
FROM {sharedNS}.sbb_istdaten AS ist
JOIN {sharedNS}.sbb_stop_times AS times
    ON CAST(ist.bpuic AS VARCHAR) = times.stop_id
JOIN {sharedNS}.sbb_stops AS stops
    ON split_part(stops.stop_id, ':', 1) = CAST(ist.bpuic AS VARCHAR)
WHERE ist.stop_name LIKE 'Lausanne'
LIMIT 5
"""
df_joint = pd.read_sql(q_joint, conn)
df_joint

# - `{sharedNS}.sbb_istdaten` and `{sharedNS}.sbb_stop_times` have completely different trip_id formats (GFTS vs. internal).
# - `{sharedNS}.sbb_stops` and `{sharedNS}.sbb_stop_times` have the same `stop_id`, however `{sharedNS}.sbb_stops` has a minor difference
# - The link between all three iceberg tables are the fields: `sbb_istdaten.bpuic`-> `sbb_stop_times.stop_id` -> `stops_stop_id`

# - **Implement the necessary transformations for reliably joining the tables on these stop locations.**
#
# 1. Strip all hexadecimal suffixes from `{sharedNS}.sbb_stop_times` to ensure same formatting with the rest
# 2. Rename `istdaten_bpuic` to `stop_id` so it is understandable what it is
# 3. Find null and invalid values, and identify any years when null or invalid values are more prevalent.

q_stop_id_nulls_invalids = f"""
WITH istdaten AS (
    SELECT 
        year(operating_day) AS year,
        'sbb_istdaten' AS source_table,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN bpuic IS NULL THEN 1 ELSE 0 END) AS null_stop_id,
        0 AS invalid_stop_id  -- integers can't have these patterns
    FROM {sharedNS}.sbb_istdaten
    GROUP BY year(operating_day)
),

stoptimes AS (
    SELECT 
        year(pub_date) AS year,
        'sbb_stop_times' AS source_table,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN stop_id IS NULL THEN 1 ELSE 0 END) AS null_stop_id,
        SUM(
            CASE 
                WHEN stop_id IS NOT NULL AND (
                    LENGTH(TRIM(stop_id)) <= 2 OR
                    REGEXP_LIKE(stop_id, '^Parent.*') OR
                    REGEXP_LIKE(stop_id, '^[^a-zA-Z0-9]+$') OR
                    REGEXP_LIKE(stop_id, '.*:.*:.*')
                )
                THEN 1 ELSE 0 
            END
        ) AS invalid_stop_id
    FROM {sharedNS}.sbb_stop_times
    GROUP BY year(pub_date)
),

stops AS (
    SELECT 
        year(pub_date) AS year,
        'sbb_stops' AS source_table,
        COUNT(*) AS total_rows,
        SUM(CASE WHEN stop_id IS NULL THEN 1 ELSE 0 END) AS null_stop_id,
        SUM(
            CASE 
                WHEN stop_id IS NOT NULL AND (
                    LENGTH(TRIM(stop_id)) <= 2 OR
                    REGEXP_LIKE(stop_id, '^Parent.*') OR
                    REGEXP_LIKE(stop_id, '^[^a-zA-Z0-9]+$') OR
                    REGEXP_LIKE(stop_id, '.*:.*:.*')
                )
                THEN 1 ELSE 0 
            END
        ) AS invalid_stop_id
    FROM {sharedNS}.sbb_stops
    GROUP BY year(pub_date)
)

SELECT * FROM istdaten
UNION ALL
SELECT * FROM stoptimes
UNION ALL
SELECT * FROM stops
ORDER BY year, source_table
"""

# +
from tabulate import tabulate

# Collect rows
rows = list(sql_fetch(q_stop_id_nulls_invalids, conn))

# Print as a table
print(tabulate(
    rows,
    headers=["Year", "Table", "Total IDs", "NULLs", "Invalid IDs"],
    tablefmt="pretty"
))

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Our findings
# - There are luckily only 97 nulls in `sbb_istdaten` in 2021, which is negligible.
#
# - `sbb_stop_times` has **millions** of Invalid ID's every single year, obviously due to the hexadecimal suffixes used for the cluster of stops in the Lausanne region being in the same geolocation. We would need to strip all these suffixes to safely join the tables.
# - `sbb_stops` has a fair share of invalid ID's as well every year, however more prevalent in 2024.
# - **2024** is the year with most invalid ID's
# - `sbb_istdaten` has no invalid ID's, which is expected since `bpuic`is an integer and contains no junk.

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### c) Type of transport - 5/10
#
# Explore the distribution of _product_id_ in _{sharedNS}.sbb_istdaten_ for the whole of 2024 and visualize it in a bar graph.
#
# - Query the istdaten table to get the total number of stop events for different types of transport in each month.
# |year|month|product|stops|
# |---|---|---|---|
# |...|...|...|...|
# - Create a facet bar chart of monthly counts, partitioned by the type of transportation. 
# - If applicable, document any patterns or abnormalities you can find.
#
# __Note__: 
# - One entry in the sbb istdaten table means one stop event, with information about arrival and departure times.
# - We recommend the facet _bar_ plot with plotly: https://plotly.com/python/facet-plots/ the monthly count of stop events per transport mode as shown below (the number of _product_id_ may differ):
#
# ```
# fig = px.bar(
#     df_ttype, x='month_year', y='stops', color='ttype',
#     facet_col='ttype', facet_col_wrap=3, 
#     facet_col_spacing=0.05, facet_row_spacing=0.2,
#     labels={'month_year':'Month', 'stops':'#stops', 'ttype':'Type'},
#     title='Monthly count of stops'
# )
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_layout(showlegend=False)
# fig.show()
# ```
#
#
# <img src="./figs/1a-example.png" alt="1a-example.png" width="800"/>
# -

# ### c) Type of transport - 5/10
#
# Explore the distribution of _product_id_ in _{sharedNS}.sbb_istdaten_ for the whole of 2024 and visualize it in a bar graph.
#
# - Query the istdaten table to get the total number of stop events for different types of transport in each month.
# |year|month|product|stops|
# |---|---|---|---|
# |...|...|...|...|
# - Create a facet bar chart of monthly counts, partitioned by the type of transportation. 
# - If applicable, document any patterns or abnormalities you can find.
#
# __Note__: 
# - One entry in the sbb istdaten table means one stop event, with information about arrival and departure times.
# - We recommend the facet _bar_ plot with plotly: https://plotly.com/python/facet-plots/ the monthly count of stop events per transport mode as shown below (the number of _product_id_ may differ):
#
# ```
# fig = px.bar(
#     df_ttype, x='month_year', y='stops', color='ttype',
#     facet_col='ttype', facet_col_wrap=3, 
#     facet_col_spacing=0.05, facet_row_spacing=0.2,
#     labels={'month_year':'Month', 'stops':'#stops', 'ttype':'Type'},
#     title='Monthly count of stops'
# )
# fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
# fig.update_yaxes(matches=None, showticklabels=True)
# fig.update_layout(showlegend=False)
# fig.show()
# ```
#
#
# <img src="./figs/1a-example.png" alt="1a-example.png" width="800"/>

pd.read_sql(
        f"""
        SELECT product_id, COUNT(*) AS num
        FROM iceberg.com490_iceberg.sbb_istdaten GROUP BY product_id
        """,
        conn)

# __Comment__:
# - Bus and BUS, the same but capitalization. We convert product_id to lower case, since there are inconsistencies.
# - We drop the invalid values, which are NULL and ''. Row 10 is unknown for instance.

pd.read_sql(
    f"""
SELECT lower(product_id) AS mode, count(*) AS num
    FROM iceberg.com490_iceberg.sbb_istdaten
    WHERE
        product_id IS NOT NULL
        AND product_id <> ''
    GROUP BY lower(product_id)
""", conn)

q_create_cleaned_table = f"""
CREATE OR REPLACE TABLE {namespace}.istdaten_modes_2024 AS
SELECT
    trim(lower(product_id)) AS mode,
    date_trunc('month', operating_day) AS month_year,
    count(*) AS num_stops
FROM iceberg.com490_iceberg.sbb_istdaten
WHERE
    product_id IS NOT NULL
    AND trim(product_id) <> ''
    AND year(operating_day) = 2024
GROUP BY
    trim(lower(product_id)),
    date_trunc('month', operating_day)
"""
pd.read_sql(q_create_cleaned_table, conn)

pd.read_sql(f"""
SELECT *
FROM {namespace}.istdaten_modes_2024
WHERE mode = 'wm-bus'
ORDER BY month_year
""", conn)

import plotly.express as px
q_plot = f"""
SELECT *
FROM {namespace}.istdaten_modes_2024
WHERE mode IN ('bus', 'tram', 'zug', 'wm-bus', 'taxi', 'metro', 'schiff', 'zahnradbahn', 'stadtbahn', 'cs', 'standseilbahn')
ORDER BY mode, month_year
"""
df_ttype = pd.read_sql(q_plot, conn)
#plotly sucks and is ignoring facets with one x-axis value, so we just add a dummy row to include wm-bus.
wm_extra = pd.DataFrame([{'mode': 'wm-bus','month_year': pd.Timestamp('2024-02-01'),'num_stops': 0}]) 
df_ttype = pd.concat([df_ttype, wm_extra], ignore_index=True)

# +
fig = px.bar(
    df_ttype,
    x='month_year',y='num_stops',
    color='mode',facet_col='mode',
    facet_col_wrap=5,facet_col_spacing=0.05,
    facet_row_spacing=0.15,
    labels={'month_year': 'Month','num_stops': '# Stops','mode': 'Transport Type'},
    title='Monthly Count of Stops by Transport Type (2024)'
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None, showticklabels=True)
fig.update_layout(showlegend=False)
fig.show()
# -

# ## Part II. 50 Points

# In this second Part, we will leverage the historical SBB data to model the public transport infrastructure within the Lausanne region.
#
# Our objective is to establish a comprehensive data representation of the public transport network, laying the groundwork for our final project. While we encourage the adoption of a data structure tailored to the specific requirements of your final project implementation, the steps outlined here provide a valuable foundation.
#
# In this part you will make good use of DQL statements of nested SELECT, GROUP BY, JOIN, IN, DISTINCT, and Geo Spatial UDF.

# You must create a managed database within your designated namespace, where you'll define the tables necessary for modeling your infrastructure. By 'managed,' we mean that you should not specify a default external location for the namespace.
#
# While it's a bit of an overkill, the safest approach is to drop and recreate the namespace the first time you run this notebook.

list(sql_fetch([
    f"""DROP SCHEMA IF EXISTS {namespace} CASCADE""", # CASCADE will drop all the tables
    f"""CREATE SCHEMA IF NOT EXISTS {namespace}""",
], conn))

# ### a) Find the stops in Lausanne region - 5/50
#
#

# * Explore _{sharedNS}.geo_ and find the records containing the _wkb_geometry_ shapes of the _Lausanne_ and _Ouest lausannois_ districts.
#      * The shape is from swiss topo
# * Find all the stops in the _Lausanne_ district from _{sharedNS}.sbb_stops_, as of the first week of July 2024 (use [geo spatial](https://trino.io/docs/471/functions/geospatial.html) functions)
# * Save the results into a table _{namespace}.sbb_stops_lausanne_region_ using the CTAS (Create Table As Select) approach.
# * Validation: you should find around $400\pm 25$ stops.
# * Table _{namespace}.sbb_stops_lausanne_region_ is a subset of table _{sharedNS}.sbb_stops_:
#     * _stop_id_
#     * _stop_name_
#     * _stop_lat_
#     * _stop_lon_

# ---
# ### Solutions
#
# - **Explore _{sharedNS}.geo_ and find the records containing the _wkb_geometry_ shapes of the _Lausanne_ and _Ouest lausannois_ districts.**

pd.read_sql(f"""DESCRIBE {sharedNS}.geo""", conn)

pd.read_sql(f"""
SELECT stop_name, stop_lon, stop_lat
FROM {sharedNS}.sbb_stops
WHERE stop_name LIKE 'Lausanne'
LIMIT 1
""" , conn)

pd.read_sql(f"""
SELECT name, region, wkb_geometry, uuid
FROM {sharedNS}.geo
WHERE name LIKE 'Lausanne'
LIMIT 1
""" , conn)

pd.read_sql(f"""
SELECT name, region, wkb_geometry, uuid
FROM {sharedNS}.geo
WHERE name LIKE 'Ouest lausannois'
LIMIT 1
""" , conn)

# - **Find all the stops in the _Lausanne_ district from _{sharedNS}.sbb_stops_, as of the first week of July 2024 (use [geo spatial](https://trino.io/docs/471/functions/geospatial.html) functions). Save the results into a table _{namespace}.sbb_stops_lausanne_region_ using the CTAS (Create Table As Select) approach.**

# In the query, we pick out stuff from {sharedNS}.sbb_stops and {sharedNS}.geo to match the variables
# We use geospatial functions to extract all of the stops in Lausanne and Ouest lausannois region
# Creating table in a CTAS approach
q_ctas = f"""
CREATE OR REPLACE TABLE {namespace}.sbb_stops_lausanne_region AS
SELECT DISTINCT stop_id, stop_name, stop_lon, stop_lat
FROM {sharedNS}.sbb_stops s
JOIN (
    SELECT ST_GeomFromBinary(wkb_geometry) AS region_geom
    FROM {sharedNS}.geo
    WHERE lower(name) IN ('lausanne', 'ouest lausannois')
) g
ON ST_Contains(g.region_geom, ST_Point(s.stop_lon, s.stop_lat))
WHERE s.pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-07' 
  AND NOT REGEXP_LIKE(stop_id, '^Parent[0-9A-Fa-f]+$')
  AND NOT REGEXP_LIKE(stop_id, '^[0-9A-Fa-f]+(:[0-9A-Fa-f]+)+$')
"""
pd.read_sql(q_ctas, conn)

# **Validation:** We do indeed have $400 \pm 25$ stops, as was expected. 
#
# Here we chose to remove the _PARENT_ stops (clusters of many stops in the same region), to ensure we only have unique stops.
#
# Also, as seen earlier, some of the stop id's have weird formatting with hexadecimales, so clean it up.

# Final table of all stops as of 1st week of July
pd.read_sql(f"SHOW TABLES IN {namespace}", conn)
pd.read_sql(f"SELECT * FROM {namespace}.sbb_stops_lausanne_region", conn)

# This table is now a subset of the larger sbb_stops table with only the relevant information.

# ### b) Find stops with real time data in Lausanne region - 5/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the stops for which real time data is reported in the _{sharedNS}.sbb_istdaten_ table for the full month of **July 2024**.
# * Report the results in a pandas DataFrame that you will call _stops_df_.
# * Validation: you should find between 3% and 4% of stops in the area of interest that do not have real time data.
# * Hint: it is recommended to first generate a list of _distinct_ stop identifiers extracted from istdaten data. This can be achieved through either a nested query or by creating an intermediate table (use your findings of Part I.b).

# ---
# #### Solution
#

# +
#generate a list of distinct stop identifiers extracted from istdaten for the full month of July 2024
q_distinct_istdaten_ids = f"""
CREATE OR REPLACE TABLE {namespace}.istdaten_stop_ids_july2024 AS
SELECT DISTINCT CAST(bpuic AS VARCHAR) AS stop_id
FROM {sharedNS}.sbb_istdaten
WHERE operating_day BETWEEN DATE '2024-07-01' AND DATE '2024-07-31'
"""
pd.read_sql(q_distinct_istdaten_ids,conn)

# using the results of table _{username}.sbb_stops_lausanne_region_ to find all the stop with real time data

q_realtime = f"""
WITH realtime AS (
    SELECT DISTINCT stop_id
    FROM {namespace}.istdaten_stop_ids_july2024
),
lausanne_stops AS (
    SELECT DISTINCT stop_id
    FROM {namespace}.sbb_stops_lausanne_region
)
SELECT
    COUNT(*) AS total_stops,
    COUNT(r.stop_id) AS with_realtime,
    COUNT(*) - COUNT(r.stop_id) AS without_realtime,
    ROUND(100.0 * (COUNT(*) - COUNT(r.stop_id)) / COUNT(*), 2) AS percent_without_realtime
FROM lausanne_stops l
LEFT JOIN realtime r
  ON l.stop_id = r.stop_id
"""
# -

# %%time
stops_df = pd.read_sql(q_realtime, conn)

### TODO - Verify the results
stops_df

# - **Findings:** We see indeed 4%, or 15 stops in total, are missing realtime, as seen in the fourth column. This is as expected from the validation criterion

# ### c) Display stops in the Lausanne Region - 3/50
#
# * Use plotly or similar plot framework to display all the stop locations in Lausanne region on a map (scatter plot or heatmap), using a different color to highlight the stops for which istdaten data is available.

# +
# query that compares regions with realtime data and without
df_compare = pd.read_sql(f"""
SELECT 
    r.stop_id, r.stop_name, r.stop_lat, r.stop_lon,
    CASE 
        WHEN i.stop_id IS NOT NULL THEN 'has_data'
        ELSE 'no_data'
    END AS realtime_status
FROM {namespace}.sbb_stops_lausanne_region r
LEFT JOIN {namespace}.istdaten_stop_ids_july2024 i
  ON r.stop_id = i.stop_id
""", conn)

# scatterplot of all locations in Lausanne
fig = px.scatter_mapbox(
    df_compare,
    lat="stop_lat", lon="stop_lon",
    color="realtime_status", hover_name="stop_name",
    mapbox_style="carto-positron",
    zoom=10.5, height=700, width=900,
    title="Lausanne Region Stops — Real-time Data Availability (July 2024)"
)

fig.show()
# -

# Note that some stops lacking real-time data may actually serve as representations for groups of stops, like `Parent8592050` for Lausanne Gare, which denotes a cluster of stops within Lausanne train station. We ignore these.

# ### d) Find stops that are within walking distances of each other - 10/50

# * Use the results of table _{username}.sbb_stops_lausanne_region_ to find all the (directed) pair of stops that are within _500m_ of each other.
# * Save the results in table _{username}.sbb_stops_to_stops_lausanne_region_
# * Validation: you should find around $3500\pm 250$ directed stop paris (each way, i.e. _A_ to _B_ and _B_ to _A_).
# * Hint: Use the Geo Spatial UDF, in spherical geopgraph.
# * Aim for the table _{namespace}.sbb_stop_to_stop_lausanne_region_:
#     * _stop_id_a_: an _{sharedNS}.sbb_stops.stop_id_
#     * _stop_id_b_: an _{sharedNS}.sbb_stops.stop_id_
#     * _distance_: straight line distance in meters from _stop_id_a_ to _stop_id_b_

# %%time
q_stops_distance = f"""
CREATE OR REPLACE TABLE {namespace}.sbb_stops_to_stops_lausanne_region AS
SELECT 
    a.stop_id AS stop_id_a,
    a.stop_name AS stop_name_a,
    b.stop_id AS stop_id_b,
    b.stop_name AS stop_name_b,
    ST_Distance(
        to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
        to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
    ) AS distance_meters
FROM {sharedNS}.sbb_stops_lausanne_region a
JOIN {sharedNS}.sbb_stops_lausanne_region b
    ON a.stop_id != b.stop_id
WHERE ST_Distance(
        to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
        to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
    ) < 500
"""


pd.read_sql(f"SELECT * FROM {sharedNS}.sbb_stops_lausanne_region", conn)


# +
### TODO - Verify the results
pd.read_sql(q_stops_distance, conn)

# Final table of all stops as of 1st week of July
pd.read_sql(f"SHOW TABLES IN {namespace}", conn)
walking_pairs = pd.read_sql(f"SELECT * FROM {namespace}.sbb_stops_to_stops_lausanne_region", conn)
walking_pairs
# -

# __Validation__:
# We see that we have $3640$ pairs, which is inside the threshold of $3500 \pm 250$.

# ### e) Finds the _stop times_ in Lausanne region - 10/50
#
# * Find the stop times and weekdays of trips (trip_id) servicing stops found previously in the Lausanne region.
# * Use the stop times and calendar information published on the same week as the stops information used to compute the stops in the Lausanne region.
# * Save the results in the table _{username}.sbb_stop_times_lausanne_region_
# * Validation: you should find around $1M\pm 50K$ trip_id, stop_id pairs in total, out of which $450K\pm 25K$ happen on Monday.
#
# At a minimum, the table should be as follow. Use the provided information to decide the best types for the fields.
#
# * _{namespace}.sbb_stop_times_lausanne_region_ (subset of _{sharedNS}.sbb_stop_times_ and _{sharedNS}.sbb_calendar_).
#     * _trip_id_
#     * _stop_id_
#     * _departure_time_
#     * _arrival_time_
#     * _monday_ (trip happens on Monday)
#     * _tuesday_
#     * _wednesday_
#     * _thursday_
#     * _friday_
#     * _saturday_
#     * _sunday_
#  
# **Hints:**
# * Pay special attention to the value ranges of the _departure_time_ and _arrival_time_ fields in the _{sharedNS}.sbb_stop_times_ table.
# * This new table will be used in the next exercise for a routing algorithm. We recommend reviewing the upcoming questions to determine the appropriate data types and potential transformations for the _departure_time_ and _arrival_time_ fields.

# %%time
pd.read_sql(f"""
CREATE OR REPLACE TABLE {namespace}.sbb_stop_times_lausanne_region AS
    SELECT DISTINCT * 
    FROM (
        SELECT 
            st.trip_id,
            st.stop_id,
            st.stop_sequence,
            st.arrival_time,
            st.departure_time,
            c.monday,
            c.tuesday,
            c.wednesday,
            c.thursday,
            c.friday,
            c.saturday,
            c.sunday
        FROM 
            {sharedNS}.sbb_stop_times st
        JOIN 
            {sharedNS}.sbb_trips t 
                ON st.trip_id = t.trip_id
        JOIN 
            {sharedNS}.sbb_calendar c 
                ON t.service_id = c.service_id
        JOIN 
            {namespace}.sbb_stops_lausanne_region lausanne 
                ON st.stop_id = lausanne.stop_id
        WHERE
            st.pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-07'
            AND t.pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-07'
            AND c.pub_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-07'
        )
""", conn)


# +
# %%time
### TODO - Verify the results
monday_true_count = pd.read_sql(
    f"""
    SELECT COUNT(*) AS monday_true_count
    FROM (
        SELECT DISTINCT trip_id, stop_id
        FROM {namespace}.sbb_stop_times_lausanne_region
        WHERE monday = TRUE
    )
    """, 
    conn
)

count_pairs = pd.read_sql(
    f"""
    SELECT COUNT(*) AS unique_pair_count
    FROM (
        SELECT DISTINCT trip_id, stop_id
        FROM {namespace}.sbb_stop_times_lausanne_region
    )
    """, 
    conn
)

print(f"Number of unique trip_id-stop_id: {count_pairs['unique_pair_count'][0]}")
print(f"Number of entries where Monday is True: {monday_true_count['monday_true_count'][0]}")

# +
# %%time
min_departure_time = '18:00:00'
max_arrival_time = '20:00:00'

smth_df = pd.read_sql(f"""
    SELECT *
    FROM {namespace}.sbb_stop_times_lausanne_region
    WHERE monday = TRUE AND departure_time > '{min_departure_time}' AND arrival_time < '{max_arrival_time}'
    """, conn)
# -

count_pairs_all = pd.read_sql(f"""
    SELECT COUNT(*) AS unique_pair_count
    FROM (
    WITH ranked_data AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY 
                   stop_id, stop_sequence, arrival_time, departure_time,
                   monday, tuesday, wednesday, thursday, friday, saturday, sunday
               ORDER BY trip_id
           ) AS row_num
    FROM {namespace}.sbb_stop_times_lausanne_region
    )
    SELECT 
        trip_id,stop_id, stop_sequence, arrival_time, departure_time,
        monday, tuesday, wednesday, thursday, friday, saturday, sunday
    FROM ranked_data
    WHERE row_num = 1
    ORDER BY trip_id, stop_sequence, departure_time)
    """, conn)
print(count_pairs_all['unique_pair_count'])

# ### f) Design considerations - 2/50
#
# We aim to use our previous findings to recommend an optimal public transport route between two specified locations at a given time on any day of a particular week in any region.
#
# Running queries on all data for the entire data set would be inefficient. Could you suggest an optimized table structure to improve the efficiency of queries on the {username}.sbb_stop_times_lausanne_region table?

# + [markdown] jp-MarkdownHeadingCollapsed=true
# __ANSWER:__ We first see that there are multiple trips that have different different `trip_id` but are otherwise exactly the same. We can filter that information to make the dataset much smaller by deleting all the repeating rows and keeping only one `trip_id`.
# -

# ### h) Isochrone Map - 15/50

# Note: This question is open-ended, and credits will be allocated based on the quality of both the proposed algorithm and its implementation. You will receive credits for proposing a robust algorithm, even if you do not carry out the implementation.
#
# Moreover, it is not mandatory to utilize the large scale database for addressing this question; plain Python is sufficient. You are free to employ any Python package you deem necessary. However, ensure that you list it as a prerequisite of this notebook so that we remember to install them.

# **Question**:
# * Given a time of day on Monday (or any other day of the week you may choose), and a starting point in Lausanne area.
# * Propose a routing algorithm (such as Bellman-Ford, Dijkstra, A-star, etc.) that leverages the previously created tables to estimate the shortest time required to reach each stop within the Lausanne region using public transport.
# * Visualize the outcomes through a heatmap (e.g., utilizing Plotly), where the color of each stop varies based on the estimated travel time from the specified starting point. See example:
#
# ![example](./figs/isochrone.png).
#
# * Hints:
#     - Focus solely on scenarios where walking between stops is not permitted. Once an algorithm is established, walking can optionally be incorporated, assuming a walking speed of 50 meters per minute. Walking being optional, bonus points (+2) will be awarded for implementing it. 
#     - If walking is not considered, a journey consists of a sequence of stop_ids, each separated by a corresponding trip_id, in chronological order. For example: stop-1, trip-1, stop-2, trip-2, ..., stop-n.
#     - Connections between consecutive stops and trips can only occur at predetermined times. Each trip-id, stop-id pair must be unique and occur at a specific time on any given day according to the timetable. If you want to catch an earlier connection, you must have taken an earlier trip; you cannot go back in time once you've arrived at a stop.
#     - Consider both a _label setting_ and a _label correcting_ method when making your design decision.

# # Calculating optimal travels using Dijkstra algorithm

# ### What is Djikstra?

# Dijkstra algorithm computes the shortest path from a given origin (A) to ***all*** other acessible points (from A) within a defined network. It is perfect for constructing an isochrone map.
# The method checks and stores distances to all nodes in vicinty () of the current node. Then moves to the closest node that has not been visited yet. Provided the network is *well-connected* meaning: no isolated stops and no connections leading far far away from the city (we restrict to lausanne & lausannoise ouest so this is not an issue). The method explores and notes distance to all nodes and does so with compexity $\mathcal{O}((V + E) \log V)$, where $V$ is number of network nodes (stops) and $E$ is number of edges (connections). This is the worst case scenario, with luck and good origin the method completes much faster.  
#
# Below is an ilusatration explaining how the algorithm stores and updates the shortest paths (source: https://www.quantamagazine.org/computer-scientists-establish-the-best-way-to-traverse-a-graph-20241025/)
# ![example](./figs/Djikstra.png).

# +
# Neccecary Packages

# # !pip install networkx
# # !pip install heapq
# # !pip install folium
# # !pip install branca
# # !pip install pyvis

# +
# Neccecary Libraries

import networkx as nx
from heapq import heappop, heappush
import folium
import numpy as np
from branca.colormap import LinearColormap
from collections import defaultdict
import heapq
import ipywidgets as widgets
from IPython.display import display, clear_output
import datetime
import plotly.graph_objects as go
# -

# ### Test previous functions (debugging, skip to Defining Functions) 

# +
# Load stops from database
stops_geo = pd.read_sql(f"""
    SELECT * 
    FROM {namespace}.sbb_stops_lausanne_region
""", conn)

pd.read_sql(f"""
    SELECT stop_id, stop_name
    FROM {namespace}.sbb_stops_lausanne_region
    WHERE stop_name = 'St-Sulpice VD, Parc Scient.'
""", conn)

# +
# Calculate the bounds of your data
max_lat = stops_geo['stop_lat'].max()
min_lat = stops_geo['stop_lat'].min()
max_lon = stops_geo['stop_lon'].max()
min_lon = stops_geo['stop_lon'].min()

# Calculate center point
center_lat = (max_lat + min_lat) / 2
center_lon = (max_lon + min_lon) / 2

# Create custom hover text with all desired information
hover_text = (
    "<b>Stop Name:</b> " + stops_geo['stop_name'] + "<br>" +
    "<b>Stop ID:</b> " + stops_geo['stop_id'].astype(str) + "<br>" +
    "<b>Latitude:</b> " + stops_geo['stop_lat'].round(6).astype(str) + "<br>" +
    "<b>Longitude:</b> " + stops_geo['stop_lon'].round(6).astype(str) + "<br>" + 
    "<b>Travel Time:</b> " + " TBA " + "<br>" + 
    "<b>Time of The Day:</b> " + " TBA " 
)

# Base figure with stop points
fig = go.Figure(go.Scattermapbox(
    lat=stops_geo['stop_lat'],
    lon=stops_geo['stop_lon'],
    mode='markers',
    marker=dict(size=6, color='red'),
    text=hover_text,  # Use the custom hover text
    hoverinfo='text',  # Show only the custom text
    showlegend=False
))

# Final layout adjustments
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
# -


pd.read_sql(f"""
    WITH ranked_data AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY 
                   stop_id, stop_sequence, arrival_time, departure_time,
                   monday, tuesday, wednesday, thursday, friday, saturday, sunday
               ORDER BY trip_id
           ) AS row_num
    FROM {namespace}.sbb_stop_times_lausanne_region
    WHERE monday = TRUE and departure_time > '06:00:00' AND arrival_time < '9:00:00'
    AND trip_id = '1212.TA.92-1-V-j24-1.30.R'	
    )
    SELECT 
        trip_id,stop_id, stop_sequence, arrival_time, departure_time,
        monday, tuesday, wednesday, thursday, friday, saturday, sunday
    FROM ranked_data
    WHERE row_num = 1
    ORDER BY trip_id, stop_sequence, departure_time
    LIMIT 40
    """, conn)

stops_df.head()


# ### Defining Functions:

def get_network_df(weekday, min_departure_time, max_arrival_time):
    '''
    Inputs: day of the week & trip time-window
    Query targets trips that match the inputs
    Notes trip sequences
    Removes duplicate stops times 
    netwrok_df will be used by the Network contructor 
    '''
    return pd.read_sql(f"""
    WITH ranked_data AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY 
                   stop_id, stop_sequence, arrival_time, departure_time,
                   monday, tuesday, wednesday, thursday, friday, saturday, sunday
               ORDER BY trip_id
           ) AS row_num
    FROM {namespace}.sbb_stop_times_lausanne_region
    WHERE {weekday} = TRUE and departure_time > '{min_departure_time}' AND arrival_time < '{max_arrival_time}'
    )
    SELECT 
        trip_id,stop_id, stop_sequence, arrival_time, departure_time,
        monday, tuesday, wednesday, thursday, friday, saturday, sunday
    FROM ranked_data
    WHERE row_num = 1
    ORDER BY trip_id, stop_sequence, departure_time
    """, conn)


def yasi():
    '''
    Yet Another Stop Index:
    returns a spesific dataframe listing:
        stop name/id (with real time)
        walking mask (bool)
    dicitonary maping stop name to stop ID
    '''
    q_realtime_ids = f"""
    CREATE OR REPLACE TABLE {namespace}.istdaten_stop_ids_july2024 AS
    SELECT DISTINCT CAST(bpuic AS VARCHAR) AS stop_id
    FROM {sharedNS}.sbb_istdaten
    WHERE operating_day BETWEEN DATE '2024-07-01' AND DATE '2024-07-31'
    """
    pd.read_sql(q_realtime_ids, conn)

    # Join with Lausanne region stops to get those with realtime
    q_stops_with_realtime = f"""
    SELECT s.stop_name, s.stop_id
    FROM {namespace}.sbb_stops_lausanne_region s
    INNER JOIN {namespace}.istdaten_stop_ids_july2024 r
    ON s.stop_id = r.stop_id
    """
    df = pd.read_sql(q_stops_with_realtime, conn)

    # Build dictionary
    stop_dict = dict(zip(df['stop_name'], df['stop_id']))


    #------------------------------------------------------------------------------------

    # Create the stop-to-stop distance table with 500m mask
    q_stops_distance = f"""
    CREATE OR REPLACE TABLE {namespace}.sbb_stops_to_stops_lausanne_region AS
    SELECT 
        a.stop_id AS stop_id_a,
        a.stop_name AS stop_name_a,
        b.stop_id AS stop_id_b,
        b.stop_name AS stop_name_b,
        ST_Distance(
            to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
            to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
        ) AS distance_meters
    FROM {sharedNS}.sbb_stops_lausanne_region a
    JOIN {sharedNS}.sbb_stops_lausanne_region b
        ON a.stop_id != b.stop_id
    WHERE ST_Distance(
            to_spherical_geography(ST_Point(a.stop_lon, a.stop_lat)),
            to_spherical_geography(ST_Point(b.stop_lon, b.stop_lat))
        ) < 500
    """
    pd.read_sql(q_stops_distance, conn)

    # Read the connected stops
    q_connected = f"""
    SELECT DISTINCT stop_id_a AS stop_id FROM {namespace}.sbb_stops_to_stops_lausanne_region
    UNION
    SELECT DISTINCT stop_id_b AS stop_id FROM {namespace}.sbb_stops_to_stops_lausanne_region
    """
    df_connected = pd.read_sql(q_connected, conn)
    connected_ids = set(df_connected['stop_id'])

    # Add 'connected' column to realtime stops
    df['connected'] = df['stop_id'].isin(connected_ids)

    return df, stop_dict


# +
def time_to_minutes(time_str):
    '''
    Convert HH:MM:SS string to minutes since midnight
    '''
    h, m, s = map(int, time_str.split(':'))
    return int(h * 60 + m + s / 60)

def format_hover_text(row):
    if pd.isna(row['travel_time']):
        travel_time_text = "<b>Travel Time:</b> unreachable<br>"
    else:
        travel_time_text = f"<b>Travel Time:</b> {round(row['travel_time'], 1)} min<br>"
    
    return (
        f"<b>Stop Name:</b> {row['stop_name']}<br>"
        f"<b>Stop ID:</b> {row['stop_id']}<br>"
        f"{travel_time_text}"
        f"<b>Coordinates:</b> {round(row['stop_lat'], 6)}, {round(row['stop_lon'], 6)}"
    )

def travel_time_heatmap(day, min_departure, max_arrival, origin_stop_id):
    '''
    Main Trip Processor
    Takes weekday & time-window & the starting point (ID)
    1. Gets relevant data from get_network_df()
        Generates a graph with non-duplicate stops
        Edges between stops represent the travel time
    2. Time Estimation
        (MAIN ISSUE) the weighting must include
            - Vehicle Travel (determined from df)
            - Estimated Disembarking time (30s)
         If junction: (trip_id changed meaning person got off the buss)
            consider stops within walking range (500m)
            - walking time (distance/50[m/s])
            - waiting time (waiting @ stop)
            
        Junction check using get_junction()   
        Note: junnctions check happen and vehicle travels/disembarkment are decoupled
    3. Combined weights for each edge in the netwrok fed to the Dijksta method (see seperate info)
        Dijkstra generastes shortest trave_times from origin stop to any other accessible stop
    4. Visualization:
        
    '''
    
    # Constants
    STOP_TIME = 0.5  # 30 seconds in minutes
    MIN_TRAVEL_TIME = 0.1  # minimum time between stops (minutes)
    DISEMBARK_TIME = 0.5 # time for disembarking 

    print("Gathering travel data...")
    #-------------------------------------------------------------------------------------------------------------------------------
    
    df = get_network_df(day, min_departure, max_arrival)
    G = nx.DiGraph()
    
    print("Processing trips...")
    #-------------------------------------------------------------------------------------------------------------------------------

    # Pre-add all unique stops once
    all_stops = df['stop_id'].unique()
    G.add_nodes_from(all_stops)
    
    # Process trips without redundant sorting
    grouped = df.sort_values(['trip_id', 'stop_sequence']).groupby('trip_id')
    
    for trip_id, trip_df in grouped:
        stops = trip_df.to_dict('records')
        
        # Use zip for cleaner consecutive pair iteration
        for current, next_ in zip(stops[:-1], stops[1:]):
            travel_time = max(
                time_to_minutes(next_['arrival_time']) - 
                time_to_minutes(current['departure_time']),
                MIN_TRAVEL_TIME
            )
            total_time = travel_time + DISEMBARK_TIME
            G.add_edge(
                current['stop_id'],
                next_['stop_id'],
                weight=total_time,
                type='vehicle',  
                trip=trip_id     
            )
    
    print("Routing using Dijkstra's algorithm...")
    #--------------------------------------------------------------------------------------------------------------------------------

    # extra safety layer: no negative weights allowed
    try:
        travel_times = nx.single_source_dijkstra_path_length(G, origin_stop_id, weight='weight')
    except ValueError as e:
        if "negative weights" in str(e):
            print("Adjusting negative weights")
            for u, v, d in G.edges(data=True):
                d['weight'] = abs(d['weight'])
            travel_times = nx.single_source_dijkstra_path_length(G, origin_stop_id, weight='weight')
        else:
            raise
    
    # Visualization
    print("Generating Heatmap...")
    #--------------------------------------------------------------------------------------------------------------------------------

    df_travel = pd.DataFrame.from_dict(travel_times, orient='index', columns=['travel_time']).reset_index()
    df_travel.rename(columns={'index': 'stop_id'}, inplace=True)
    stops_with_times = pd.merge(stops_geo, df_travel, on='stop_id', how='left')
    
    hover_text = stops_with_times.apply(format_hover_text, axis=1)
    
    fig = go.Figure(go.Scattermapbox(
        lat=stops_with_times['stop_lat'],
        lon=stops_with_times['stop_lon'],
        mode='markers',
        marker=dict(
            size=6,
            color=stops_with_times['travel_time'],
            colorscale='Plasma',
            opacity=0.8,
            cmin=0,
            cmax=stops_with_times['travel_time'].quantile(0.95),
            colorbar=dict(title='Travel Time (min)')
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.5,
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        # title=f"Travel Time from Stop {origin_stop_id}"
    )
    
    fig.show()


# -

# ### Implementing Connections
#

def travel_time_heatmap_V2(day, min_departure, max_arrival, origin_stop_id):
    '''
    Inputs: 
    
        weekday, search time-window, origin stop id, 

        get_network_df:
            tripID, stopID: no duplicates 
            stop_sequence: ordered increment
            arrival_time/departure_time
            boolean weekday serivice
        walking_pairs:
            stopID_a, stopID_b, distance [m]

    V2 accounts for:
        - Vehicle travel time
        - Diembarking time
        - Welking time if transfer
        - Waiting time if transfer

    1. Build Network:
        cache all walking connections withing max_walk_dist
        build network with stops as nodes and connections as edges with weight (Time)
        
    2. Calculate Time:
        Vehicle edges: driving + fixed disembarking time
        Transfer edges: waiting
        Walking edges: distance/50[m/min]
        
    3. Pathfinding using Dijkstra algorithm
        handles negative weights
        returns travel time to each non-isolated node in the network (otherwise display:'unavailable')
        
    4. Visulization
        
    '''
    # Constants
    DISEMBARK_TIME = 1  # 30 seconds
    MIN_TRAVEL_TIME = 0  # 6 seconds
    WALKING_SPEED = 50     # meters/minute
    MAX_WALK_DIST = 100    # meters
    TRANSFER_WAIT = 1      # minute minimum

    print("Gathering travel data...")
    #-------------------------------------------------------------------------------------------------------------------------------
    df = get_network_df(day, min_departure, max_arrival)
    G = nx.DiGraph()
    
    # Pre-cache walking connections
    walk_pairs = walking_pairs[walking_pairs['distance_meters'] <= MAX_WALK_DIST]
    walk_graph = defaultdict(dict)
    for _, row in walk_pairs.iterrows():
        walk_time = row['distance_meters'] / WALKING_SPEED
        walk_graph[row['stop_id_a']][row['stop_id_b']] = walk_time
        walk_graph[row['stop_id_b']][row['stop_id_a']] = walk_time

    # Add all stops first
    G.add_nodes_from(df['stop_id'].unique())

    print("Processing trips...")
    #-------------------------------------------------------------------------------------------------------------------------------
    # Process trips and connections in one pass
    last_trip_info = {}
    for trip_id, trip_df in df.groupby('trip_id'):
        stops = trip_df.sort_values('stop_sequence').to_dict('records')
        
        # Add vehicle edges
        for i in range(len(stops)-1):
            curr, next_ = stops[i], stops[i+1]
            travel_time = max(
                time_to_minutes(next_['arrival_time']) - 
                time_to_minutes(curr['departure_time']),
                MIN_TRAVEL_TIME
            )
            G.add_edge(
                curr['stop_id'], next_['stop_id'],
                weight=travel_time + DISEMBARK_TIME,
                type='vehicle',
                trip=trip_id
            )
        
        # Process transfers at each stop
        for stop in stops:
            stop_id = stop['stop_id']
            if stop_id in last_trip_info:
                last_trip, last_arrival = last_trip_info[stop_id]
                if last_trip != trip_id:
                    wait_time = max(
                        time_to_minutes(stop['departure_time']) - 
                        time_to_minutes(last_arrival),
                        TRANSFER_WAIT
                    )
                    # Add transfer edge
                    G.add_edge(
                        last_trip, trip_id,
                        weight=wait_time,
                        type='transfer',
                        via_stop=stop_id
                    )
                    
                    # Add walking connections
                    for neighbor, walk_time in walk_graph.get(stop_id, {}).items():
                        G.add_edge(
                            stop_id, neighbor,
                            weight=walk_time,
                            type='walk'
                        )
            
            last_trip_info[stop_id] = (trip_id, stop['arrival_time'])

    print("Routing using Dijkstra's algorithm...")
    #--------------------------------------------------------------------------------------------------------------------------------
    try:
        travel_times = nx.single_source_dijkstra_path_length(
            G, origin_stop_id, weight='weight', 
            cutoff=120  # Max 2 hours for performance
        )
    except ValueError as e:
        if "negative weights" in str(e):
            for u, v, d in G.edges(data=True):
                d['weight'] = abs(d['weight'])
            travel_times = nx.single_source_dijkstra_path_length(
                G, origin_stop_id, weight='weight'
            )

   
    # Visualization
    print("Generating Heatmap...")
    #--------------------------------------------------------------------------------------------------------------------------------

    df_travel = pd.DataFrame.from_dict(travel_times, orient='index', columns=['travel_time']).reset_index()
    df_travel.rename(columns={'index': 'stop_id'}, inplace=True)
    stops_with_times = pd.merge(stops_geo, df_travel, on='stop_id', how='left')
    
    hover_text = stops_with_times.apply(format_hover_text, axis=1)
    
    fig = go.Figure(go.Scattermapbox(
        lat=stops_with_times['stop_lat'],
        lon=stops_with_times['stop_lon'],
        mode='markers',
        marker=dict(
            size=6,
            color=stops_with_times['travel_time'],
            colorscale='Plasma',
            opacity=0.8,
            cmin=0,
            cmax=stops_with_times['travel_time'].quantile(0.95),
            colorbar=dict(title='Travel Time (min)')
        ),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.5,
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        # title=f"Travel Time from Stop {origin_stop_id}"
    )
    
    fig.show()


def get_heatmap(stops_df, stop_dict):
    # Widget controls
    weekday_dropdown = widgets.Dropdown(
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        description='Weekday:',
        value='Monday',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='30%')
    )

    stop_input = widgets.Combobox(
        value='St-Sulpice VD, Parc Scient.',
        placeholder='Type stop name...',
        options=sorted(stop_dict.keys()),
        description='Start at:',
        ensure_option=True,
        style={'description_width': '100px'},
        layout=widgets.Layout(width='30%')
    )

    departure_time = widgets.Text(
        value='18:03:00',
        description='Departure:',
        placeholder='HH:MM:SS',
        layout=widgets.Layout(width='200px')
    )

    arrival_time = widgets.Text(
        value='20:00:00',
        description='Arrival:',
        placeholder='HH:MM:SS',
        layout=widgets.Layout(width='200px')
    )

    run_button = widgets.Button(
        description='Generate Map',
        button_style='success',
        icon='map'
    )

    
    output = widgets.Output()

    # Button Event
    def on_run_button_click(b):
        with output:
            clear_output()
            name = stop_input.value

            assert name in stop_dict, (
            f"Stop name {stop_name} is either misspelled or missing real-time")

            stop_id = stop_dict[name]
            weekday = weekday_dropdown.value
            min_dep = departure_time.value
            max_arr = arrival_time.value

            #Calculate:
            print(f"Generating travel time heatmap for {weekday} from: {name} (ID: {stop_id})")
            travel_time_heatmap_V2(weekday, min_dep, max_arr, stop_id)
            
    run_button.on_click(on_run_button_click)
    
    # Display the widgets
    display(
        widgets.VBox([
            weekday_dropdown,
            stop_input,
            widgets.HBox([departure_time, arrival_time]),
            run_button,
            output
        ])
    )

walking_pairs.head()

stops_df, stop_dict = yasi()

get_heatmap(stops_df , stop_dict)

# ![example](./figs/isochrone.png).




