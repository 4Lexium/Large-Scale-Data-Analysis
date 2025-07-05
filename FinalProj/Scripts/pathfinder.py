# +
# PySpark core functions and types
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lit, udf, expr, when, explode,
    from_unixtime, to_timestamp, unix_timestamp,
    year, month, dayofmonth, dayofweek, hour, minute,
    radians, sin, cos, atan2, sqrt, pow, exp, abs as abs_,
    substring, lower, trim, regexp_replace,
    countDistinct, coalesce, lead, lag,
    sum as spark_sum, max as spark_max, avg, variance, stddev,
    least, greatest, first
)
from pyspark.sql.types import StringType, DoubleType, ArrayType
from pyspark.sql.window import Window

# Datetime and timezone utilities
from datetime import datetime, timedelta
import pytz

# Geospatial and distance calculation
import geopandas as gpd
from shapely.geometry import Point
import shapely.wkb
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2

# Data science utilities
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Graph utilities
import networkx as nx
from collections import defaultdict
from heapq import heappush, heappop

# Widgets and display
import ipywidgets as widgets
import folium
from IPython.display import display, clear_output
from datetime import datetime, timedelta
import math
import pandas as pd


# +
import base64 as b64
import json
import time
import re
import os
import warnings
import pwd

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="pandas only supports SQLAlchemy connectable .*")



# -

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'U1'


def test():
    return "Imported successfully!"


# +
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


@udf(DoubleType())
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat2), math.radians(lat1)
    delta_phi = math.radians(lat1 - lat2)
    delta_lambda = math.radians(lon1 - lon2)
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
def get_all_close_stop_pairs(spark, radius_meters=500, saveas='stop_to_stop'):
    stops_df = spark.read \
        .option("header", True) \
        .parquet(f"{hadoopFS}/user/com-490/group/U1/stops")

    filtered_df = (
        stops_df
        .select("stop_id", "stop_name", "stop_lat", "stop_lon")
        .distinct()
    )

    # Self-join on stop_id
    joined = filtered_df.alias("a").crossJoin(filtered_df.alias("b")) \
        .filter(col("a.stop_id") < col("b.stop_id"))

    base_pairs = joined.withColumn(
        "distance",
        haversine(
            col("a.stop_lat"), col("a.stop_lon"),
            col("b.stop_lat"), col("b.stop_lon")
        )
    ).filter(col("distance") <= radius_meters)

    # if distance=0 add 40m
    selected_pairs = base_pairs.select(
        col("a.stop_id").alias("from_stop"),
        col("a.stop_name").alias("from_stop_name"),
        col("b.stop_id").alias("to_stop"),
        col("b.stop_name").alias("to_stop_name"),
        when(col("distance") <= 40, 40).otherwise(col("distance")).alias("distance")
    )

    # Reverse directions
    reversed_pairs = selected_pairs.select(
        col("to_stop").alias("from_stop"),
        col("to_stop_name").alias("from_stop_name"),
        col("from_stop").alias("to_stop"),
        col("from_stop_name").alias("to_stop_name"),
        col("distance")
    )

    # Make it bidirectional
    final_df = selected_pairs.union(reversed_pairs)

    return final_df, filtered_df

def get_data(spark, master, model, target_arrival_str, dow, time_window_sec=3600):
    """
    First Step Towards constructing a Newtowrk and paths!

    Inputs: master, model sparkDf, target/origin stop ID str, Day of week: dow int
    
    Args:
        master_df: Spark DataFrame containing scheduled transits: stopA / stopB / nominal trival time (scedule) / dow / tripID / expected arrival @B
        model_df: Spark DataFrame containing our model predicitions for expected delay and std dev @ given stopID/hour/dow
        target_arrival_str: Target arrival time HH:MM:SS
        time_window_sec: Time window in seconds (default 60min)

    What happens:
        Global filter filters master based on day of week that we get from target_arrival_str, and the time-window we consider: [target_time-3600sec, target_time]
        Join filtered master with the model on dow, hour and stopA
        You have now a edge/connection in vacuum between A and B with nominalT, t_arrival @B, and Delay with risk @A
        Finally Using stopID, lat, lon, CONSTANT paramters for walking pace, and max walking distance you create walking edges with the same strucutre as transit edges (but risk and tripID are None)
    Returns: 
        transit_edges, walking_edges, exisiting_stops (stops acessible by foot or vehicle, will be used to make Nodes) 
    """
    
    # Handle dates and time strings
    target_arrival = datetime.strptime(target_arrival_str, "%H:%M:%S")
    window_start = target_arrival - timedelta(seconds=time_window_sec)
    target_hour = target_arrival.hour
    hour_options = [target_hour, target_hour - int(time_window_sec//3600)]
    # Format back to string for Spark filter — NO milliseconds
    target_arrival_str = target_arrival.strftime("%H:%M:%S")
    window_start_str = window_start.strftime("%H:%M:%S")
    
    # Benchmark to see how time info is interpreted 
    # print(f"time: {target_arrival_str}, {window_start_str}, {dow}, {target_hour}")
    
    # Filter master data for the target day and arrival time window
    global_filter = master.filter(
        (col("dow") == int(dow)) &
        (col("t_arrival").between(window_start_str, target_arrival_str))
    )
    # Join master with model and create transit edges
    model_alias = model.alias("m")
    edges_alias = global_filter.alias("e")

    final_edges = model_alias.join(
        edges_alias,
        (col("e.from_stop") == col("m.stop_id")) &
        (col("e.dow") == col("m.dow")) &
        (col("m.hour").isin(hour_options)),
        how="inner"
    )
    # instead of (hour(col("e.t_arrival"))==col("m.hour")),  
    transit_edges = final_edges.select(
        col("e.trip_id"),
        col("e.from_stop"),
        col("e.to_stop"),
        col("e.T_nominal"),
        col("e.t_arrival"),
        col("m.mean_delay").alias("R_risk"),
        col("m.std_dev_delay").alias("std_dev"),
        lit("transit").alias("edge_type")
    )
    
    # Create walking edges
    
    # Constant Parameters
    WALK_SPEED = 1.4  # m/s
    #MAX_WALK_TIME = 900  # seconds
    #MAX_WALK_DISTANCE = WALK_SPEED * MAX_WALK_TIME  # meters  
    MAX_WALK_DISTANCE = 500
    walk_df, existing_stops = get_all_close_stop_pairs(spark, radius_meters=MAX_WALK_DISTANCE)

    walking_edges = (
        walk_df.withColumn("trip_id", lit(None).cast("string"))
                .withColumn(
                    "T_nominal",
                    (col("distance") / lit(WALK_SPEED)).cast("double")
                )
               .withColumn("t_arrival", lit(None).cast("string")) 
               .withColumn("R_risk", lit(0).cast("int"))
               .withColumn("std_dev", lit(0).cast("int"))
               .withColumn("edge_type", lit("walk"))
    ).drop('from_stop_name', 'to_stop_name', 'distance')

    # Enforce correct column structure
    edge_columns = ["trip_id", "from_stop", "to_stop", "T_nominal", "t_arrival", "R_risk", "std_dev", "edge_type"]
    transit_edges = transit_edges.select(*edge_columns)
    walking_edges = walking_edges.select(*edge_columns)
       
    return existing_stops, transit_edges, walking_edges

def build_graph_from_edges(transit_edges: DataFrame, walking_edges: DataFrame, risk:bool=False) -> nx.DiGraph:
    """
    Inputs: transit/walking edges, Include risk (boolean)
    
    Step 2: Use the transit and walking edges and the stopIDs to form a DiGraph with nodes and edges.
    For logic behing static vs dynamic Dijkstra check the README or the comment markdown cells

    Merge walking edges with transit edges. Calculate Static weight based on R_risk and T_nominal (Risk weight contribution is optional with risk=False)
    Between each pair of nodes we take only the edge with minimal weight to create a (DiGraph) with minimal required data stored
    
    Return: DiGraph
    """
    
    # Union and convert to pandas
    weight_expr = "T_nominal + R_risk" if risk else "T_nominal"
    edge_df = (
        transit_edges.selectExpr("from_stop", "to_stop", f"{weight_expr} as weight")
        .unionByName(
            walking_edges.selectExpr("from_stop", "to_stop", f"{weight_expr} as weight")
        )
        .dropna(subset=["from_stop", "to_stop", "weight"])
        .toPandas()
    )
    edge_df = edge_df.groupby(["from_stop", "to_stop"], as_index=False)["weight"].min()
    # Build graph
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
            G.add_edge(row["from_stop"], row["to_stop"], weight=row["weight"])
    
    return G

def dijkstra_path(G: nx.DiGraph, origin: str, target: str):
    """
    Input: DiGraph with stopsID as Nodes and shortest AB trips as edges per node pair in nodes
            origin stopID and target stop ID
    Perform networxx library Dijkstra's algorithm to find the shortest path based on static T_nominal and R_risk
    Return: path as list of stop_ids.
    """
    if origin not in G or target not in G:
        print(f"Origin or target not in graph: {origin}, {target}")
        return []
        
    try:
        path = nx.dijkstra_path(G, source=origin, target=target, weight="weight")
        return path
    except nx.NetworkXNoPath:
        print(f"No path from {origin} to {target}")
        return []


def multi_path(G: nx.DiGraph, origin: str, target: str, k: int = 2):
    """
    Input: DiGraph, origin ID and target ID
    Extension of Dijkstra algorithm called Yen's algorithm  (newtorxx.shortest_simple_paths)
    Return multipole shortest paths by 'weight' from origin to target.
    Default: Function must be optimized to k larger than 2. 
    Return: dijkstra_path and the spur_path (second best)
    """
    if origin not in G or target not in G:
        print(f"Origin or target not in graph: {origin}, {target}")
        return []

    try:
        paths_gen = nx.shortest_simple_paths(G, origin, target, weight="weight")
        paths = list(next(paths_gen) for _ in range(2))  # top 2 paths
        return paths[0], paths[1] if len(paths) > 1 else None
    except nx.NetworkXNoPath:
        print(f"No path from {origin} to {target}")
        return [], []


def plot_stop_network_with_path(graph: nx.DiGraph, existing_stops_df, path=None, alt_path=None):
    """
    Input: dijksrta path and spur path if exist, existing stops (ID, lat, lon)
    Plot the network graph with node positions based on lat/lon.
    Highligts Best in red and optinally next best in orange 
    """
    
    # Convert Spark DataFrame to Pandas
    stop_coords = existing_stops_df.select("stop_id", "stop_lat", "stop_lon").toPandas()

    # Ensure stop_id is string and lat/lon are floats
    stop_coords["stop_id"] = stop_coords["stop_id"].astype(str)
    stop_coords["stop_lat"] = stop_coords["stop_lat"].astype(float)
    stop_coords["stop_lon"] = stop_coords["stop_lon"].astype(float)

    # Create a position dictionary
    pos = {
        row["stop_id"]: (row["stop_lon"], row["stop_lat"])
        for _, row in stop_coords.iterrows()
        if row["stop_id"] in graph.nodes  
    }

    # Visulaization
    
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='gray')
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', alpha=0.5)

    # Draw spur-path first/bottom
    if alt_path and len(alt_path) > 1:
        alt_edges = [(alt_path[i], alt_path[i + 1]) for i in range(len(alt_path) - 1)]
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=alt_path, node_color="orange", node_size=50)
        nx.draw_networkx_edges(graph, pos=pos, edgelist=alt_edges, edge_color="orange", width=2)

    # Draw primary dijkstra path
    if path and len(path) > 1:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=path, node_color="red", node_size=70)
        nx.draw_networkx_edges(graph, pos=pos, edgelist=path_edges, edge_color="red", width=3)

    plt.title("Shortest Path Through Stop Network using Static Dijkstra (W=T+R) and (optionally) Spur-path with Yen's extension \n Primary (red) and Spur-path (orange)")
    plt.show()
    
def prob_delay_less_than(waiting_time, walking_time, mean, std_dev):
    """
    Computes log(P(delay ≤ threshold)) assuming a normal distribution.
    Handles degenerate case where std_dev = 0.
    """
    threshold = waiting_time-walking_time
    if std_dev == 0 or threshold <= 0:
        # Degenerate distribution: P(X ≤ threshold) = 1 if threshold ≥ mean, else 0
        return 0.0 if threshold >= mean else -math.inf  # log(1) = 0, log(0) = -inf
    
    z = (threshold - mean) / std_dev
    return math.log(norm.cdf(z))


def dynamic_backtracing(path, target_arrival_str, transit_edges, walking_edges):
    """
    Final step:
    Input: path result of the static path finding algorithm, target_arrival (HH::MM::SS), transit and walking edges
    For each edge if not None (walking) tripID, from stop, to stop, T_nominal, t_arrival @B, R_risk @A, std dev of R
    Now with a given skeleton of stops we assing the best travel option based on the 3 paramters t, T, R
    Imporant: we backpropagate, up untill this point we forward propagated 
    Define temp variables: CLK, best_walk, best_transit, best_walk_time, best_transit time, danger (cumulative confidence of the R risk delay)
        As you are at some stopB wanting to go to B
        CLK: allows you to thow away un causal transit edges (they arrive at B after you "spawn/arrive" at B, so you cant catch it)
        Updating CLK thought waiting time, T nominal and checking up with t_arrival is something you cant do with static algorithm like Dijkstra standard lib function
        Find best walk among walks and then best transit among transits (important to segregate, before final comparison and danger factor calculation)
    Use prob_delay_less_than function to determine danger factor for each path link, logarithmic output allows a simple += accumulation
    Returns:
        path_rendering: list with best option if found, conatining all needed information: from/to, updated CLK @A (so now you know when you leave at each stop frowards in time), Tnominal, Risk, Danger factor (for each link) with exp()
        danger: cumulative, with exp()
        itsover: How big the time split is between when you wanted to arrive vs when you will arive worst case. How much time you have left
    """
    def parse_time(t):
        if isinstance(t, datetime):
            return t
        if isinstance(t, str):
            return datetime.strptime(t, "%H:%M:%S")
        return None

    def format_time(t_dt):
        return t_dt.strftime("%H:%M:%S")

    # Initialize clock with user-given arrival time
    CLK = parse_time(target_arrival_str)

    # Build fast lookup for edges: {(from_stop, to_stop): [edge_dicts]}
    transit_pd = transit_edges.select("from_stop", "to_stop", "t_arrival", "T_nominal", "R_risk", "std_dev", "trip_id").toPandas()
    walking_pd = walking_edges.select("from_stop", "to_stop", "T_nominal", "trip_id").toPandas()
    
    edge_lookup = {}
    for df, edge_type in [(transit_pd, "transit"), (walking_pd, "walk")]:
        for _, row in df.iterrows():
            key = (row["from_stop"], row["to_stop"])
            edge_data = row.to_dict()
            edge_data["edge_type"] = edge_type
            edge_lookup.setdefault(key, []).append(edge_data)

    # Begin walking backward through the path
    path_rendering = []
    tripID = None
    danger = 0
    z=0
    for i in reversed(range(len(path) - 1)):
        z+=1
        stopA = path[i]
        stopB = path[i + 1]
        options = edge_lookup.get((stopA, stopB), [])
        best_walk = None
        best_transit = None
        best = None
        min_total_time_transit = float("inf")
        min_total_time_walk = float("inf")
        for edge in options:
            edge_type = edge["edge_type"]
            t_nominal = edge.get("T_nominal", 0)
            r_risk = edge.get("R_risk", 0) or 0
            stddev = edge.get("std_dev", 0) or 0
            new_tripID = edge.get("trip_id")

            if edge_type == "transit":
                t_arrival = parse_time(edge.get("t_arrival"))
                if new_tripID == tripID: 
                    wait_time = 0
                elif t_arrival is None or t_arrival > CLK:
                    continue #skip non causal edge
                else:
                    wait_time = (CLK - t_arrival).total_seconds()
                 
                total_time_transit = wait_time + t_nominal
                
                if total_time_transit < 0:
                    continue  # sanity check!

                if total_time_transit < min_total_time_transit:
                    min_total_time_transit = total_time_transit 
                    best_transit = {
                        "from_stop": stopA,
                        "to_stop": stopB,
                        "new_clk": CLK - timedelta(seconds=total_time_transit),
                        "edge_type": edge_type,
                        "trip_id": new_tripID,
                        "t_nominal": t_nominal,
                        "wait_time": wait_time,
                        "r_risk": r_risk,
                        "stddev": stddev
                    }

            elif edge_type == "walk":
                # no causality check and no waiting needed
                wait_time = 0
                total_time_walk = wait_time + t_nominal

                if total_time_walk < min_total_time_walk:
                    min_total_time_walk = total_time_walk
                    best_walk = {
                        "from_stop": stopA,
                        "to_stop": stopB,
                        "new_clk": CLK - timedelta(seconds=total_time_walk),
                        "edge_type": edge_type,
                        "trip_id": new_tripID,
                        "t_nominal": t_nominal,
                        "wait_time": wait_time,
                        "r_risk": r_risk,
                        "stddev": stddev
                    }
            else:
                continue #invalid transit


        # Compare best of walks vs best of transit
        """
        Printing for Benchmarking
        print("+++++++++++++++++++++++++++++++++++++++++")
        print(stopA, stopB)
        if best_transit is not None:
            print(f'transit WAIT {best_transit["wait_time"]}')
            print(f'transit NOMINAL {best_transit["t_nominal"]}')
            print(f'transit RISK {best_transit["r_risk"]}')
            print(f'std dev risk {best_transit["stddev"]}')
        else:
            print("no transit found")
        if best_walk is not None:
            print(f'WALKING {best_walk["t_nominal"]}')
        else: 
            print("no walking edges made")
        """
        if min_total_time_transit != float("inf") and  min_total_time_walk != float("inf"):
            continue  
        if min_total_time_transit <= min_total_time_walk:
            best = best_transit
            if best is None:
                print(f"[ERROR] No link found for {stopA} to {stopB} at CLK={format_time(CLK)}, May The Force Guide You Furher🙏🙏🙏")
                break
            danger += prob_delay_less_than(
                best.get("wait_time", 0) or 0,
                min_total_time_walk or 0,
                best.get("r_risk", 0) or 0,
                best.get("stddev", 0) or 0
            )
        else:
            best = best_walk
        # print(f'DANGER {danger}')
        if best is None:
            print(f"No viable edge from {stopA} to {stopB} at CLK={format_time(CLK)}")
            break

        path_rendering.append((
            best["from_stop"],
            best["to_stop"],
            format_time(best["new_clk"]),
            [int(best["t_nominal"]//60), int(best["t_nominal"]%60)],
            best["edge_type"],
            best["trip_id"],
            best["r_risk"],
            math.exp(danger)*100
        ))
        CLK = best["new_clk"]
        tripID = best["trip_id"]
                   
    return path_rendering, danger

def display_path(path_rendering, stop_name_dict):
    """
    Inpuut: path_rendering from previous step
    Create a nice pandas df to view the journey. 
    Bottom -> Up
    """
    # column names matching the tuple path rendering
    columns = [
        "from_stop", "to_stop", "time", "travel_time", "edge_type", "trip_id", "r_risk", "danger_score"
    ]
    
    # Convert to DataFrame
    path_df = pd.DataFrame(path_rendering, columns=columns)
    
    # Add stop names
    path_df["from_stop_name"] = path_df["from_stop"].map(stop_name_dict).fillna("Unknown")
    path_df["to_stop_name"] = path_df["to_stop"].map(stop_name_dict).fillna("Unknown")
    
    # Add emojis for edge type
    edge_icons = {
        "transit": "🚌",
        "walk": "🚶"
    }
    path_df["edge_icon"] = path_df["edge_type"].map(edge_icons).fillna("❓ Unknown")
    
    # reorder and rename to make nicer
    path_df = path_df[[
        "time",
        "edge_icon",
        "from_stop", "from_stop_name",
        "to_stop", "to_stop_name",
        "trip_id",
        "travel_time",
        "r_risk",
        "danger_score"
    ]]

    path_df["danger_score"] = path_df["danger_score"].round(2)
    # Display the final table
    display(path_df)

    return path_df


def get_network_dict(existing_stops, transit_edges, walking_edges):

    """
    Inputs: existingstops for nodes, transit/walking for edges
    In case of the alternative TIme Dependant Dijkstra modification we must build our own graph using t, T, R
    Take neccecary info from the walking and transit edges 
    Return: the network dictionary
    """
    
    nodes = existing_stops.select("stop_id").rdd.flatMap(lambda x: x).collect()
    walking_edges_py = walking_edges.select(
        "from_stop", "to_stop", "T_nominal"
    ).rdd.map(lambda row: {
        "from_stop": row["from_stop"],
        "to_stop": row["to_stop"],
        "T_nominal": row["T_nominal"],
        "t_arrival": None,
        "R_risk": 1,
        "edge_type": "walk",
        "trip_id": None
    }).collect()
    
    transit_edges_py = transit_edges.select(
        "from_stop", "to_stop", "T_nominal", "t_arrival", "R_risk", "trip_id"
    ).rdd.map(lambda row: {
        "from_stop": row["from_stop"],
        "to_stop": row["to_stop"],
        "T_nominal": row["T_nominal"],
        "t_arrival": row["t_arrival"],
        "R_risk": row["R_risk"],
        "edge_type": "transit",
        "trip_id": row["trip_id"]
    }).collect()
    graph = defaultdict(list)

    for edge in walking_edges_py + transit_edges_py:
        graph[edge["to_stop"]].append({
            "from_stop": edge["from_stop"],
            "T_nominal": edge["T_nominal"],
            "t_arrival": edge["t_arrival"],
            "R_risk": edge["R_risk"],
            "edge_type": edge["edge_type"],
            "trip_id": edge["trip_id"]
        })
    
    return graph

def dynamic_dijkstra(graph, origin, destination, target_time):
    """
    Work in progress
    Idea is to use the base idea of Dijkstra and move in short steps and map our surroundings
    Except now, we need to evaluate causality and risk of trips directkly inside the walking. So its no longer static weights.
    See comments for "simplified" Risk logic
    Lacks some of the "depper" logic used in the standard lib Dijkstra like cumulative weight and #connection evaluation
    Most importantly once a you move shortest step, this becomes the path, and the path is set in stone!
    returns rendered map (shortest time to all noticed stops) and rendered path (with all stops visited and connection information, how we got there)
    """
    
    def parse_time(t_str):
        return datetime.strptime(t_str, "%H:%M:%S") if t_str else None

    def format_time(t_dt):
        return t_dt.strftime("%H:%M:%S")

    CLK = parse_time(target_time)
    print(CLK)
    map_rendering = {}
    heap = []
    visited = {destination}
    path_rendering = []
    # Push initial stop with CLK
    heappush(heap, (destination, CLK, None, None))  # current_stop (B), clk_time, from_stop(into the past! A), tripID

    while heap:
        stopB, clk, from_stop, tripID = heappop(heap)

        best_option = None
        best_option_walk = None
        min_total_time = float("inf")
        min_total_time_walk = float("inf")
        j=0
        z=0
        for edge in graph.get(stopB, []):
            j+=1
            stopA = edge["from_stop"]
            edge_type = edge["edge_type"]
            if edge_type == 'transit':
                t_arrival = parse_time(edge["t_arrival"]) if edge["t_arrival"] else None
            elif edge_type == 'walk':
                t_arrival = clk
            t_nominal = edge["T_nominal"]
            r_risk = edge["R_risk"] or 0
            new_tripID = edge["trip_id"] 

            # check causality and risk
            if edge_type == "transit" and tripID is not None:
                if new_tripID is not None and new_tripID == tripID:
                    continue
                if new_tripID != tripID and (r_risk > 300 or t_arrival > clk):   # if its actually best option to continue on the current buss route, you should gain imunity to delays for the turn, and also the delay gets dragged over
                    continue

            # only for transit, and if tripIDs are not None (excludes first iteration, and isntacne without new tripID), get skipped over
            # if theres a change in tripID, meaning a disembarking we need to acess risk (> 5min delay = missed connection remember R @ A )
            # and causality t arrival at B can NOT be later then current CLK
            # now, if tripID is the same, we dont get off the buss, none of this matters 
            # if you make it to here you are either: walked, stayed on the same buss, or changed but got in time to catch AB @B 

            # Find time it will take you to travel back to A
            if edge_type == "transit":
                if new_tripID == tripID:   # you dont get off, now waiting just nominal travel 
                    wait_time = 0
                else:
                    wait_time = (clk - t_arrival).total_seconds()
                total_time = wait_time + t_nominal

            elif edge_type == "walk":
                wait_time = 0
                total_time  = wait_time + t_nominal
            else:
                continue  # Unknown edge type

            # Render Map (BINOCULAR MAN SCAN)
            A_time = clk - timedelta(seconds=total_time)
            if stopA not in map_rendering:
                map_rendering[stopA] = A_time
            elif A_time > map_rendering[stopA]:
                map_rendering[stopA] = A_time

            # Find Shortest path backwards (but you cant go to an already visited node)
            if stopA in visited:
                continue  
            
            # Find best path
            if total_time < min_total_time and edge_type == "transit":
                min_total_time = total_time
                best_option = (stopA, stopB, total_time, t_nominal, wait_time, clk-timedelta(seconds=total_time), r_risk, edge_type, new_tripID)
            if total_time < min_total_time_walk and edge_type == "walk":
                min_total_time_walk = total_time
                best_option_walk = (stopA, stopB, total_time, t_nominal, wait_time, clk-timedelta(seconds=total_time), r_risk, edge_type, new_tripID)    

        # Move to the best option
        if best_option or best_option_walk:
            if not best_option:
                print("no transit possible so we walk")
                print(best_option_walk)
                stopA, stopB, total_time, t_nominal, wait_time, new_clk, r_risk, edge_type, new_tripID = best_option_walk
            elif not best_option_walk:
                print("nowhere to walk, i guess we wait for the next bus")
                print(best_option)
                stopA, stopB, total_time, t_nominal, wait_time, new_clk, r_risk, edge_type, new_tripID = best_option
            elif best_option[4] > 600 and best_option[4] > best_option_walk[4]:
                print("dont wanna wait more than 10min, resort to walking")
                print(best_option_walk)
                stopA, stopB, total_time, t_nominal, wait_time, new_clk, r_risk, edge_type, new_tripID = best_option_walk
            else:
                print("vehilce was genuinely better")
                print(best_option)
                stopA, stopB, total_time, t_nominal, wait_time, new_clk, r_risk, edge_type, new_tripID = best_option
 
            heappush(heap, (stopA, new_clk, None, new_tripID))    #the best A becomes the new B, and now look for C which is again unknown
            
            path_rendering.append ((
                stopB,  # we went backwards from B to A
                stopA,
                new_clk, # clk at A
                edge_type,
                new_tripID,
                round(r_risk / t_nominal, 3) if t_nominal > 0 else 0  # risk to nominal ratio
            ))
            
            visited.add(stopA)
            # Now we check if Origin appeared!
            if origin in visited:
                print("origin reached")
                break
            if origin in map_rendering:
                print("origin discovered")
        else:
            print("best option is false, fatal round")
               
    return path_rendering, map_rendering, visited

def create_stop_selector_widget(spark, master, model, stop_name_dict, time_window_sec=3600):
    options = list(stop_name_dict.values())
    name_to_id = {v: k for k, v in stop_name_dict.items()}  # invert dictionary

    origin = widgets.Combobox(
        placeholder='Type origin stop',
        value = "Lausanne, Bellerive",
        options=options,
        description='Origin:',
        layout=widgets.Layout(width='300px'),
        ensure_option=True,
        continuous_update=False
    )

    target = widgets.Combobox(
        placeholder='Type target stop',
        value = "Ecublens VD, EPFL (bus)",
        options=options,
        description='Target:',
        layout=widgets.Layout(width='300px'),
        ensure_option=True,
        continuous_update=False
    )

    target_arrival = widgets.Text(
        value='15:00:00',
        placeholder='HH:MM:SS',
        description='Arrival Time:',
        layout=widgets.Layout(width='300px')
    )

    day_options = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }

    day_selector = widgets.Dropdown(
        options=day_options,
        value=3,
        description='Day of Week:',
        layout=widgets.Layout(width='300px')
    )

    submit_button = widgets.Button(description="Submit", button_style='success')
    output = widgets.Output()

    def on_submit_clicked(b):
        with output:
            clear_output()
            origin_id = name_to_id.get(origin.value)
            target_id = name_to_id.get(target.value)
            dow = day_selector.value
            print("Submitted values:")
            print("+++++++++++++++++++++++++++++++++++++++++")
            print(f"Origin: {origin.value} ({origin_id})")
            print(f"Target: {target.value} ({target_id})")
            print(f"Arrival Time: {target_arrival.value}")
            print(f"Day of Week (DOW): {dow}")
            print(" ")
            print("Default Hidden Values")
            print("+++++++++++++++++++++++++++++++++++++++++")
            print("Time Window: 60min")
            print("Walking Pace 1.4m/s")
            print("Max Walking Range: 500m")

            submit_button.origin = origin_id
            submit_button.target = target_id
            submit_button.arrival = target_arrival.value
            submit_button.dow = dow
            
            existing_stops, transit_edges, walking_edges = get_data(spark, master, model, submit_button.arrival, submit_button.dow)
            graph = build_graph_from_edges(transit_edges, walking_edges, risk=True)
            path_prim, path_spur = multi_path(graph, submit_button.origin, submit_button.target)
            print("Primary Path (Dijksra Alg.)")
            print(path_prim)
            print(" ")
            print("Auxillary Spur Path (Yen's Alg.)")
            print(path_spur)
            print(" ")
            print("Generating Plot. This might take a while...")
            plot_stop_network_with_path(graph, existing_stops, path_prim, path_spur)
            print("Back-Tracing Primary Path")
            path_rendering, danger = dynamic_backtracing(path_prim, submit_button.arrival, transit_edges, walking_edges)
            path_rendering_df = display_path(path_rendering, stop_name_dict)
            print(f"Cumulative Confidence Factor: {math.exp(danger)}")
            if path_spur is not None:
                print(" ")
                print("Back-Tracing Auxillary/Spur Path")
                path_rendering, danger = dynamic_backtracing(path_spur, submit_button.arrival, transit_edges, walking_edges)
                path_rendering_df = display_path(path_rendering, stop_name_dict)
                print(f"Cumulative Confidence Factor: {math.exp(danger)}")
    submit_button.on_click(on_submit_clicked)

    display(origin, target, target_arrival, day_selector, submit_button, output)

    return submit_button

