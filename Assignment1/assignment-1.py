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
# # DSLab Assignment 1 - Data Science with CO2
#
# ## Hand-in Instructions
#
# - __Due: **.03.2025 23h59 CET__
# - Create a fork of this repository under your group name, if you do not yet have a group, you can fork it under your username.
# - `./setup.sh` before you can start working on this notebook.
# - `git push` your final verion to the master branch of your group's repository before the due date.
# - Set the group name variable below, e.g. GROUP_NAME='Z9'
# - Add necessary comments and discussion to make your codes readable.
# - Let us know if you need us to install additional python packages.

# %%
GROUP_NAME='U1'

# %%

# %% [markdown]
# ## Carbosense
#
# The project Carbosense establishes a uniquely dense CO2 sensor network across Switzerland to provide near-real time information on man-made emissions and CO2 uptake by the biosphere. The main goal of the project is to improve the understanding of the small-scale CO2 fluxes in Switzerland and concurrently to contribute to a better top-down quantification of the Swiss CO2 emissions. The Carbosense network has a spatial focus on the City of Zurich where more than 50 sensors are deployed. Network operations started in July 2017.
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#
# <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_ZLMT_3.JPG" width="156">  <img src="http://carbosense.wdfiles.com/local--files/main:sensors/LP8_sensor_SMALL.jpg" width="300">

# %% [markdown]
# ## Description of the assignment
#
# In this assignment, we will curate a set of **CO2 measurements**, measured from cheap but inaccurate sensors, that have been deployed in the city of Zurich from the Carbosense project. The goal of the exercise is twofold: 
#
# 1. Learn how to deal with real world sensor timeseries data, and organize them efficiently using python dataframes.
#
# 2. Apply data science tools to model the measurements, and use the learned model to process them (e.g., detect drifts in the sensor measurements). 
#
# The sensor network consists of 46 sites, located in different parts of the city. Each site contains three different sensors measuring (a) **CO2 concentration**, (b) **temperature**, and (c) **humidity**. Beside these measurements, we have the following additional information that can be used to process the measurements: 
#
# 1. The **altitude** at which the CO2 sensor is located, and the GPS coordinates (latitude, longitude).
#
# 2. A clustering of the city of Zurich in 17 different city **zones** and the zone in which the sensor belongs to. Some characteristic zones are industrial area, residential area, forest, glacier, lake, etc.
#
# ## Prior knowledge
#
# The average value of the CO2 in a city is approximately 400 ppm. However, the exact measurement in each site depends on parameters such as the temperature, the humidity, the altitude, and the level of traffic around the site. For example, sensors positioned in high altitude (mountains, forests), are expected to have a much lower and uniform level of CO2 than sensors that are positioned in a business area with much higher traffic activity. Moreover, we know that there is a strong dependence of the CO2 measurements, on temperature and humidity.
#
# Given this knowledge, you are asked to define an algorithm that curates the data, by detecting and removing potential drifts. **The algorithm should be based on the fact that sensors in similar conditions are expected to have similar measurements.** 
#
# ## To start with
#
# The following csv files in the `~/shared/data/` folder will be needed: 
#
# 1. `CO2_sensor_measurements.csv`
#     
#    __Description__: It contains the CO2 measurements `CO2`, the name of the site `LocationName`, a unique sensor identifier `SensorUnit_ID`, and the time instance in which the measurement was taken `timestamp`.
#     
# 2. `temperature_humidity.csv`
#
#    __Description__: It contains the temperature and the humidity measurements for each sensor identifier, at each timestamp `Timestamp`. For each `SensorUnit_ID`, the temperature and the humidity can be found in the corresponding columns of the dataframe `{SensorUnit_ID}.temperature`, `{SensorUnit_ID}.humidity`.
#     
# 3. `sensor_metadata_updated.csv`
#
#    __Description__: It contains the name of the site `LocationName`, the zone index `zone`, the altitude in meters `altitude`, the longitude `LON`, and the latitude `LAT`. 
#
# Import the following python packages:

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.linear_model import LinearRegression

import time
from ipywidgets import interactive, widgets, interact
from sklearn.model_selection import TimeSeriesSplit

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

plt.rcParams['font.family'] = 'DejaVu Sans' # prevent font not found warnings

# %% [markdown]
# ## PART I: Handling time series with pandas (10 points)
#
# The following scripts will copy the data to your home folder if it is not already present. They will output the absolute path of the data in your home folder, but for portability across different user accounts, we recommend using the shorthand _~/shared/data_ in your code.

# %%
# !./setup.sh

# %%
DATA_DIR='~/shared/data/'

# %%
CO2 = f'{DATA_DIR}/CO2_sensor_measurements.csv'
humidity = f'{DATA_DIR}/temperature_humidity.csv'
meta = f'{DATA_DIR}/sensors_metadata_updated.csv'

# %% [markdown]
# ### a) **10/10**
#
# Merge the `CO2_sensor_measurements.csv`, `temperature_humidity.csv`, and `sensors_metadata.csv`, into a single dataframe. 
#
# * The merged dataframe contains:
#     - index: the time instance `timestamp` of the measurements
#     - columns: the location of the site `LocationName`, the sensor ID `SensorUnit_ID`, the CO2 measurement `CO2`, the `temperature`, the `humidity`, the `zone`, the `altitude`, the longitude `lon` and the latitude `lat`.
#
# | timestamp | LocationName | SensorUnit_ID | CO2 | temperature | humidity | zone | altitude | lon | lat |
# |:---------:|:------------:|:-------------:|:---:|:-----------:|:--------:|:----:|:--------:|:---:|:---:|
# |    ...    |      ...     |      ...      | ... |     ...     |    ...   |  ... |    ...   | ... | ... |
#
#
#
# * For each measurement (CO2, humidity, temperature), __take the average over an interval of 30 min__. 
#
# * If there are missing measurements, __interpolate them linearly__ from measurements that are close by in time.
#
# __Hints__: The following methods could be useful
#
# 1. ```python 
# pandas.DataFrame.resample()
# ``` 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
#     
# 2. ```python
# pandas.DataFrame.interpolate()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
#     
# 3. ```python
# pandas.DataFrame.mean()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html
#     
# 4. ```python
# pandas.DataFrame.append()
# ```
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html

# %% editable=true slideshow={"slide_type": ""}
# Read data
co2_df = pd.read_csv(CO2, delimiter='\t')
co2_df['timestamp'] = pd.to_datetime(co2_df['timestamp'])
co2_df.set_index('timestamp', inplace=True)
# Group by LocationName and SensorUnit_ID, then resample each group
co2_df = co2_df.groupby(['LocationName', 'SensorUnit_ID']).resample('30min').agg({
    'CO2': 'mean'
})

co2_df = co2_df.sort_values(by=["SensorUnit_ID", 'timestamp']).reset_index()
co2_df["SensorUnit_ID"] = co2_df["SensorUnit_ID"].astype(str)
# co2_df


# %% editable=true slideshow={"slide_type": ""}
temp_humid_df = pd.read_csv(humidity, delimiter='\t')

temp_humid_df['Timestamp'] = pd.to_datetime(temp_humid_df['Timestamp'])

# Melt the dataframe into long format
temp_humid_df = temp_humid_df.melt(id_vars=["Timestamp"], var_name="Sensor_Measurement", value_name="Value")

# Extract SensorUnit_ID and Measurement_Type
temp_humid_df["SensorUnit_ID"] = temp_humid_df["Sensor_Measurement"].str.split(".").str[0]
temp_humid_df["Measurement_Type"] = temp_humid_df["Sensor_Measurement"].str.split(".").str[1]

# Pivot to get temperature and humidity in separate columns
temp_humid_df = temp_humid_df.pivot_table(index=["SensorUnit_ID", "Timestamp"], 
                                               columns=["Measurement_Type"], 
                                               values="Value").reset_index()

temp_humid_df.columns = ["SensorUnit_ID", "timestamp", "humidity", "temperature"]
temp_humid_df.set_index("timestamp", inplace=True)
temp_humid_df = temp_humid_df.groupby("SensorUnit_ID").resample("30min").mean().reset_index()

# temp_humid_df

# %%
metadata = pd.read_csv(meta)
metadata = metadata.drop(columns=['Unnamed: 0','X','Y'])

location_sensor_map = co2_df[["LocationName", "SensorUnit_ID"]].drop_duplicates()

metadata = pd.merge(metadata, location_sensor_map, on="LocationName", how="right")
# metadata

# %% editable=true slideshow={"slide_type": ""}
merged_df = pd.merge(metadata, temp_humid_df, on=["SensorUnit_ID"], how="right")

final_df = pd.merge(merged_df, co2_df, on=["SensorUnit_ID", "LocationName", "timestamp"], how="left")
                    
# Sort the DataFrame by SensorUnit_ID and timestamp
final_df = final_df.sort_values(by=["SensorUnit_ID", "timestamp"])

# Interpolate missing values for each SensorUnit_ID
final_df = final_df.interpolate()

final_df = final_df[["timestamp", "LocationName", "SensorUnit_ID", "CO2", "temperature", "humidity", "zone", "altitude", "LON", "LAT"]]
final_df = final_df.rename(columns={"LON": "lon", "LAT": "lat"})
final_df


# %% [markdown]
# ## PART II: Data visualization (15 points)

# %% [markdown]
# ### a) **5/15** 
# Group the sites based on their altitude, by performing K-means clustering. 
# - Find the optimal number of clusters using the [Elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)). 
# - Wite out the formula of metric you use for Elbow curve.
# - Perform clustering with the optimal number of clusters and add an additional column `altitude_cluster` to the dataframe of the previous question indicating the altitude cluster index. 
# - Report your findings.
#
# __Note__: [Yellowbrick](http://www.scikit-yb.org/) is a very nice Machine Learning Visualization extension to scikit-learn, which might be useful to you. 

# %%
final_df.info()

# %% [markdown]
# __Answer:__  The Elbow method groups the sensor stations into altitude clusters based on how close individual datapoints fall within the proposed cluster centers. "Closenes" or low distortion is calculated through sum of of squared distances.  
#
# $$
# \sum_{j=1}^k \sum_{x_i \in C_j} \| x_i - \mu_j \|^2
# $$
#
# $\textbf{x}_i$ being the datapoint belonging to cluster $C_j$, and $\mathbf{\mu}_j$ being the cluster centroid.   
#
# Elbow method does a sweep over number of clusters and checks the distortion. Small $k$ will fail to capture finer differences in altitude. Once a certain $k$ value is reached, further incerementation will not improve the fit. This point is knowns as the elbow.
#
# We found that the optimal amount of clusters is $k=5$

# %%
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Copy the dataset
df_copy = final_df.copy()

# Extract altitude values as a 2D NumPy array
altd = df_copy[['altitude']].values  # Ensures correct shape (n,1)
# zone = df_copy['zone']??????????????

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 20), metric='distortion', timings=False)   # metric used: 'distortion'

# Visualize the optimal k and extract it
visualizer.fit(altd)
visualizer.show()

best_k = visualizer.elbow_value_  
print(f"best k={best_k}")


# Now fit KMeans with the best k
final_model = KMeans(n_clusters=best_k, random_state=42)      # 42 for portabile ML
final_df['cluster_id'] = final_model.fit_predict(altd)

print("found cluster tags:")
print(final_df['cluster_id'].unique())
# print(df_copy.head())

# %% [markdown]
# ### b) **5/15** 
#
# Use `plotly` (or other similar graphing libraries) to create an interactive plot of the monthly median CO2 measurement for each site with respect to the altitude. 
#
# Add proper title and necessary hover information to each point, and give the same color to stations that belong to the same altitude cluster.

# %%
def plot_co2(location):

    if not location:
        return
      
    # print(location[0]) 
    df_copy=final_df.copy()
    df_copy["month"] = df_copy["timestamp"].dt.to_period("M")
    df_selection = df_copy[df_copy.LocationName.isin(np.array(location))]
    df_grouped = df_selection.groupby(["month", "altitude", "LocationName", "cluster_id"])["CO2"].median().reset_index()
    # pivot_df = df_grouped.pivot(index="altitude", columns="month", values="CO2")  
    
    # Plot using Plotly Express

    fig_cluster = px.bar(
        df_grouped, x="LocationName", y="CO2",
        color= "cluster_id",
        title=f"Monthly median CO2 concentration [ppm] for each Location aranged by altitude",
        labels={"CO2": "Median CO2 [ppm]", "cluster_id": "Cluster ID", "LocationName":"Location Name (altd incremented: LtR)", "altitude": "Altitude [m]"},
        barmode="group",
    hover_data={  
        "CO2": True,  
        "cluster_id": True,  
        "LocationName": True,  
        "altitude": True  
        } 
    )
    fig_cluster.show()
    
    fig_altd = px.bar(
        df_grouped, x="LocationName", y="CO2",
        color= "altitude",
        title=f"Monthly median CO2 concentration [ppm] for each Location aranged by cluster",
        labels={"CO2": "Median CO2 [ppm]", "cluster_id": "Cluster ID", "LocationName":"Location Name", "altitude": "Altitude [m]"},
        barmode="group",
        hover_data={  
            "CO2": True,  
            "cluster_id": True,  
            "LocationName": True,  
            "altitude": True  
        } 
    )
    fig_altd.show()


    fig_scatter = px.scatter(
        df_grouped, 
        x="altitude", 
        y="CO2",
        color="cluster_id",
        # size="CO2", 
        title="CO2[ppm] - Altitude[m] Scatter Plot: visualization of the clustering",
        labels={
            "CO2": "Median CO2 [ppm]",
            "cluster_id": "Cluster ID",
            "LocationName": "Location Name",
            "altitude": "Altitude [m]"
        },
        color_continuous_scale="Viridis",  
        hover_data={  
            "CO2": True,  
            "cluster_id": True,  
            "LocationName": True,  
            "altitude": True  
        } 
    )

    fig_scatter.show()

# %%
default_values = tuple(final_df["LocationName"].unique())
location_selector = widgets.SelectMultiple(
    options = final_df.LocationName.unique(),
    value = default_values,
    description = 'Location: '
)

_ = interact(plot_co2, location = location_selector)

# %% [markdown]
# __Answer__: 
# The method detects 5 altitude clusters. The top bar-plot visualises monthly median CO2 measurements for each location. The colour gradient suggests there are certain altitude levels. Right under, the same information is ranked based on cluster ID. Colour gradient reveals correlation between altitude levels and the clusters. Finally we use CO2-altitude scatter plot to show how the location sites group up into these clusters. 
#
# Out of the 5 total clusters, 3 contain majority of the locations. They center around 450, 500 and 650m. Most likely, the grouping corresponds to stations around Zurich, the low-lying northern part of Switzerland and mountainous south. The 2 clusters at 700m and 850m correspond to single stations grouped by themselves. 
#
# <img src="http://carbosense.wdfiles.com/local--files/main:project/CarboSense_MAP_20191113_LowRes.jpg" width="500">
#

# %% [markdown]
# ### c) **5/15**
#
# Use `plotly` (or other similar graphing libraries) to plot an interactive time-varying density heatmap of the mean daily CO2 concentration for all the stations. Add proper title and necessary hover information.
#
# __Hints:__ Check following pages for more instructions:
# - [Animations](https://plotly.com/python/animations/)
# - [Density Heatmaps](https://plotly.com/python/mapbox-density-heatmaps/)

# %%
import plotly.express as px


# %%
def plot_co2_DHM(location):
    if not location:
        return
    df_copy = final_df.copy()
    
    df_copy["Day"] = df_copy["timestamp"].dt.to_period("D")
    df_selection = df_copy[df_copy.LocationName.isin(location)]
    aggregate_by = "Day"
    df_aggregated = df_selection.groupby([aggregate_by, "lat", "lon", "altitude", "LocationName"])["CO2"].mean().reset_index()
    min_val = 300
    max_val = 600


    fig_DHM = px.density_mapbox(df_aggregated, 
        lat='lat', 
        lon='lon', 
        z='CO2',  # CO2 values for heat intensity
        animation_frame=aggregate_by,
        color_continuous_scale="Viridis", 
        title="CO2 Concentration [ppm] Daily Heatmap for October 2017",
        mapbox_style="carto-positron",
        range_color=[min_val, max_val],
        labels={"CO2": "D-av. CO2 [ppm]", "LocationName":"Location Name", "altitude": "Altitude (m)"},
        hover_data={'LocationName': True, 'CO2': True, 'lat': True, 'lon': True, 'altitude':True},
        zoom=11,
        height=800,
        width=1200)

    fig_DHM.show()


# %%
default_values = tuple(final_df["LocationName"].unique())
location_selector_DHM = widgets.SelectMultiple(
    options = final_df.LocationName.unique(),
    value = default_values,
    description = 'Location: '
)

_ = interact(plot_co2_DHM, location = location_selector_DHM)

# %% [markdown]
# ## PART III: Model fitting for data curation (35 points)

# %% [markdown]
# ### a) **5/35**
#
# The domain experts in charge of these sensors report that one of the CO2 sensors `ZSBN` is exhibiting a drift on Oct. 24. Verify the drift by visualizing the CO2 concentration of the drifting sensor and compare it with some other sensors from the network. 

# %%
final_df.head()


# %%
def check_drift_test(location):
    if not location:
        return
    df_copy = final_df.copy()
    df_selection = df_copy[
        (final_df['timestamp'].dt.date >= pd.to_datetime('2017-10-15').date())&
        (final_df['timestamp'].dt.date < pd.to_datetime('2017-10-30').date())
    ]
    
    places=["ZSBN"]+list(location)
    df_places = df_selection[df_selection.LocationName.isin(places)]

   
    fig_Drift = px.line(df_places,x='timestamp',y='CO2',color="LocationName")
    fig_Drift.show()

# %%
# Selecti
default_values = tuple(np.array(final_df["LocationName"].unique())[1:8])
sensor_compare = widgets.SelectMultiple(
    options=final_df["LocationName"].unique(),  # Get unique locations
    value = default_values,
    description="Select Locations to compare:"
)

_ = interact(check_drift_test, location = sensor_compare)

print(f'{np.array(final_df["LocationName"].unique())[6]}, Index: {6} has a noticable drift')

# %% [markdown]
# We see that on the 24.11 there is a sharp drop in the magnitute of the measured CO2 values of the ZSBN sensor. We expect that, had the drift not happened, the graph will follow similar trend as the rest. 

# %% [markdown]
# ### b) **10/35**
#
# The domain experts ask you if you could reconstruct the CO2 concentration of the drifting sensor had the drift not happened. You decide to:
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features the covariates not affected by the malfunction (such as temperature and humidity)
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __95% confidence interval__ obtained from cross validation: assume that the prediction error follows a normal distribution and is independent of time.
# - What do you observe? Report your findings.
#
# __Note:__ Cross validation on time series is different from that on other kinds of datasets. The following diagram illustrates the series of training sets (in orange) and validation sets (in blue). For more on time series cross validation, there are a lot of interesting articles available online. scikit-learn provides a nice method [`sklearn.model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
#
# ![ts_cv](https://player.slideplayer.com/86/14062041/slides/slide_28.jpg)

# %%
tscv = sklearn.model_selection.TimeSeriesSplit()

drift_sensor_df = final_df[
    (final_df['SensorUnit_ID'] == '1031') &
    (final_df['timestamp'].dt.date >= pd.to_datetime('2017-10-01').date())&
    (final_df['timestamp'].dt.date < pd.to_datetime('2017-10-24').date())
]

residuals = []

for train_index, test_index in tscv.split(drift_sensor_df):
    # Split the DataFrame into training and testing sets
    train_df = drift_sensor_df.iloc[train_index]
    test_df = drift_sensor_df.iloc[test_index]
    
    X_train = train_df[['temperature', 'humidity']]
    Y_train = train_df['CO2']
    
    X_test = test_df[['temperature', 'humidity']]
    Y_test = test_df['CO2']
    
    lin_model = sklearn.linear_model.LinearRegression()
    lin_model.fit(X_train, Y_train)
    
    Y_pred = lin_model.predict(X_test)
    # Evaluate the model
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    print("MSE: ", mse)

    residuals.extend(Y_test - Y_pred)

residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)
confidence_interval = 1.96 * residuals_std  # 1.96 is the z-score for 95% confidence

test_df.loc[:, "CO2"] = Y_pred
drift_sensor = final_df[(final_df['SensorUnit_ID'] == '1031')]

after_drift_sensor_df = drift_sensor[drift_sensor['timestamp'].dt.date >= pd.to_datetime('2017-10-24').date()]
Y_pred = lin_model.predict(after_drift_sensor_df[['temperature', 'humidity']])
after_drift_sensor_df.loc[:, 'CO2'] = Y_pred
predict_df = pd.concat([test_df, after_drift_sensor_df])

drift_sensor_label = drift_sensor[drift_sensor['timestamp'].dt.date >= pd.to_datetime('2017-10-20 03:30:00').date()].assign(sensor_label='Drift Sensor')
predict_label = predict_df.assign(sensor_label='Predict Data')

combined_df = pd.concat([drift_sensor_label, predict_label])

# Plot the combined DataFrame
fig = px.line(
    combined_df,
    x="timestamp",
    y="CO2",
    color="sensor_label",  # Differentiate lines by sensor_label
    title="CO2 Levels: ZSBN Drift Sensor vs. Train Data"
)

fig.add_trace(
    go.Scatter(
        x=predict_df['timestamp'],
        y=predict_df['CO2'] + confidence_interval,
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper CI'
    )
)

fig.add_trace(
    go.Scatter(
        x=predict_df['timestamp'],
        y=predict_df['CO2'] - confidence_interval,
        fill='tonexty',  # Fill area between this trace and the previous one
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Lower CI'
    )
)

vline_times = [
    '2017-10-20 03:30:00',
    '2017-10-24 00:00:00'
]

# Add vertical lines to the plot
for vline_time in vline_times:
    fig.add_vline(
        x=pd.to_datetime(vline_time),  # Convert to datetime if necessary
        line_dash="dash",              # Line style: "solid", "dot", "dash", etc.
        line_color="black",              # Color of the vertical line
    )
    
fig.show()

# %% [markdown]
# __Answer:__ Using covariates of the sensor that were not affected by the malfunction we manage to reproduce a very rough predicition of future measurements. The drifting lowers the value of the measurements but we expect predictions to match the shape of the actual measurements. We see that our model correctly predicts the positions of the big peaks and dips but is very smooth and fails to capture the smaller variations in the measurement. The large error margin (the shaded region) and large mean square error further confirms that the estimation is not very good.

# %% [markdown]
# ### c) **10/35**
#
# In your next attempt to solve the problem, you decide to exploit the fact that the CO2 concentrations, as measured by the sensors __experiencing similar conditions__, are expected to be similar.
#
# - Find the sensors sharing similar conditions with `ZSBN`. Explain your definition of "similar condition".
# - Fit a linear regression model to the CO2 measurements of the site, by considering as features:
#     - the information of provided by similar sensors
#     - the covariates associated with the faulty sensors that were not affected by the malfunction (such as temperature and humidity).
# - Create an interactive plot with `plotly` (or other similar graphing libraries):
#     - the actual CO2 measurements
#     - the values obtained by the prediction of the linear model for the entire month of October
#     - the __confidence interval__ obtained from cross validation
# - What do you observe? Report your findings.

# %%
drifting_sensor_id = "1031"

pre_drift_start = pd.to_datetime("2017-09-01")
pre_drift_end   = pd.to_datetime("2017-10-20 03:30:00")

stable_mask = (
    (final_df["timestamp"] >= pre_drift_start) &
    (final_df["timestamp"] < pre_drift_end)
)
stable_df = final_df[stable_mask].copy()

sensor_stats = (
    stable_df.groupby("SensorUnit_ID")
    .agg({"altitude": "mean", "temperature": "mean", "humidity": "mean"})
    .reset_index()
)

drifting_stats = sensor_stats[sensor_stats["SensorUnit_ID"] == drifting_sensor_id]
drifting_alt = drifting_stats["altitude"].values[0]
drifting_temp = drifting_stats["temperature"].values[0]
drifting_hum = drifting_stats["humidity"].values[0]

def euclidean_distance(row):
    return np.sqrt(
        (row["altitude"] - drifting_alt)**2 +
        (row["temperature"] - drifting_temp)**2 +
        (row["humidity"] - drifting_hum)**2
    )

sensor_stats["distance"] = sensor_stats.apply(euclidean_distance, axis=1)
sensor_stats = sensor_stats[sensor_stats["SensorUnit_ID"] != drifting_sensor_id]

# Pick top-5 similar sensors
K = 5
sensor_stats_sorted = sensor_stats.sort_values("distance")
similar_sensors = sensor_stats_sorted.head(K)["SensorUnit_ID"].tolist()

print("Similar sensors for ZSBN (1031):", similar_sensors)
# Focus on the entire month of October
oct_start = pd.to_datetime("2017-10-20")
oct_end   = pd.to_datetime("2017-10-31 23:59:59")

mask_oct = (
    (final_df["timestamp"] >= oct_start) &
    (final_df["timestamp"] <= oct_end) &
    (final_df["SensorUnit_ID"].isin([drifting_sensor_id] + similar_sensors))
)
oct_df = final_df[mask_oct].copy()

# Pivot CO2 for drifting + similar sensors
pivot_co2 = oct_df.pivot_table(
    index="timestamp",
    columns="SensorUnit_ID",
    values="CO2"
).add_prefix("CO2_")

# We also need drifting sensor's temperature/humidity (assuming they're reliable)
pivot_temp = oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id].pivot_table(
    index="timestamp",
    columns="SensorUnit_ID",
    values="temperature"
).add_prefix("temp_")

pivot_hum = oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id].pivot_table(
    index="timestamp",
    columns="SensorUnit_ID",
    values="humidity"
).add_prefix("hum_")

merged_df = pivot_co2.join(pivot_temp, how="inner").join(pivot_hum, how="inner")
merged_df = merged_df.reset_index().sort_values("timestamp")

similar_sensor_co2_cols = [f"CO2_{sid}" for sid in similar_sensors]
feature_cols = similar_sensor_co2_cols + [f"temp_{drifting_sensor_id}", f"hum_{drifting_sensor_id}"]

target_col = f"CO2_{drifting_sensor_id}"  # We'll predict drifting sensor's CO₂

# Subset the data for cross-validation: 10/01–10/24
cv_start = pd.to_datetime("2017-10-01")
cv_end   = pd.to_datetime("2017-10-24")
cv_mask = (merged_df["timestamp"] >= cv_start) & (merged_df["timestamp"] < cv_end)
cv_df = merged_df[cv_mask].copy()

X_all = cv_df[feature_cols]
Y_all = cv_df[target_col]

tscv = sklearn.model_selection.TimeSeriesSplit()
residuals = []

for train_index, test_index in tscv.split(cv_df):
    train_df = cv_df.iloc[train_index]
    test_df  = cv_df.iloc[test_index]
    
    X_train = train_df[feature_cols]
    Y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    Y_test = test_df[target_col]
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    print("MSE:", mse)
    
    # Collect residuals for confidence interval
    residuals.extend(Y_test - Y_pred)

residuals_mean = np.mean(residuals)
residuals_std  = np.std(residuals)
confidence_interval = 1.96 * residuals_std

print("Mean residual:", residuals_mean)
print("Std residual:", residuals_std)
print("95% conf interval +/-:", confidence_interval)

# Train final model on ALL data from 10/01 to 10/24
final_model = LinearRegression()
final_model.fit(X_all, Y_all)
X_entire_month = merged_df[feature_cols]
Y_pred_month = final_model.predict(X_entire_month)

merged_df["CO2_pred"] = Y_pred_month
merged_df["CO2_pred_upper"] = merged_df["CO2_pred"] + confidence_interval
merged_df["CO2_pred_lower"] = merged_df["CO2_pred"] - confidence_interval
import plotly.express as px
import plotly.graph_objects as go

fig = px.line(
    merged_df,
    x="timestamp",
    y=target_col,  # Actual drifting sensor CO2
    title="ZSBN (1031) Actual vs Predicted CO₂ (Entire October)"
)
fig.update_traces(name="Actual (ZSBN) CO₂", showlegend=True)

# Add predicted line
fig.add_scatter(
    x=merged_df["timestamp"],
    y=merged_df["CO2_pred"],
    mode='lines',
    name="Predicted CO₂"
)

# Add confidence interval
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_upper"],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Upper 95% CI"
    )
)
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_lower"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Lower 95% CI"
    )
)

vline_times = [
    '2017-10-24 00:00:00',  # drift
]
for vline in vline_times:
    fig.add_vline(
        x=pd.to_datetime(vline),
        line_dash="dash",
        line_color="black"
    )

fig.show()


# %% [markdown]
# __Answer__: We define sensors with similar conditions to be sensors that have similar *altitude*, *humidity* and *temperature* based on the "Prior knowledge" section. We use Euclidian distance to determine the 5 sensors with conditions closest to the faulty device and train a linear regression model on their CO2 measurements to predict the future measurements of the faulty sensor if the drift didn't happen.
#
# This method offers significant improvement over part b), it captures the large peaks and dips of the graph. It also follows approximately the same trends as the measurement of the drifted sensor but lifted up as if the drift did not happen. The improved accuracy is also shown by the smaller error margin and mean square error.

# %% [markdown]
# ### d) **10/35**
#
# Now, instead of feeding the model with all features, you want to do something smarter by using linear regression with fewer features.
#
# - Start with the same sensors and features as in question c)
# - Leverage at least two different feature selection methods
# - Create similar interactive plot as in question c)
# - Describe the methods you choose and report your findings

# %%
from sklearn.model_selection import TimeSeriesSplit, train_test_split
drifting_sensor_id = "1031"
similar_sensors = ['1014', '1018', '1033', '1298', '1017'] 

oct_start = pd.to_datetime("2017-10-20")
oct_end   = pd.to_datetime("2017-10-31 23:59:59")

mask_oct = (
    (final_df["timestamp"] >= oct_start) &
    (final_df["timestamp"] <= oct_end) &
    (final_df["SensorUnit_ID"].isin([drifting_sensor_id] + similar_sensors))
)
oct_df = final_df[mask_oct].copy()

# Pivot CO2
pivot_co2 = oct_df.pivot_table(
    index="timestamp",
    columns="SensorUnit_ID",
    values="CO2"
).add_prefix("CO2_")

# Pivot drifting sensor’s temperature/humidity
pivot_temp = (oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id]
              .pivot_table(index="timestamp", columns="SensorUnit_ID", values="temperature")
              .add_prefix("temp_"))

pivot_hum = (oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id]
             .pivot_table(index="timestamp", columns="SensorUnit_ID", values="humidity")
             .add_prefix("hum_"))

# Merge
merged_df = pivot_co2.join(pivot_temp, how="inner").join(pivot_hum, how="inner")
merged_df = merged_df.reset_index().sort_values("timestamp")

cv_start = pd.to_datetime("2017-10-01")
cv_end   = pd.to_datetime("2017-10-24")

cv_mask = (merged_df["timestamp"] >= cv_start) & (merged_df["timestamp"] < cv_end)
cv_df = merged_df[cv_mask].copy()

feature_candidates = [f"CO2_{sid}" for sid in similar_sensors] + [f"temp_{drifting_sensor_id}", f"hum_{drifting_sensor_id}"]
target_col = f"CO2_{drifting_sensor_id}"

# %% [markdown]
# ## __Method 1__: Filter-Based selected features by absolute correlation 

# %%
corr_scores = {}
Y = cv_df[target_col]

for feat in feature_candidates:
    corr = cv_df[[feat, target_col]].corr().iloc[0,1]
    corr_scores[feat] = abs(corr)  # use absolute correlation

# Sort features by descending correlation
sorted_features_corr = sorted(corr_scores.keys(), key=lambda f: corr_scores[f], reverse=True)

# Picking top 2 correlated features
topN = 2
selected_feats_corr = sorted_features_corr[:topN]

print("Features sorted by absolute correlation to drifted sensor:")
for f in sorted_features_corr:
    print(f"{f}: {corr_scores[f]:.3f}")

print(f"\nTop {topN} candidates (Filter-based) = {selected_feats_corr}")

# %%
oct_start = pd.to_datetime("2017-10-20")
oct_end   = pd.to_datetime("2017-10-31 23:59:59")

mask_oct = (
    (final_df["timestamp"] >= oct_start) &
    (final_df["timestamp"] <= oct_end) &
    (final_df["SensorUnit_ID"].isin([drifting_sensor_id, "1033", "1017"]))
)
oct_df = final_df[mask_oct].copy()

# 2) Pivot CO₂ to columns: "CO2_1031", "CO2_1033", "CO2_1017"
pivot_co2 = oct_df.pivot_table(
    index="timestamp",
    columns="SensorUnit_ID",
    values="CO2"
).add_prefix("CO2_")

# 3) Pivot drifting sensor’s temperature/humidity if needed
# (Optional if not using them as features, but let's show the approach)
pivot_temp = (
    oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id]
    .pivot_table(index="timestamp", columns="SensorUnit_ID", values="temperature")
    .add_prefix("temp_")
)

pivot_hum = (
    oct_df[oct_df["SensorUnit_ID"] == drifting_sensor_id]
    .pivot_table(index="timestamp", columns="SensorUnit_ID", values="humidity")
    .add_prefix("hum_")
)

# 4) Merge into a single DF
merged_df = pivot_co2.join(pivot_temp, how="inner").join(pivot_hum, how="inner")
merged_df = merged_df.reset_index().sort_values("timestamp")

from sklearn.model_selection import TimeSeriesSplit

# 1) Define the training window: Oct 1–24
train_start = pd.to_datetime("2017-10-01")
train_end   = pd.to_datetime("2017-10-24")

train_mask = (merged_df["timestamp"] >= train_start) & (merged_df["timestamp"] < train_end)
train_df = merged_df[train_mask].copy()

# 2) Features + target
feature_cols = ["CO2_1033","CO2_1017"]  # top 2 from correlation
target_col = "CO2_1031"

X_all = train_df[feature_cols]
Y_all = train_df[target_col]

tscv = TimeSeriesSplit(n_splits=5)
residuals = []

for train_idx, test_idx in tscv.split(train_df):
    sub_train_df = train_df.iloc[train_idx]
    sub_test_df  = train_df.iloc[test_idx]
    
    X_train = sub_train_df[feature_cols]
    Y_train = sub_train_df[target_col]
    X_test  = sub_test_df[feature_cols]
    Y_test  = sub_test_df[target_col]
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    # collect residuals
    residuals.extend(Y_test - Y_pred)

residuals = np.array(residuals)
res_mean = residuals.mean()   # usually near 0
res_std  = residuals.std()
ci_95 = 1.96 * res_std

print("Residual std:", round(res_std,2))
print("95% conf interval +/-:", round(ci_95,2))
# 1) Final model on full training window
final_model = LinearRegression()
final_model.fit(X_all, Y_all)

# 2) Predict entire month
X_entire = merged_df[feature_cols]
Y_pred_entire = final_model.predict(X_entire)

merged_df["CO2_pred"] = Y_pred_entire
merged_df["CO2_pred_upper"] = merged_df["CO2_pred"] + ci_95
merged_df["CO2_pred_lower"] = merged_df["CO2_pred"] - ci_95
fig = px.line(
    merged_df,
    x="timestamp",
    y="CO2_1031",  # actual drifting sensor's CO₂
    title="ZSBN (1031) Actual vs Predicted CO₂ (Top 2 Corr Features)"
)
fig.update_traces(name="Actual (Drifting Sensor)", showlegend=True)

# Add predicted line
fig.add_scatter(
    x=merged_df["timestamp"],
    y=merged_df["CO2_pred"],
    mode='lines',
    name="Predicted CO₂"
)

# Add confidence interval
# confidence band
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_upper"],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Upper CI"
    )
)
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_lower"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Lower CI"
    )
)

fig.show()


# %% [markdown]
# ### Describe the method and report findings
#
# - This method picks out the humidity and temperature from the drifted sensor as reliable enough readings, which means temperature and humidity is **STRONGLY** correlated to the CO2 levels and very useful for reconstructing and curating the data.
# - We use the CO2 from other sensors in the same network, here those that same similar conditions. The top two candidates (with highest correlation) come from the sensors **1033** and **1017**, but the rest also hold useful CO2 patterns.

# %% [markdown]
# ## __Method 2__: Recursive Feature Element (RFE)
#
# Given an external estimator that assigns weights to features, we wish to select features by recursively considering smaller and smaller sets of features.

# %%
from sklearn.feature_selection import RFE

# We decide how many features we want to keep, say 3 again
n_features_to_select = 3

rfe_estimator = LinearRegression()
rfe_selector = RFE(rfe_estimator, n_features_to_select=n_features_to_select, step=1)

X = cv_df[feature_candidates]
Y = cv_df[target_col]

rfe_selector.fit(X, Y)

selected_feats_rfe = []
for feat, rank in zip(feature_candidates, rfe_selector.ranking_):
    if rank == 1:
        selected_feats_rfe.append(feat)

print("RFE-chosen features:", selected_feats_rfe)


tscv = TimeSeriesSplit(n_splits=5)

def evaluate_features(feature_list):
    residuals = []
    X_all = cv_df[feature_list]
    Y_all = cv_df[target_col]
    
    for train_idx, test_idx in tscv.split(cv_df):
        X_train = X_all.iloc[train_idx]
        Y_train = Y_all.iloc[train_idx]
        X_test  = X_all.iloc[test_idx]
        Y_test  = Y_all.iloc[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
        residuals.extend(Y_test - Y_pred)
        print("MSE:", mse)
    
    residuals = np.array(residuals)
    ci_95 = 1.96 * residuals.std()
    
    return ci_95, residuals.std()

# Evaluate the correlation-based subset
corr_ci, corr_res_std = evaluate_features(selected_feats_corr)
print(f"[Filter-based] 95% CI = +/- {corr_ci:.2f}, residual std = {corr_res_std:.2f}")

# Evaluate the RFE-based subset
rfe_ci, rfe_res_std = evaluate_features(selected_feats_rfe)
print(f"[Wrapper-based] 95% CI = +/- {rfe_ci:.2f}, residual std = {rfe_res_std:.2f}")



# %% [markdown]
# ### Describe the method and report findings
#
# - This method picks out the humidity and temperature from the drifted sensor as well, the same as our previous method, which means temperature and humidity is **STRONGLY** correlated to the CO2 levels (as expected from prior knowledge).
# - The difference with RFE is that it recursively picked out one sensor for the CO2 readings to be the best candidate. We can see that the MSE for the RFE method is much better comparatively, even for the outlier spike, and the confidence interval for the prediction is stronger.
#
# We can conclude with that the only necessary features to curating the data is the drifted sensors temperature and humidity and a reliable neighbors CO2 levels.

# %%
# Final model training
final_model = LinearRegression()
train_mask_full = (merged_df["timestamp"] >= cv_start) & (merged_df["timestamp"] < cv_end)
train_df = merged_df[train_mask_full].copy()

X_train_full = train_df[selected_feats_corr]  # If you pick the filter-based subset
Y_train_full = train_df[target_col]

final_model.fit(X_train_full, Y_train_full)

# Predict entire October
X_entire = merged_df[selected_feats_corr]
Y_pred_entire = final_model.predict(X_entire)

ci_95 = corr_ci  # or rfe_ci if using RFE
merged_df["CO2_pred"] = Y_pred_entire
merged_df["CO2_pred_upper"] = merged_df["CO2_pred"] + ci_95
merged_df["CO2_pred_lower"] = merged_df["CO2_pred"] - ci_95

fig = px.line(
    merged_df,
    x="timestamp",
    y=target_col,  # Actual drifting sensor's CO₂
    title="ZSBN (1031) Actual vs Predicted CO₂ (Fewer Features)"
)
fig.update_traces(name="Actual (ZSBN) CO₂", showlegend=True)

#prediction
fig.add_scatter(
    x=merged_df["timestamp"],
    y=merged_df["CO2_pred"],
    mode='lines',
    name="Predicted (model)"
)

# confidence band
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_upper"],
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Upper CI"
    )
)
fig.add_trace(
    go.Scatter(
        x=merged_df["timestamp"],
        y=merged_df["CO2_pred_lower"],
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="Lower CI"
    )
)

fig.show()

# %% [markdown]
# __Comment:__ It is clear that we are curating and removing the drift by training the model on a period before the drift such that it knows the sensor's normal behavior in relation to temperature and humidity and the others CO2 levels. After the drift starts at 24th. of October, we rely on the model's predicted CO2 rather than the faulty sensor's actual reading. We give our best prediction to the CO2 signal with our reconstructed model with even less features, so there was no point in using all the features as earlier.

# %% [markdown]
# # That's all, folks!
