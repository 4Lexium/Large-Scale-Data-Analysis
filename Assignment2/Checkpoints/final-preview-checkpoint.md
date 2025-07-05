# Final Assignment: Robust Journey Planning

PREVIEW

**Executive summary:** Build a robust SBB journey planner, and make a short video presentation of it, to be done as a **team project**.
* A short (7min max) video presentation of your work is due before **May 22nd, 23:59 CEST**
* In-progress code (not necessarily functional) is due before **May 26th, 23:59**, final code is due before **May 30, 23:59 CEST**.

----
## HOW-TO

**TODO**

_This section intentionally blank. Students must complete it with instructions on how to use the code in their project._

----
## Problem Motivation

Imagine you are a regular user of the public transport system, and you are checking the operator's schedule to meet your friends for a class reunion.
The choices are:

1. You could leave in 10mins, and arrive with enough time to spare for gossips before the reunion starts.

2. You could leave now on a different route and arrive just in time for the reunion.

Undoubtedly, if this is the only information available, most of us will opt for option 1.

If we now tell you that option 1 carries a fifty percent chance of missing a connection and be late for the reunion. Whereas, option 2 is almost guaranteed to take you there on time. Would you still consider option 1?

Probably not. However, most public transport applications will insist on the first option. This is because they are programmed to plan routes that offer the shortest travel times, without considering the risk factors.

----
## Problem Description

In this final project you will build your own _robust_ public transport route planner to improve on that. You will reuse the SBB dataset (See next section: [Dataset Description](#Dataset-Description)).

Given a desired **date** and **arrival time**, your route planner will compute the fastest route between departure and arrival stops within a provided confidence tolerance expressed as interquartiles.
For instance, "what route from _A_ to _B_ is the fastest at least _Q%_ of the time if I want to arrive at _B_ before instant _T_". Note that *confidence* is a measure of a route being feasible within the travel time computed by the algorithm.

The output of the algorithm is a list of routes between _A_ and _B_ and their confidence levels. The routes must be sorted from latest (fastest) to earliest (longest) departure time at _A_, they must all arrive at _B_ before _T_ with a confidence level greater than or equal to _Q_. Ideally, it should be possible to visualize the routes on a map.

In order to answer this question you will need to:

- Model the public transport infrastructure for your route planning algorithm using the data provided to you.
- Build a predictive model using the historical arrival/departure time data, and **at a minimum** one external sources of data (e.g. weather data).
- Implement a robust route planning algorithm using this predictive model.
- Test and **validate** your results. Note that we will put a particular emphasis on the scientific validation of your method.
- Implement a simple Jupyter-based visualization to demonstrate your method, using Jupyter widgets such as [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_guide.html).

Solving this problem accurately can be difficult. You are allowed a few **simplifying assumptions**:

- We only consider journeys at reasonable hours of the day, and on a typical business day, and assuming a recent schedule.
- We allow short walking distances for transfers between two stops, and assume a walking speed of _50m/1min_ on a straight line "As the Crows Flies", i.e. regardless of obstacles, human-built or natural, such as building, highways, rivers, or lakes.
- The total walking distance over the full trip should not exceed a maximum (default 500m, you can make it configurable). 
- In the assessment, we will only consider journeys that start and end on known station coordinates (train station, bus stops, etc.), never from a random location. However, walking from the departure stop to a nearby stop (or nearby stop to arrival stop) is allowed as long as the total walking distance is within the maximum.
- We only consider stops in an area that will be specified at a later time (the _uuid_ of [Geo spatial data](#Geo-spatial-data) should be configurable).
- If necessary, stops within the specified area can be reached through transfers at stops located outside the area. However, you do not need to determine the optimal set of external stops that minimizes travel time between stops within the area of interest.
- There is no penalty for assuming that delays or travel times on the public transport network of **two different lines** are uncorrelated with one another, **however** try to capture dependencies if possible, such as dependencies of delays on a same route;
 e.g. if a train is late at a stop, it is expected to be late at subsequent stops.
- Once a route is computed, a traveller is expected to follow the planned routes to the end, or until it fails (i.e. miss a connection).
  You **do not** need to address the case where travellers are able to defer their decisions and adapt their journey "en route", as more information becomes available. This would require us to consider all alternative routes (contingency plans) in the computation of the uncertainty levels, which is more difficult to implement.
- The planner does not need to account for the inconvenience to the traveler in the event of a plan failure. Two routes with the same travel time, within the specified uncertainty tolerance, are considered equivalent, even if the consequences of one route failing are significantly worse for the traveler (e.g., being stranded overnight), while the other route's failure would result in a less severe outcome.
- All other things being equal, you will prefer routes with the minimum walking distance and minimum number of transfers.
- You do not need to optimize the computation time of your method, as long as the run-time is reasonable.
- When computing a path you may pick the timetable of a recent week and assume that it is unchanged.

Upon request, and with clear instructions from you, we can help prepare the data in a form that is easier for you to process (within the limits of our ability, and time availability). In which case the data will be accessible to all.

----
## Project Submission Checklist

1. The assignment is a python code and/or well-annotated notebook; report-like.

1. All projects must be submitted to gitlab, as a group project (under your group name).

1. Provide instructions on how to test your project in the **HOW TO** section of the project's _README.md_ file. Include a link to your video presentation.

1. Project sizes, including history, must not exceed 100Mb. Use HDFS for your larger data sets.

----
## Video Presentations Checklist

1. Each member of the team must have something to say.

1. Use Zoom (or other tools) to record your group video.

1. Save the video as an mp4 file.

1. Upload your video to moodle under `Final assignment - video presentation`, or

1. If you choose to upload the video to an external platform, please include the link to the video in the **HOW TO** section at the top of the _README.md_ file in your final assignment.

Please, **DO NOT** save the video in your git project, **DO NOT** send a video embedded in a PowerPoint presentations, and **DO NOT** use any format other than mp4 videos. We must be able to stream the videos in our web browsers.

---- 
## Grading Method

After reviewing your videos, we will organize short Q&A discussions of 10 minutes per group. These discussions will be scheduled on the last week - actual day and times to be discussed on a case by case basis.

Think of yourselves as a startup trying to sell your solution to the board of a public transport company who has a good grasp of the state of the art. Your video is your elevator pitch. It must be short and convincing. In it you describe the viability
of the following aspects:

1. Data and data transformations used by your method
1. Method used to model the public transport network
1. Method used to create the predictive models
1. Route planning algorithm
1. Validation method

Your final grades will be based on the videos and Q&A, and implementation, taking into account:

1. Clarity and conciseness of the video presentation, code and Q&A
1. Team work, formulation and decomposition of the problem into smaller tasks between team members
1. Originality of the solution design, analytics, and presentation
1. Functional quality of the implementation (does it work?)
1. Explanation of the pro's and con's / shortcomings of the proposed solution

**DO NOT** underestimate the importance of your video presentations.

----
## Dataset Description

For this project we will use the data published on the [Open Data Platform Mobility Switzerland](<https://opentransportdata.swiss/en>).
For grading we will use recent SBB data limited around a city area of our choice (hold out data). We recommend that you develop a solution around Lausanne area and make sure that the solution will be able to work in a different or slightly larger area, and different period. We will select the area using a different region from the geo shapes table. The region _uuid_ from the table should therefore be a well documented parameter of your solution available as variable in the notebook. 

We will also provide you with a simulated realtime feed of Istdaten data.

For your convenience, we have defined the table for each of the data set described below in the database _iceberg.com490_iceberg_. You can list the tables with the command _SHOW TABLES IN iceberg.com490_iceberg_.  **DO NOT** drop the database, **DO NOT** drop the tables.

#### Actual (IstDaten) data

Students should already be familiar with the [istdaten](https://opentransportdata.swiss/en/cookbook/actual-data/)  data.
Several years of historical data are available in the _iceberg.com490_iceberg.sbb_istdaten_ Table.

See assignments and exercises of earlier weeks for more information about this data, and methods to access it.

* **sbb_istdaten**:

    - _operating_day_: day of operation when the journey took place. 		
    - _trip_id_: correspond to the journey ID
    - _operator_id_: operator code
    - _operator_abrv_: operator short name
    - _operator_name_: operator name
    - _product_id_: mode of transport (bus, tram, ...)
    - _line_id_: identify the line
    - _line_text_: name of the line, e.g. 701
    - _circuit_id_: schedule and route of a given vehicle
    - _transport_: description of the type of transport
    - _unplanned_: if true this is an additional (unplanned) journey
    - _failed_: if true, the journey failed (didn't complete)
    - _bpuic_: stop BPUIC
    - _stop_name_: stop name (until mid 2021 you could also find the BPUIC there)
    - _arr_time_: expected arrival time
    - _arr_actual_: actual arrival time
    - _arr_status_: method used to compute actual arrival time
    - _dep_time_: expected departure time
    - _dep_actual_: actual departure time
    - _dep_status_: method used to compute actual departure time
    - _transit_: if true, vehicle does not stop at the location

#### Timetable data

Timetable data are available from opentransportdata.swiss's [timetable](https://opentransportdata.swiss/en/cookbook/gtfs/) data set.

The timetables are updated weekly (_pub_date_). It is ok to assume that the changes are small, and you can use the schedule of the most recent week for the day of the trip. However, note that public transport services may run at different times or not at all depending on the day of the week.

The full description of the GTFS format is available in the opentransportdata.swiss data [timetable cookbooks](https://opentransportdata.swiss/en/cookbook/gtfs/).

We provide a summary description of the data below:

* **sbb_stops**: ([doc](https://opentransportdata.swiss/en/cookbook/gtfs/#stopstxt))

    - _pub_date_: date of publication (weekly)
    - _stop_id_: unique identifier (PK) of the stop, it is the BPUIC with platform information
    - _stop_name_: long name of the stop
    - _stop_lat_: stop latitude (WGS84)
    - _stop_lon_: stop longitude
    - _location_type_:
    - _parent_station_: if the stop is one of many collocated at a same location, such as platforms at a train station

* **sbb_stop_times**:

    - _pub_date_: date of publication (weekly)
    - _trip_id_: identifier (FK) of the trip, unique for the day - e.g. _1.TA.1-100-j19-1.1.H_
    - _arrival_time_: scheduled (local) time of arrival at the stop (same as DEPARTURE_TIME if this is the start of the journey)
    - _departure_time_: scheduled (local) time of departure at the stop 
    - _stop_id_: stop (station) BPUIC identifier (FK), from stops.txt
    - _stop_sequence_: sequence number of the stop on this trip id, starting at 1.
    - _pickup_type_:
    - _drop_off_type_:

* **sbb_trips**: ([doc](https://opentransportdata.swiss/en/cookbook/gtfs/#tripstxt))

    - _pub_date_: date of publication (weekly)
    - _route_id_: identifier (FK) for the route. A route is a sequence of stops. It is time independent.
    - _service_id_: identifier (FK) of a group of trips in the calendar, and for managing exceptions (e.g. holidays, etc).
    - _trip_id_: is one instance (PK) of a vehicle journey on a given route - the same route can have many trips at regular intervals; a trip may skip some of the route stops.
    - _trip_headsign_: displayed to passengers, most of the time this is the (short) name of the last stop.
    - _trip_short_name_: internal identifier for the trip_headsign (note TRIP_HEADSIGN and TRIP_SHORT_NAME are only unique for an agency)
    - _direction_id_: if the route is bidirectional, this field indicates the direction of the trip on the route.
    
* **sbb_calendar**:

    - _pub_date_: date of publication (weekly)
    - _service_id_: identifier (PK) of a group of trips sharing a same calendar and calendar exception pattern.
    - _monday_.._sunday_: FALSE (0) or TRUE (1) for each day of the week, indicating occurence of the service on that day.
    - _start_date_: start date when weekly service id pattern is valid
    - _end_date_: end date after which weekly service id pattern is no longer valid
    
* **sbb_routes**: ([doc](https://opentransportdata.swiss/en/cookbook/gtfs/#routestxt))

    - _route_id_: identifier for the route (PK)
    - _agency_id_: identifier of the operator (FK)
    - _route_short_name_: the short name of the route, usually a line number
    - _route_long_name_: (empty)
    - _route_desc_: _Bus_, _Zub_, _Tram_, etc.
    - _route_type_:
    
**Note:** PK=Primary Key (unique), FK=Foreign Key (refers to a Primary Key in another table)

The other tables are:

* _sbb_calendar_dates_: contains exceptions to the weekly patterns expressed in _sbb_calendar_.
* _sbb_transfers_: contains the transfer times between stops or platforms.
* _agency_: operator information, under HDFS _/data/com-490/csv/sbb/agency_

Figure 1. better illustrates the above concepts relating stops, routes, trips and stop times on a real example (route _11-3-A-j19-1_, direction _0_)

 ![journeys](figs/journeys.svg)
 
 _Figure 1._ Relation between stops, routes, trips and stop times. The vertical axis represents the stops along the route in the direction of travel.
             The horizontal axis represents the time of day on a non-linear scale. Solid lines connecting the stops correspond to trips.
             A trip is one instances of a vehicle journey on the route. Trips on same route do not need
             to mark all the stops on the route, resulting in trips having different stop lists for the same route.

#### Geo spatial data

You will find useful geospatial shapes in the table _iceberg.com490_iceberg.geo_`.

* **geo**:
    * _uuid_: unique identifier of the shape (swiss topo)
    * _name_: a human readable name of the shape
    * _country_: country containing the shape, e.g. _CH_
    * _region_: region or canton, e.g. _VD_
    * _level_: shape class level, e.g. _city_, _canton_, ...
    * _wkb_geometry_: binary (WKB) representation of the shape that can be used in Geospatial UDF functions.

#### Misc data

We ask to use another source of data.

**Weather**:

Some had some success using weather data to predict traffic delays.

If you want to give a try, web services such as [wunderground](https://www.wunderground.com/history/daily/ch/r%C3%BCmlang/LSZH/date/2022-1-1), can be a good
source of historical weather data. For your convenience we have made available to you a copy of the last few years of weather data reported at various airports in Swizerland.

You can find the data on HDFS under _/data/com-490/json/weather_history/_ and _/data/com-490/json/weather_stations/_.

**Others**:

You are of course free to use any other sources of data of your choice that might find helpful.

You may for instance download regions of openstreetmap [OSM](https://www.openstreetmap.org/#map=9/47.2839/8.1271&layers=TN),
which includes a public transport layer. If the planet OSM is too large for you,
you can find frequently updated exports of the [Swiss OSM region](https://planet.osm.ch/).

----
## Hints

Before you get started, we offer a few hints:

- Reserve some time to Google-up the state of the art before implementing. There is a substantial amount of work on this topic. Look for *time-dependent*, or *time-varying networks*, and *stochastic route planning under uncertainty*.
- You should already be acquainted with the data.
However, as you learn more about the state of the art, spend time to better understand your data.
Anticipate what can and cannot be done from what is available to you, and plan your design strategy accordingly. Do not hesitate to complete the proposed data sources with your own if necessary.
- Start small with a simple working solution and improve on it.
In a first version, assume that all trains and buses are always sharp on time.
Focus on creating a sane collaborative environment that you can use to develop and test your work in team as it evolves.
Next, work-out the risk-aware solution gradually - start with a simple predictive model and improve it. In addition you can test your algorithm on selected pairs of stops before generalizing to the full public transport network under consideration.

----
## FAQ

This section will be updated with the Frequently Asked Questions during the course of this project. Please stay tuned.

##### 1 - Q: Do we need to take into account walking times at the connections?
* **A**: Yes, but since we do not have the details of the platforms at each location, we can use a universal formula to come up with a reasonable walking time.
We must also allow time for transfers between different modes of transports, such as from bus to tramways.
You can use the transfer time information available from `transfers.txt` from the [timetables](#Timetable-data).
Otherwise, we assume that `2min` mininum are required for transfers within a same location
(i.e. same lat,lon coordinates), to which you add _1min per 50m_ walking time
to connect two stops that are at most _500m_ appart, on a straight line distance between their two lat,lon. 

##### 2 - Q: Can we assume statistical independence between the observed delays?
* **A**: See simplifying assumptions in **Problem Description**.
You will incur no penalty for assuming that the delay of a given train (or other mode of transport, ...), at a given location and time is
independent of the delays for all other trains, locations, and times. Even if our experience tells us that this is most of the time not the case.
Also, you must assume that you have no real-time delays information at the time you plan your journey, which limits the benefits you could gain by assuming such a dependency.

##### 3 - Q: Can I take advantage of the fact that a connection departs late most of the time to allow a plan that would otherwise not be possible according to the official schedule.
* **A**: You may discover that you could take advantage of connections that have a high probability of departing late.
However, this is not recommended, or it should come with a warning.
Imagine from a user experience perspective, how would you react if you are being proposed an impossible plan in which a transfer is scheduled to depart before you arrive?
Furthermore, who would you blame if the plan fails: the planner that came up with a theoretically infeasible plan, or the operator who respected their schedule?

----

```python

```
