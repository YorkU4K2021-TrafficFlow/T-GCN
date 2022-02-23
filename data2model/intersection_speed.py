"""
This file constructs an average speed over time for each intersection listed
in intersections.csv, and saves it in speed_over_time.csv.
Each column in speed_over_time.csv represents an intersection row in intersections.csv:
    column 0 in speed_over_time.csv corresponds to row 0 in intersections.csv

Each row represents the average speed of that intersection at a given point in time.
"""

import pandas as pd
import numpy as np


sim = pd.read_csv('../dataset/geo2csv.csv')
# intersections = pd.read_csv('../dataset/sections.csv.csv')
FNAME = '../dataset/speed_over_time_sections.csv'

intersections = pd.read_csv('../dataset/intersections.csv')
FNAME = '../dataset/speed_over_time.csv'


def drop_col(str_val):
    global sim

    try:
        sim = sim.drop([str_val], axis=1)
    except Exception as e:
        pass


def getDistance(latitude1, longitude1, latitudes, longitudes):
    #  Source:   https://www.geeksforgeeks.org/program-distance-two-points-earth/#:~:text=For%20this%20divide%20the%20values,is%20the%20radius%20of%20Earth.
    R = 6378.1  # earth's radius in km
    lat1 = np.deg2rad(latitude1)
    lon1 = np.deg2rad(longitude1)
    lat2 = np.deg2rad(latitudes)
    lon2 = np.deg2rad(longitudes)

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    d = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    # cons = 2 * np.arcsin(np.sqrt(d))
    cons = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return R * cons


# remove unnecessary columns to save space
for i in ['vehicle_angle', 'vehicle_id','vehicle_lane','vehicle_pos', 'vehicle_slope','vehicle_type']:
    drop_col(i)

sim.to_csv('../dataset/geo2csv.csv',index=False)

# create the average speed table for each sensor/intersection at different timestamps
# during a simulation
table = []

for name, group in sim.groupby('timestep_time'):
    # For each new timestamp, get the average speed of each intersection(\sensor)
    timestamp = name
    ys = group['vehicle_y'].values
    xs = group['vehicle_x'].values
    spd = group['vehicle_speed'].values

    avg_speed_res = []

    for i, row in intersections.iterrows():
        long = row['longitude']
        lat = row['latitude']

        haversine_distances = getDistance(lat, long,
                                          latitudes=ys,
                                          longitudes=xs)

        # look for intersections that are within 50 meters
        indexes = np.where(haversine_distances < 0.01)[0]

        # speeds
        speeds = spd[indexes]

        if len(speeds) > 0:
            avg_spd = np.mean(speeds)
        else:
            avg_spd = 0.0

        avg_speed_res.append(avg_spd)
    table.append(avg_speed_res)


table = pd.DataFrame(table)
table.to_csv(FNAME, index=False)