"""
This file constructs an average speed over time for each sensor configuratino listed,
and saves it in speed_over_time.csv.
Each column in speed_over_time_[x]_sensors.csv represents an sensor row in [x]_sensors.csv:
    column 0 in speed_over_time_[x]_sensors.csv corresponds to row 0 in [x]_sensors.csv

Each row represents the average speed of that intersection at a given point in time.
"""
import itertools

import pandas as pd
import numpy as np
import time

# ------------------------------ PARAMETERS ------------------------------
SIM = ['0', '1', '2', '3', '4', '5', '6', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24']
SIM_DIR = '/Volumes/LaCie/csv/24_hour_'
SENSORS_DIR = '../dataset/sensors/csv/'
WITHIN_X_KM = 0.05  # distance to associate a car to a sensor
TIMESTAMP = 1  # what 1 timestamp accounts for in seconds
SEQ_LEN = 5  # aggregate timestamps into what sequence length size (5 minutes, 10 minutes ...)
SENSORS = ['30_sensors', '60_sensors', '90_sensors', '120_sensors']
DIR = '/Volumes/LaCie/csv/'
OUT_NAME = 'spd_over_time/speed_over_time'
IS_DIRECTIONAL = False
DELETE_COLS = False
N1 = [0, 180]  # [0-180)               ->     even index
N2 = [180, 360]  # [180-360)           ->     odd index
# -----------------------------------------------------------------------


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
    cons = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))

    return R * cons


if DELETE_COLS:
    # remove unnecessary columns to save space
    for i in ['vehicle_id', 'vehicle_lane', 'vehicle_pos', 'vehicle_slope', 'vehicle_type']:
        drop_col(i)

    sim.to_csv('../dataset/' + SIM, index=False)

start = time.time()
for sensor in SENSORS:
    for DATA_id in SIM:
        sim = pd.read_csv(SIM_DIR + DATA_id + '.csv')
        sensors = pd.read_csv(SENSORS_DIR + sensor + '.csv')

        # create the average speed table for each sensor/intersection at different timestamps
        # during a simulation
        table = []
        speeds = np.zeros(len(sensors) * 2)  # avg speed per sensor within 5 minutes
        counts = np.zeros(len(sensors) * 2)  # counts number of detected cars within 5 minutes

        for name, group in sim.groupby('timestamp'):
            # For each new timestamp, get the average speed of each intersection(\sensor)
            timestamp = name
            ys = group['y'].values
            xs = group['x'].values
            spd = group['speed'].values
            angle = group['angle'].values

            avg_speed_res = []

            for i, row in sensors.iterrows():
                # For each sensor at place i, get the sum of speeds within WITHIN_X_KM,
                # and number of instances. Average the speed after SEQ_LEN minutes
                long = row['longitude']
                lat = row['latitude']
                haversine_distances = getDistance(lat, long, latitudes=ys, longitudes=xs)

                # look for intersections that are within 50 meters
                indexes = np.where(haversine_distances < WITHIN_X_KM)[0]
                even_ind = np.where(np.logical_and(angle[indexes] >= N1[0], angle[indexes] < N1[1]))[0]
                odd_ind = np.setdiff1d(indexes, indexes[even_ind])

                # speeds - even
                speeds[i * 2] += np.sum(spd[indexes[even_ind]].tolist())
                counts[i * 2] += len(spd[indexes[even_ind]])
                # speeds - odd
                speeds[i * 2 + 1] += np.sum(spd[odd_ind].tolist())
                counts[i * 2 + 1] += len(spd[odd_ind])

            # Each timestamp is 1 sec. Want to average over 5 minutes (300 timestamps)
            if (timestamp + 1) % (60 * SEQ_LEN / TIMESTAMP) == 0:
                avg_speed_res = np.array(speeds) / np.array(counts)
                avg_speed_res = np.nan_to_num(avg_speed_res)
                table.append(avg_speed_res)

                speeds = np.zeros(len(sensors) * 2)  # avg speed per sensor within 5 minutes
                counts = np.zeros(len(sensors) * 2)  # counts number of detected cars within 5 minutes

        print('--> About to save to csv')
        table = pd.DataFrame(table)
        table.to_csv(DIR + OUT_NAME + '_' + sensor + '.csv', mode='a', header=False, index=False)
        print('--> Done for ' + str(DATA_id))

end  = time.time()
print(str((end-start)/60)+' min')