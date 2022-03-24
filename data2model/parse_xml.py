"""
This file parses through the 24 hour simulation.
Sine the simulation file is too big, it had to be loaded to memory in chunks.
The output files had to be divided into multiple files due to the size of the output.
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as etree

BUF_SIZE = 300*200
NUM_HOURS = 1
timestamps = []
x = []
y = []
angle = []
speed = []

t = 0
counter = 0

with open('/Volumes/LaCie/24hour_mar_09','r') as f:
    lines = f.readlines(BUF_SIZE)
    while lines:
        for line in lines:
            if '<timestep time=' in line:
                t = int(float(line.split('"')[1]))
                print('\t==>'+str(t))

                if t % (NUM_HOURS * 3600) == 0 and t != 0:
                    df = pd.DataFrame(columns=['timestamp','x','y','angle','speed'])
                    df['timestamp'] = timestamps
                    df['x'] = x
                    df['y'] = y
                    df['angle'] = angle
                    df['speed'] = speed

                    print('---------------------')
                    print(" num elements = " + str(len(x)))
                    df.to_csv('/Volumes/LaCie/csv/24_hour_'+str(counter)+'.csv',index=False)

                    print('Done uploading /Volumes/LaCie/csv/24_hour_'+str(counter)+'.csv !!!')
                    print('---------------------')

                    x, y, angle, speed, timestamps = [], [], [], [], []
                    counter += 1

            elif '<vehicle id=' in line:
                tmp = line.split('"')
                x.append(tmp[3])
                y.append(tmp[5])
                angle.append(tmp[7])
                speed.append(tmp[11])
                timestamps.append(t)
        lines = f.readlines(BUF_SIZE)

if len(x) > 0:
    df = pd.DataFrame(columns=['timestamp','x','y','angle','speed'])
    df['timestamp'] = timestamps
    df['x'] = x
    df['y'] = y
    df['angle'] = angle
    df['speed'] = speed
    df.to_csv('/Volumes/LaCie/csv/24_hour_'+str(counter+1)+'.csv',index=False)
