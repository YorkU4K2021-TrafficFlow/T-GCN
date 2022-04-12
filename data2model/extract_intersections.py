"""
This file extracts the coordinates of intersections from a file,
and saves it in a csv file (intersections.csv)

The columns are longitude and latiude;
the rows are the intersections.

If SUMO parameter is set to True, it will process an xml file derived from SUMO;
otherwise, it will process a text file(html formal) derived from intersection_selector.html
"""

import numpy as np

import pandas as pd

SUMO = False

if SUMO:
    import xml.dom.minidom as xm

    file = xm.parse('../dataset/intersections.xml')
    print(file.nodeName)
    nodes = file.getElementsByTagName('node')
    print(nodes.length)

    longitude = []
    latitude = []

    for node in nodes:
        longitude.append(node.getAttribute('x'))
        latitude.append(node.getAttribute('y'))

    df = pd.DataFrame(columns=['latitude', 'longitude'])
    df['longitude'] = longitude
    df['latitude'] = latitude

    df = df.drop_duplicates()

    print(df.head(10))
    print(df.dtypes)
    df.to_csv('../dataset/intersections.csv',index=False)
else:
    import re
    import ast

    FNAMES = ['30_sensors','50_sensors','60_sensors','90_sensors','100_sensors','120_sensors']
    DIR = '../dataset/sensors'

    for fname in FNAMES:
        file = open(DIR +'/'+ fname+'.txt', 'r')

        content = file.read()
        coordinates = re.findall('\[.*?]',content)
        file.close()
        coordinates = [ast.literal_eval(coordinate) for coordinate in coordinates]

        coordinates = np.array(coordinates)
        coordinates = coordinates[:len(coordinates)-1]

        df = pd.DataFrame(columns=['latitude', 'longitude'])
        df['longitude'] = coordinates[:,1]
        df['latitude'] = coordinates[:,0]

        df = df.drop_duplicates()

        print(df.head(10))
        print(df.dtypes)
        df.to_csv(DIR + '/csv/' + fname + '.csv',index=False)
