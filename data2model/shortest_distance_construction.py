"""
This file constructs the cost (distance of shortest path) between sensor a and sensor b.
Since the path from a->b and b->a is not necessarily the same, the time complexity is Θ(n^2),
where n is the number of intersections.

The program saves the result in shortest_dist.csv.

The columns represent source intersection, destination intersection, and cost.
"""

import pandas as pd
import urllib.request
import json


# Get paths from OSRM for a given source `a` and destination `b`
def getDistance(a, b):
    q = 'http://router.project-osrm.org/route/v1/car/' + str(a[1]) + ',' + str(a[0]) + \
        ';' + str(b[1]) + ',' + str(b[0]) + '?alternatives=true&geometries=geojson&overview=full'
    print(q)
    req = urllib.request.Request(q)
    with urllib.request.urlopen(req) as response:
        routing_data = json.loads(response.read())

    dist = 0

    for i in range(len(routing_data['routes'])):
        dist += (round(routing_data['routes'][i]['distance'] / 1000.0, 5))    # m -> Km

    return dist


# intersections = pd.read_csv('../dataset/intersections.csv', header=0)
intersections = pd.read_csv('../dataset/sections.csv', header=0)
from_vals = []
to_vals = []
cost_vals = []

id_1 = 0

for id_1, row1 in intersections.iterrows():
    for id_2, row2 in intersections.iterrows():
        a = [row1['latitude'], row1['longitude']]
        b = [row2['latitude'], row2['longitude']]

        dist = getDistance(a, b)
        from_vals.append(id_1)
        to_vals.append(id_2)
        cost_vals.append(dist)

print(cost_vals)
print(from_vals)
print(to_vals)
print(len(cost_vals))

df = pd.DataFrame(columns=['from', 'to', 'cost'])
df['from'] = from_vals
df['to'] = to_vals
df['cost'] = cost_vals

# df.to_csv('../dataset/shortest_dist.csv',index=False)
df.to_csv('../dataset/shortest_dist_sections.csv',index=False)