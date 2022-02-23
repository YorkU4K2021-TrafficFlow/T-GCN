"""
Taken and was adjusted from DCRNN [https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py]
    @inproceedings{li2018dcrnn_traffic,
      title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
      author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
      booktitle={International Conference on Learning Representations (ICLR '18)},
      year={2018}
    }

This file creates an adjacency matrix for the different nodes/ intersections/ section of the graph,
and saves it at dataset/adj_mx_intersections.csv or dataset/adj_mx_sections.csv (depending on INTERSECTION
parameter).

The INTERSECTION parameter specifies whether the input relates to intersections or sections (differentiates between
2 datasets).

The output file is a NxN adjacency matrix, where N is the number of nodes. The inner values represent how close the
nodes are to each other.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd


INTERSECTION = True


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    print(sensor_id_to_ind)

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue

        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        print("\t",row[2])
    print()
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    if INTERSECTION:
        sensors = pd.read_csv('../dataset/intersections.csv')
        sensor_ids = [i for i in range(len(sensors))]
        distance_df = pd.read_csv('../dataset/shortest_dist.csv', dtype={'from': 'int', 'to': 'int'})
    else:
        sensors = pd.read_csv('../dataset/sections.csv')
        sensor_ids = [i for i in range(len(sensors))]
        distance_df = pd.read_csv('../dataset/shortest_dist_sections.csv', dtype={'from': 'int', 'to': 'int'})

    normalized_k = 0.1

    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)

    # Save file

    adj_mx = pd.DataFrame(adj_mx)
    print("adj_mx:\n"+str(adj_mx)+"\n")
    print("adj_mx:\n"+str(len(adj_mx))+"\n")
    print("adj_mx:\n"+str(len(adj_mx[0]))+"\n")

    if INTERSECTION:
        adj_mx.to_csv('../dataset/adj_mx_intersections.csv', index=False, header=None)
    else:
        adj_mx.to_csv('../dataset/adj_mx_sections.csv', index=False, header=None)
