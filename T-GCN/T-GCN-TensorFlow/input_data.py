# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'../../data/sz_adj.csv',header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'../../data/sz_speed.csv')
    return sz_tf, adj


def load_los_data(dataset):
    los_adj = pd.read_csv(r'../../data/los_adj.csv',header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'../../data/los_speed.csv')
    return los_tf, adj


def load_our_data_intersections(dataset):
    adj = pd.read_csv(r'../../dataset/adj_mx_intersections.csv',header=None)
    adj = np.mat(adj)
    tf = pd.read_csv(r'../../dataset/speed_over_time.csv')
    return tf, adj


def load_our_data_sections(dataset):
    adj = pd.read_csv(r'../../dataset/adj_mx_sections.csv',header=None)
    adj = np.mat(adj)
    tf = pd.read_csv(r'../../dataset/speed_over_time_sections.csv')
    return tf, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    # trains from 0 to length - seq_len - pre_len
    #       (seq_len represents 1 hour, pre_len represents the prediction time(15/30/45/60)
    # ex: if seq_len(12) + pre_len(3) = 15:   (75 minutes in case of los_loop)
    #           a = train_data[0:15], [1:16], [2:17], ... , [len-15:len]
    #           X =           [0:12], [1:13], [2: 14], ..., [len-15:len-3] // (first 1 hour)
    #           Y =           [12:15],[13:16],[14:17], ..., [len-3:len]    // (last 15 minutes)
    # for every mini-batch, it looks into the label (Y)
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
    
