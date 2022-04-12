"""
The goal of this script is to remove the repetition in the model output file.
"""

import pandas as pd
import tensorflow as tf
from input_data import preprocess_data, load_sz_data, load_los_data, load_our_data, load_our_data_5_days
import numpy as np
from visualization import plot_custom_result, plot_result
import os

# ------------------------------ PROPOSED HYPER-PARAMETERS ------------------------------
GRU_UNITS = [8, 16, 32, 64, 100]#, 128]  # following the paper suggested values
MAX_EPOCH = 3000  # maximun epoch value following the paper
SAVE_AFTER = 500  # analyze units at each 1000'th epoch mark (aka: 1000, 2000, ...)
BATCH_SIZE = 32  # generally, 32 is sufficient
NUM_SENSORS = ['30_sensors', '60_sensors', '90_sensors']#, '120_sensors']
PRE_LENS = [3]#, 6, 9, 12]
IS_5_DAYS = True

# ---------------------------------------------------------------------------------------


def reformat(pred, ground, PRE_LEN):
    res_pred = []
    res_true = []

    for i in range(len(pred)):
        if i % PRE_LEN == 0:
            res_pred.append(pred[i])
            res_true.append(ground[i])

    for i in pred[((len(pred)-1) % PRE_LEN)*-1:]:
        res_pred.append(i)

    for i in ground[((len(ground)-1) % PRE_LEN)*-1:]:
        res_true.append(i)

    res_pred = np.array(res_pred)
    res_true = np.array(res_true)
    return res_pred, res_true


for PRE_LEN in PRE_LENS:
    for num_sensors in NUM_SENSORS:
        for unit_index in range(len(GRU_UNITS)):
            # ------------ Settings ------------
            flags = tf.app.flags
            FLAGS = flags.FLAGS
            flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
            flags.DEFINE_integer('training_epoch', MAX_EPOCH, 'Number of epochs to train.')
            flags.DEFINE_integer('gru_units', GRU_UNITS[unit_index], 'hidden units of gru.')  # sz_taxi: 100, lop_loop: 64
            flags.DEFINE_integer('seq_len', 12,
                                 'time length of inputs.')  # sz_taxi: 4, los_loop: 12  --> represents 1 hour of information
            flags.DEFINE_integer('pre_len', PRE_LEN,
                                 'time length of prediction.')  # sz_taxi: 1/2/3/4, los_loop: 3/6/9/12 for [15/30/45/60] min respectively  --> represents 15/20/45/60 min of information
            flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
            flags.DEFINE_integer('batch_size', BATCH_SIZE, 'batch size.')
            flags.DEFINE_string('dataset', num_sensors,
                                'sz or los or intersections or sections or 30_sensors or 50_sensors.')
            flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
            model_name = FLAGS.model_name
            data_name = FLAGS.dataset
            train_rate = FLAGS.train_rate
            seq_len = FLAGS.seq_len
            output_dim = pre_len = FLAGS.pre_len
            batch_size = FLAGS.batch_size
            lr = FLAGS.learning_rate
            training_epoch = FLAGS.training_epoch
            gru_units = FLAGS.gru_units

            ###### load data ######
            if data_name == 'sz':
                data, adj = load_sz_data('sz')
            elif data_name == 'los':
                data, adj = load_los_data('los')
            elif IS_5_DAYS:
                data, adj, prediction = load_our_data_5_days(data_name)
            else:
                data, adj, prediction = load_our_data(data_name)

            time_len = data.shape[0]
            num_nodes = data.shape[1]
            data1 = np.mat(data, dtype=np.float32)

            #### normalization
            max_value = np.max(data1)
            data1 = data1 / max_value
            trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

            test_label = np.reshape(testY, [-1, num_nodes])
            test_label1 = test_label * max_value
            prediction = prediction.to_numpy()

            # plot original result
            if IS_5_DAYS:
                path = '../../data2model/hyperparam_tuning_5_days/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)
            else:
                path = '../../data2model/hyperparam_tuning/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)

            if not os.path.exists(path):
                os.makedirs(path)

            plot_result(prediction, test_label1, path)

            prediction, test_label1 = reformat(prediction, test_label1, PRE_LEN)

            # save compressed version
            if IS_5_DAYS:
                path = '../../data2model/hyperparam_tuning_5_days/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)
            else:
                path = '../../data2model/hyperparam_tuning/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)

            if not os.path.exists(path):
                os.makedirs(path)

            var = pd.DataFrame(prediction)
            var.to_csv(path + '/test_result-compressed.csv', index=False, header=False)

            # plot compressed result
            if IS_5_DAYS:
                path = '../../data2model/hyperparam_tuning_5_days/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)
            else:
                path = '../../data2model/hyperparam_tuning/%s/%s_epochs/%s_units/%s_pre' % (num_sensors, MAX_EPOCH, gru_units, pre_len)

            if not os.path.exists(path):
                os.makedirs(path)
            plot_custom_result(prediction, test_label1, path)

            flags_dict = FLAGS._flags()
            keys_list = [keys for keys in flags_dict]
            for keys in keys_list:
                FLAGS.__delattr__(keys)
            tf.reset_default_graph()