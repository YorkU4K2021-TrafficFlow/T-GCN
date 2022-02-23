# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_sz_data,load_los_data, load_our_data_sections,  load_our_data_intersections
from tgcn import tgcnCell

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import time

time_start = time.time()

# ------------------------------ PROPOSED HYPER-PARAMETERS ------------------------------
GRU_UNITS = [8, 16, 32, 64, 100, 128]  # following the paper suggested values
MAX_EPOCH = 10                       # maximun epoch valu following the paper
SAVE_AFTER = 5                      # analyze units at each 1000'th epoch mark (aka: 1000, 2000, ...)
BATCH_SIZE = 32                        # generally, 32 is sufficient

# --------------------------------------- Results ---------------------------------------
RMSE = np.array([[0]*int(MAX_EPOCH/SAVE_AFTER)]*len(GRU_UNITS))
MAE = np.array([[0]*int(MAX_EPOCH/SAVE_AFTER)]*len(GRU_UNITS))
ACCURACY = np.array([[0]*int(MAX_EPOCH/SAVE_AFTER)]*len(GRU_UNITS))
R2 = np.array([[0]*int(MAX_EPOCH/SAVE_AFTER)]*len(GRU_UNITS))
VAR = np.array([[0]*int(MAX_EPOCH/SAVE_AFTER)]*len(GRU_UNITS))

# ---------------------------------------------------------------------------------------


def TGCN(_X, _weights, _biases):
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states


###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var


def save_and_visualize(gru_units, pre_len, training_epoch, batch_rmse, totalbatch, test_rmse, test_pred, test_label1,
                       path,test_acc,test_mae, test_r2, test_var, time_start, batch_loss):
    global RMSE, MAE, ACCURACY, R2, VAR
    time_end = time.time()
    print(time_end-time_start,'s')
    print('\n')
    print("results for: gru_units="+str(gru_units)+", pre_len="+str(pre_len)+", and epochs="+str(training_epoch))
    print('\n')

    ############## visualization ###############
    b = int(len(batch_rmse)/totalbatch)
    batch_rmse1 = [i for i in batch_rmse]
    train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
    batch_loss1 = [i for i in batch_loss]
    train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

    index = test_rmse.index(np.min(test_rmse))
    test_result = test_pred[index]
    var = pd.DataFrame(test_result)
    var.to_csv(path+'/test_result.csv',index = False,header = False)
    plot_result(test_result,test_label1,path)
    plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

    print('min_rmse:%r'%(np.min(test_rmse)),
          'min_mae:%r'%(test_mae[index]),
          'max_acc:%r'%(test_acc[index]),
          'r2:%r'%(test_r2[index]),
          'var:%r'%(test_var[index]))

    te = int(training_epoch/SAVE_AFTER)-1
    RMSE[gru_units][te] = np.min(test_rmse)
    MAE[gru_units][te] = test_mae[index]
    ACCURACY[gru_units][te] = test_acc[index]
    R2[gru_units][te] = test_r2[index]
    VAR[gru_units][te] = test_var[index]


def vis_results(path):
    global RMSE, MAE, ACCURACY, R2, VAR, GRU_UNITS, MAX_EPOCH, SAVE_AFTER

    for i in range(int(MAX_EPOCH/SAVE_AFTER)):
        x_axis = [x for x in range(len(GRU_UNITS))]
        plt.plot(x_axis, RMSE[:,i], color='black', marker='s', label='RMSE')
        plt.plot(x_axis, MAE[:,i], color='red', marker='s', label='MAE')
        plt.xlabel("Hidden Units")
        plt.title("RMSE and MAE over hidden layers - "+str((i+1)*SAVE_AFTER) + ' epochs')
        plt.grid(axis = 'y')
        plt.legend()
        plt.xticks(x_axis, GRU_UNITS)
        plt.savefig(path+'/MAE_RMSE_'+str((i+1)*SAVE_AFTER)+'_epochs.png')
        plt.show()

        plt.clf()

        plt.plot(x_axis, ACCURACY[:,i], color='black', marker='s', label='Accuracy')
        plt.plot(x_axis, R2[:,i], color='red', marker='s', label='R2')
        plt.plot(x_axis, VAR[:,i], color='green', marker='s', label='Var')
        plt.xlabel("Hidden Units")
        plt.title("Accuracy, R2 and MAE over hidden layers - "+str((i+1)*SAVE_AFTER) +' epochs')
        plt.grid(axis = 'y')
        plt.legend()
        plt.xticks(x_axis, GRU_UNITS)
        plt.savefig(path+'/Accuracy_R2_Var_'+str((i+1)*SAVE_AFTER)+'_epochs.png')
        plt.show()

        plt.clf()


for unit_index in range(len(GRU_UNITS)):
    # ------------ Settings ------------
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('training_epoch', MAX_EPOCH, 'Number of epochs to train.')
    flags.DEFINE_integer('gru_units', GRU_UNITS[unit_index], 'hidden units of gru.')   # sz_taxi: 100, lop_loop: 64
    flags.DEFINE_integer('seq_len', 1, 'time length of inputs.')   # sz_taxi: 4, los_loop: 12  --> represents 1 hour of information
    flags.DEFINE_integer('pre_len', 15, 'time length of prediction.')   # sz_taxi: 1/2/3/4, los_loop: 3/6/9/12 for [15/30/45/60] min respectively  --> represents 15/20/45/60 min of information
    flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'batch size.')
    flags.DEFINE_string('dataset', 'sections', 'sz or los or intersections or sections.')
    flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
    model_name = FLAGS.model_name
    data_name = FLAGS.dataset
    train_rate =  FLAGS.train_rate
    seq_len = FLAGS.seq_len
    output_dim = pre_len = FLAGS.pre_len
    batch_size = FLAGS.batch_size
    lr = FLAGS.learning_rate
    training_epoch = FLAGS.training_epoch
    gru_units = FLAGS.gru_units

    ###### load data ######
    if data_name == 'sz':
        data, adj = load_sz_data('sz')
    if data_name == 'los':
        data, adj = load_los_data('los')
    if data_name == 'intersections':
        data, adj = load_our_data_intersections('intersections')
    if data_name == 'sections':
        data, adj = load_our_data_sections('sections')

    time_len = data.shape[0]
    num_nodes = data.shape[1]
    data1 =np.mat(data,dtype=np.float32)

    #### normalization
    max_value = np.max(data1)
    data1  = data1/max_value
    trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

    totalbatch = int(trainX.shape[0]/batch_size)
    training_data_count = len(trainX)

    ###### placeholders ######
    inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
    labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

    # Graph weights
    weights = {
        'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
    biases = {
        'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}

    if model_name == 'tgcn':
        pred,ttts,ttto = TGCN(inputs, weights, biases)

    y_pred = pred

    ###### optimizer ######
    lambda_loss = 0.0015
    Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    label = tf.reshape(labels, [-1,num_nodes])
    ##loss
    loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
    ##rmse
    error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    ###### Initialize session ######
    variables = tf.global_variables()
    saver = tf.train.Saver(tf.global_variables())
    #sess = tf.Session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    out = '../../data2model/hyperparam_tuning/%s'%(model_name)
    #out = 'out/%s_%s'%(model_name,'perturbation')
    path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
    path = os.path.join(out,path1)
    if not os.path.exists(path):
        os.makedirs(path)

    x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
    test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]

    for epoch in range(training_epoch):
        for m in range(totalbatch):
            mini_batch = trainX[m * batch_size : (m+1) * batch_size]
            mini_label = trainY[m * batch_size : (m+1) * batch_size]
            _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                     feed_dict = {inputs:mini_batch, labels:mini_label})
            batch_loss.append(loss1)
            batch_rmse.append(rmse1 * max_value)

        # Test completely at every epoch
        loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                             feed_dict = {inputs:testX, labels:testY})
        test_label = np.reshape(testY,[-1,num_nodes])
        rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
        test_label1 = test_label * max_value
        test_output1 = test_output * max_value
        test_loss.append(loss2)
        test_rmse.append(rmse * max_value)
        test_mae.append(mae * max_value)
        test_acc.append(acc)
        test_r2.append(r2_score)
        test_var.append(var_score)
        test_pred.append(test_output1)

        print('Iter:{}'.format(epoch),
              'train_rmse:{:.4}'.format(batch_rmse[-1]),
              'train_loss:{:.4}'.format(batch_loss[-1]),
              'test_loss:{:.4}'.format(loss2),
              'test_rmse:{:.4}'.format(test_rmse[-1]),
              'test_mae:{:.4}'.format(test_mae[-1]),
              'test_acc:{:.4}'.format(acc),
              'test_r2:{:.4}'.format(r2_score),
              'test_var:{:.4}'.format(var_score))

        if (epoch % SAVE_AFTER == 0):
            save_and_visualize(unit_index, pre_len, epoch, batch_rmse, totalbatch, test_rmse, test_pred,
                               test_label1, path,test_acc,test_mae, test_r2, test_var, time_start, batch_loss)
            # saver.save(sess, path+'/model_100/TGCN_pre_%r'%epoch, global_step = epoch)

    save_and_visualize(unit_index, pre_len, training_epoch, batch_rmse, totalbatch, test_rmse, test_pred, test_label1,
                       path,test_acc,test_mae, test_r2, test_var, time_start, batch_loss)

    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    tf.reset_default_graph()


path = '../../data2model/hypeerparam_results'
if not os.path.exists(path):
    os.makedirs(path)
vis_results(path=path)