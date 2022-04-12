"""
This script creates plots that show the relationship between number of GRU units to
`param` ['ACCURACY', 'MAE', 'R2', 'RMSE', 'VAR'] over different numbers of epochs.


Each row in 'param' corresponds to a GRU unit in that order: (8, 16, 32, 64, 100)
Each column in 'param' corresponds to number of epochs in a 500 increment (500, 1000, 1500, 2000, 2500, 3000)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GRU_UNITS = [8, 16, 32, 64, 100]  # ,128]
sensors = ['30', '60', '90']#, '120']
params = ['ACCURACY', 'MAE', 'R2', 'RMSE', 'VAR']
x_axis = [x for x in range(len(GRU_UNITS))]


def viz(v, label, path, title):
    global GRU_UNITS, x_axis

    plt.plot(x_axis, v[0], color='black', marker='s', label='30 sensors')
    plt.plot(x_axis, v[1], color='red', marker='s', label='60 sensors')
    plt.plot(x_axis, v[2], color='blue', marker='s', label='90 sensors')
    # plt.plot(x_axis, v[3], color='green', marker='s', label='120 sensors')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    plt.grid(None, which='major', linestyle='-.')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.2)

    plt.xlabel("Hidden Units")
    plt.ylabel(label)
    plt.title(title)

    plt.xticks(x_axis, GRU_UNITS)
    plt.savefig(path + '.png', bbox_inches='tight')

    plt.show()
    plt.clf()


def viz_bar(v, label, path, title):
    global GRU_UNITS, x_axis

    X = np.arange(5)
    plt.bar(X-0.3, v[0], color='black', label='30 sensors', width=0.2)
    plt.bar(X-0.1, v[1], color='red', label='60 sensors', width=0.2)
    plt.bar(X+0.1, v[2], color='blue', label='90 sensors', width=0.2)
    # plt.bar(X+0.3, v[3], color='green', label='120 sensors', width=0.2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    plt.grid(None, which='major', linestyle='-.')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.2)

    plt.xlabel("Hidden Units")
    plt.ylabel(label)
    plt.title(title)

    plt.xticks(x_axis, GRU_UNITS)
    plt.savefig(path + '.png', bbox_inches='tight')

    plt.show()
    plt.clf()


def contrast_data_length(v, w, label, path, title, diff_path, diff_title):
    global GRU_UNITS, x_axis

    plt.plot(x_axis, v[0], color='red', marker='s', label='30 sensors - 1 day')
    plt.plot(x_axis, v[1], color='firebrick', marker='s', label='60 sensors - 1 day')
    plt.plot(x_axis, v[2], color='maroon', marker='s', label='90 sensors - 1 day')
    # plt.plot(x_axis, v[3], color='green', marker='s', label='120 sensors')

    plt.plot(x_axis, w[0], color='deepskyblue', marker='v', ls='--', label='30 sensors - 5 days')
    plt.plot(x_axis, w[1], color='teal', marker='v', ls='--', label='60 sensors - 5 days')
    plt.plot(x_axis, w[2], color='darkslategray', marker='v', ls='--', label='90 sensors - 5 days')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    plt.grid(None, which='major', linestyle='-.')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.2)

    plt.xlabel("Hidden Units")
    plt.ylabel(label)
    plt.title(title)

    plt.xticks(x_axis, GRU_UNITS)
    plt.savefig(path + '.png', bbox_inches='tight')

    plt.show()
    plt.clf()

    plt.plot(x_axis, w[0]-v[0], color='black', marker='s', label='30 sensors')
    plt.plot(x_axis, w[1]-v[1], color='red', marker='s', label='60 sensors')
    plt.plot(x_axis, w[2]-v[2], color='blue', marker='s', label='90 sensors')
    # plt.plot(x_axis, w[3]-v[3], color='green', marker='s', label='120 sensors')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    plt.grid(None, which='major', linestyle='-.')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.2)

    plt.xlabel("Hidden Units")
    plt.ylabel(label)
    plt.title(diff_title)

    plt.xticks(x_axis, GRU_UNITS)
    plt.savefig(diff_path + '.png', bbox_inches='tight')

    plt.show()
    plt.clf()


def contrast_data_length_bar(v, w, label, path, title):
    global GRU_UNITS, x_axis

    X = np.arange(5)
    plt.bar(X-0.3, w[0]-v[0], color='black',  label='30 sensors', width=0.2)
    plt.bar(X-0.1, w[1]-v[1], color='red', label='60 sensors', width=0.2)
    plt.bar(X+0.1, w[2]-v[2], color='blue',  label='90 sensors', width=0.2)
    # plt.bar(x_axis, w[3]-v[3], color='green', label='120 sensors')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    plt.grid(None, which='major', linestyle='-.')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--', alpha=0.2)

    plt.xlabel("Hidden Units")
    plt.ylabel(label)
    plt.title(title)

    plt.xticks(x_axis, GRU_UNITS)
    plt.savefig(path + '.png', bbox_inches='tight')

    plt.show()
    plt.clf()


# evaluation of sensors for 1-day and 5-day data separately
for IS_5_DAYS in [True, False]:
    for param in params:
        v1, v2, v3, v4, v5, v6 = [], [], [], [], [], []

        for sensor in sensors:
            if IS_5_DAYS:
                df = pd.read_csv('hyperparam_results_5_days/'+sensor+'_sensors/3_len/'+param + '.csv', header=None)
            else:
                df = pd.read_csv('hyperparam_results/' + sensor + '_sensors/3_len/' + param + '.csv', header=None)

            v1.append(df[0].values)
            v2.append(df[1].values)
            v3.append(df[2].values)
            v4.append(df[3].values)
            v5.append(df[4].values)
            v6.append(df[5].values)

        v = [v1, v2, v3, v4, v5, v6]

        for i in range(len(v)):
            if IS_5_DAYS:
                viz(v[i], label=param, path='hyperparam_results_5_days/3_len/' + param + '-' + str(500 * (i + 1)) + '_epochs',
                    title=param + ' at each hidden layer for different number of sensors\n' + str(
                        500 * (i + 1)) + ' epochs')
            else:
                viz(v[i], label=param, path='hyperparam_results/3_len/' + param + '-' + str(500 * (i + 1)) + '_epochs',
                        title=param + ' at each hidden layer for different number of sensors\n' + str(
                            500 * (i + 1)) + ' epochs')

        for i in range(len(v)):
            if IS_5_DAYS:
                viz_bar(v[i], label=param, path='hyperparam_results_5_days/3_len/bar_' + param + '-' + str(500 * (i + 1)) + '_epochs',
                        title=param + ' at each hidden layer for different number of sensors\n' + str(
                            500 * (i + 1)) + ' epochs')
            else:
                viz_bar(v[i], label=param, path='hyperparam_results/3_len/bar_' + param + '-' + str(500 * (i + 1)) + '_epochs',
                        title=param + ' at each hidden layer for different number of sensors\n' + str(
                            500 * (i + 1)) + ' epochs')

for param in params:
    v1, v2, v3, v4, v5, v6 = [], [], [], [], [], []
    w1, w2, w3, w4, w5, w6 = [], [], [], [], [], []

    for sensor in sensors:
        df5 = pd.read_csv('hyperparam_results_5_days/'+sensor+'_sensors/3_len/'+param + '.csv', header=None)
        df = pd.read_csv('hyperparam_results/' + sensor + '_sensors/3_len/' + param + '.csv', header=None)

        v1.append(df[0].values)
        v2.append(df[1].values)
        v3.append(df[2].values)
        v4.append(df[3].values)
        v5.append(df[4].values)
        v6.append(df[5].values)

        w1.append(df5[0].values)
        w2.append(df5[1].values)
        w3.append(df5[2].values)
        w4.append(df5[3].values)
        w5.append(df5[4].values)
        w6.append(df5[5].values)

    v = [v1, v2, v3, v4, v5, v6]
    w = [w1, w2, w3, w4, w5, w6]

    for i in range(len(v)):
        contrast_data_length(v[i], w[i], label=param, path='res/' + param + '-' + str(500 * (i + 1)) + '_epochs',
            title=param + ' 1-day vs. 5-days for different sensor configuration\n' + str(500 * (i + 1)) + ' epochs',
            diff_path='res/diff/' + param + '-' + str(500 * (i + 1)) + '_epochs_diff',
            diff_title=param + ' 5-days minus 1-day difference for different sensor configuration\n' + str(500 * (i + 1)) + ' epochs',)

        contrast_data_length_bar(v[i], w[i], label=param,
             path='res/diff/bar_' + param + '-' + str(500 * (i + 1)) + '_epochs_diff',
             title=param+' 5-days minus 1-day difference for different sensor configuration\n' + str(500 * (i + 1)) + ' epochs')
