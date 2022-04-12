"""
This script creates animations that show the relationship between number of GRU units to
`param` ['ACCURACY', 'MAE', 'R2', 'RMSE', 'VAR'] over different numbers of epochs.


Each row in 'param' corresponds to a GRU unit in that order: (8, 16, 32, 64, 100)
Each column in 'param' corresponds to number of epochs in a 500 increment (500, 1000, 1500, 2000, 2500, 3000)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===========================================================================
GRU_UNITS = [8, 16, 32, 64, 100]  # ,128]
sensors = ['30', '60', '90']  # , '120']
params = ['ACCURACY', 'MAE', 'R2', 'RMSE', 'VAR']
x_axis = [x for x in range(len(GRU_UNITS))]
SIZE = 6
writer = animation.FFMpegWriter(fps=2, extra_args=['-vcodec', 'libx264'])
INDEX, v, w, label, path, title, min_val, max_val = None, None, None, None, None, None, None, None


# ===========================================================================


def mutual_change():
    global INDEX, label, title, GRU_UNITS, axs, plt, min_val, max_val

    plt.setp(axs[0], xticks=x_axis, xticklabels=GRU_UNITS)
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # grid
    axs[0].grid(None, which='major', linestyle='-.')
    axs[0].minorticks_on()
    axs[0].set_ylim(min_val, max_val)
    axs[0].grid(b=True, which='minor', linestyle='--', alpha=0.2)
    axs[0].set_xlabel("Hidden Units")
    axs[0].set_ylabel(label)
    plt.suptitle(title + str(500 * (INDEX + 1)) + ' epochs')
    # axs[0].set_xticks(x_axis, GRU_UNITS)
    # plt.show()


def invisible():
    # invisible 2nd plot
    axs[1].axis('off')


# ===========================================================================


def vis_data():
    axs[0].plot(x_axis, v[INDEX][0], color='black', marker='s', label='30 sensors')
    axs[0].plot(x_axis, v[INDEX][1], color='red', marker='s', label='60 sensors')
    axs[0].plot(x_axis, v[INDEX][2], color='blue', marker='s', label='90 sensors')
    # plt.plot(x_axis, v[3], color='green', marker='s', label='120 sensors')
    mutual_change()


def viz(frameno):
    global INDEX, v, x_axis, axs, plt
    axs[0].cla()
    invisible()
    vis_data()

    INDEX += 1
    return axs


def init_vis():
    axs[0].plot(x_axis, v[INDEX][0], color='black', marker='s', label='30 sensors')
    axs[0].plot(x_axis, v[INDEX][1], color='red', marker='s', label='60 sensors')
    axs[0].plot(x_axis, v[INDEX][2], color='blue', marker='s', label='90 sensors')
    # plt.plot(x_axis, v[3], color='green', marker='s', label='120 sensors')


# ========================================


def vis_bar_data():
    global INDEX, v

    X = np.arange(5)
    axs[0].bar(X - 0.3, v[INDEX][0], color='black', label='30 sensors', width=0.2)
    axs[0].bar(X - 0.1, v[INDEX][1], color='red', label='60 sensors', width=0.2)
    axs[0].bar(X + 0.1, v[INDEX][2], color='blue', label='90 sensors', width=0.2)
    # axs[0].bar(X+0.3, v[3], color='green', label='120 sensors', width=0.2)
    mutual_change()

def viz_bar(frameno):
    global INDEX, v, x_axis, axs, plt

    axs[0].cla()
    invisible()
    vis_bar_data()

    INDEX += 1
    return axs


def init_vis_bar():
    vis_bar_data()


# ========================================


def contrast_data_length_data():
    global axs, x_axis, v, w, INDEX

    axs[0].plot(x_axis, v[INDEX][0], color='red', marker='s', label='30 S - 1 D')
    axs[0].plot(x_axis, v[INDEX][1], color='firebrick', marker='s', label='60 S - 1 D')
    axs[0].plot(x_axis, v[INDEX][2], color='maroon', marker='s', label='90 S - 1 D')
    # axs[0].plot(x_axis, v[INDEX][3], color='black', marker='s', label='120 sensors')

    axs[0].plot(x_axis, w[INDEX][0], color='deepskyblue', marker='v', ls='--', label='30 S - 5 D')
    axs[0].plot(x_axis, w[INDEX][1], color='teal', marker='v', ls='--', label='60 S - 5 D')
    axs[0].plot(x_axis, w[INDEX][2], color='darkslategray', marker='v', ls='--', label='90 S - 5 D')

    mutual_change()


def contrast_data_length(frameno):
    global INDEX, v, x_axis, axs, plt

    axs[0].cla()
    invisible()
    contrast_data_length_data()
    mutual_change()

    INDEX += 1
    return axs


def init_contrast_length():
    contrast_data_length_data()


# ========================================


def contast_data_diff_data():
    global axs, x_axis, v, w, INDEX

    axs[0].plot(x_axis, w[INDEX][0] - v[INDEX][0], color='black', marker='s', label='30 sensors')
    axs[0].plot(x_axis, w[INDEX][1] - v[INDEX][1], color='red', marker='s', label='60 sensors')
    axs[0].plot(x_axis, w[INDEX][2] - v[INDEX][2], color='blue', marker='s', label='90 sensors')
    # axs[0].plot(x_axis, w[INDEX][3]-v[INDEX][3], color='green', marker='s', label='120 sensors')

    mutual_change()


def contast_data_diff(frameno):
    global INDEX, axs

    axs[0].cla()
    invisible()
    contast_data_diff_data()

    INDEX += 1
    return axs


def init_contast_data_diff():
    contast_data_diff_data()


# ========================================


def contrast_data_length_bar_data():
    global axs, x_axis, v, w, INDEX

    X = np.arange(5)
    axs[0].bar(X-0.3, w[INDEX][0]-v[INDEX][0], color='black',  label='30 sensors', width=0.2)
    axs[0].bar(X-0.1, w[INDEX][1]-v[INDEX][1], color='red', label='60 sensors', width=0.2)
    axs[0].bar(X+0.1, w[INDEX][2]-v[INDEX][2], color='blue',  label='90 sensors', width=0.2)
    # axs[0].bar(x_axis, w[INDEX][3]-v[INDEX][3], color='green', label='120 sensors')
    mutual_change()


def contrast_data_length_bar(frameno):
    global INDEX, axs

    axs[0].cla()
    invisible()
    contrast_data_length_bar_data()

    INDEX += 1
    return axs


def init_contrast_data_length_bar():
    contrast_data_length_bar_data


# -------------------------------------------------------
# -------------------------------------------------------


# evaluation of sensors for 1-day and 5-day data separately
for IS_5_DAYS in [True, False]:
    for param in params:
        v1, v2, v3, v4, v5, v6 = [], [], [], [], [], []

        for sensor in sensors:
            if IS_5_DAYS:
                df = pd.read_csv('hyperparam_results_5_days/' + sensor + '_sensors/3_len/' + param + '.csv',
                                 header=None)
            else:
                df = pd.read_csv('hyperparam_results/' + sensor + '_sensors/3_len/' + param + '.csv', header=None)

            v1.append(df[0].values)
            v2.append(df[1].values)
            v3.append(df[2].values)
            v4.append(df[3].values)
            v5.append(df[4].values)
            v6.append(df[5].values)

        v = [v1, v2, v3, v4, v5, v6]

        min_val = min([np.array(v1).min(), np.array(v2).min(), np.array(v3).min(),
                       np.array(v4).min(), np.array(v5).min(), np.array(v6).min()]) - 0.05

        max_val = max([np.array(v1).max(), np.array(v2).max(), np.array(v3).max(),
                       np.array(v4).max(), np.array(v5).max(), np.array(v6).max()]) + 0.05

        if IS_5_DAYS:
            label = param
            path = 'hyperparam_results_5_days/3_len/' + param
            title = param + ' at each hidden layer for different number of sensors\n'
        else:
            label = param
            path = 'hyperparam_results/3_len/' + param
            title = param + ' at each hidden layer for different number of sensors\n'

        # line plot
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [6, 1]})
        INDEX = 0
        anim = animation.FuncAnimation(fig, viz, interval=1400, init_func=init_vis,
                                       repeat=False, blit=False, frames=SIZE)
        anim.save(path + '.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
        plt.close('all')

        # bar plot
        min_val = 0
        INDEX = 0
        path = path + '_bar'
        anim = animation.FuncAnimation(fig, viz_bar, interval=1400, init_func=init_vis_bar,
                                       repeat=False, blit=False, frames=SIZE)
        anim.save(path + '.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
        plt.close('all')


# -------------------------------------------------------
# 1-day vs. 5-days evaluation
# -------------------------------------------------------


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

    min_val = min([np.array(v1).min(), np.array(v2).min(), np.array(v3).min(),
                   np.array(v4).min(), np.array(v5).min(), np.array(v6).min(),
                   np.array(w1).min(), np.array(w2).min(), np.array(w3).min(),
                   np.array(w4).min(), np.array(w5).min(), np.array(w6).min()]) - 0.05

    max_val = max([np.array(v1).max(), np.array(v2).max(), np.array(v3).max(),
                   np.array(v4).max(), np.array(v5).max(), np.array(v6).max(),
                   np.array(w1).max(), np.array(w2).max(), np.array(w3).max(),
                   np.array(w4).max(), np.array(w5).max(), np.array(w6).max()]) + 0.05

    # --- animate contrast ---

    # line plot - all
    label = param
    path = 'res/diff/' + param + '_all'
    title = param + ' 1-day vs. 5-days for different sensor configuration\n'
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [6, 1]})
    INDEX = 0
    anim = animation.FuncAnimation(fig, contrast_data_length, interval=1400, init_func=init_contrast_length,
                                   repeat=False, blit=False, frames=SIZE)
    anim.save(path + '.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
    plt.close('all')

    # line plot - difference
    min_val = min([(np.array(w1)-np.array(v1)).min(), (np.array(w2)-np.array(v2)).min(),
                   (np.array(w3)-np.array(v3)).min(), (np.array(w3)-np.array(v4)).min(),
                   (np.array(w5)-np.array(v5)).min(), (np.array(w6)-np.array(v6)).min()]) - 0.05

    max_val = max([(np.array(w1)-np.array(v1)).max(), (np.array(w2)-np.array(v2)).max(),
                   (np.array(w3)-np.array(v3)).max(), (np.array(w3)-np.array(v4)).max(),
                   (np.array(w5)-np.array(v5)).max(), (np.array(w6)-np.array(v6)).max()]) + 0.05

    path = 'res/diff/' + param + '_diff'
    title = param + ' 5-days minus 1-day difference for different sensor configuration\n'
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [6, 1]})
    INDEX = 0
    anim = animation.FuncAnimation(fig, contast_data_diff, interval=1400, init_func=init_contast_data_diff,
                                   repeat=False, blit=False, frames=SIZE)
    anim.save(path + '.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
    plt.close('all')

    # bar plot - difference
    min_val = min([(np.array(w1)-np.array(v1)).min(), (np.array(w2)-np.array(v2)).min(),
                   (np.array(w3)-np.array(v3)).min(), (np.array(w3)-np.array(v4)).min(),
                   (np.array(w5)-np.array(v5)).min(), (np.array(w6)-np.array(v6)).min()])

    max_val = max([(np.array(w1)-np.array(v1)).max(), (np.array(w2)-np.array(v2)).max(),
                   (np.array(w3)-np.array(v3)).max(), (np.array(w3)-np.array(v4)).max(),
                   (np.array(w5)-np.array(v5)).max(), (np.array(w6)-np.array(v6)).max()])

    if param == 'MAE' or param == 'RMSE':
        max_val = 0
        min_val += 0.05
    else:
        min_val = 0
        max_val += 0.05

    path = 'res/diff/' + param + '_diff_bar'
    title = param + ' 5-days minus 1-day difference for different sensor configuration\n'
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [6, 1]})
    INDEX = 0
    anim = animation.FuncAnimation(fig, contrast_data_length_bar, interval=1400, init_func=init_contrast_data_length_bar,
                                   repeat=False, blit=False, frames=SIZE)
    anim.save(path + '.mp4', writer=writer, savefig_kwargs={'bbox_inches': 'tight'})
    plt.close('all')
