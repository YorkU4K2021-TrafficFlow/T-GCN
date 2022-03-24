# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_result(test_result,test_label1,path):
    ##all test result visualization
    fig1 = plt.figure(figsize=(10,4))
    #    ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    # x= [12*i for i in range(9)]
    # labels(['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','00:00'])
    # plt.xticks(x, labels)
    plt.plot(a_pred,'r-',label='prediction', linewidth=0.4)
    plt.plot(a_true,'b-',label='true', linewidth=0.3)
    plt.legend(loc='best',fontsize=10)
    plt.title('Test all')
    plt.xlabel('timestamp number')
    plt.ylabel('speed (m/s)')
    plt.savefig(path+'/test_all.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    ## oneday test result visualization
    fig1 = plt.figure(figsize=(10,4))
    #    ax1 = fig1.add_subplot(1,1,1)
    #   24h*60/15=96 -> 1 day
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    plt.plot(a_pred,'r-',label="prediction", linewidth=0.5)
    plt.plot(a_true,'b-',label="true", linewidth=0.5)
    x= [12*i for i in range(9)]
    labels=(['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00','00:00'])
    plt.xticks(x, labels)
    plt.title('Test one day')
    plt.xlabel('timestamp number')
    plt.ylabel('speed (m/s)')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_oneday.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    plt.close('all')


def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(train_rmse, 'r-', label="train RMSE")
    plt.plot(test_rmse, 'b-', label="test RMSE")
    plt.title('RMSE per iteration')
    plt.xlabel('iteration')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.title('Train loss per iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    # fig1 = plt.figure(figsize=(5,4))
    # plt.plot(train_rmse,'b-', label='train_rmse')
    # plt.title('train rmse per iteration')
    # plt.xlabel('iteration')
    # plt.ylabel('train rmse')
    # plt.legend(loc='best',fontsize=10)
    # plt.savefig(path+'/train_rmse.png')
    # # plt.show()
    # plt.clf()

    ### accuracy
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(test_acc, 'b-', label="test accuracy")
    plt.title('Accuracy per iteration')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    ### rmse
    # fig1 = plt.figure(figsize=(5,4))
    # plt.plot(test_rmse, 'b-', label="test_rmse")
    # plt.title('test rmse per iteration')
    # plt.xlabel('iteration')
    # plt.ylabel('test rmse')
    # plt.legend(loc='best',fontsize=10)
    # plt.savefig(path+'/test_rmse.png')
    # # plt.show()
    # plt.clf()

    ### mae
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(test_mae, 'b-', label="test MAE")
    plt.title('Test MAE per iteration')
    plt.xlabel('iteration')
    plt.ylabel('test Mean Absolute Error (MAE)')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

    plt.close('all')