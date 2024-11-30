from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import pandas as pd


def data_processing(filename1, filename2):

    '''
        load dataset for model training
    '''

    snrs = ""
    SNR = ""
    data = pickle.load(open(filename1, 'rb'), encoding='latin')
    label = pickle.load(open(filename2, 'rb'), encoding='latin')

    snrs = sorted(list(set(data.keys())))
    X = []
    Y = []
    SNR = []
    for snr in snrs:
        X.append(data[snr])
        Y.append(label[snr])
        for i in range(data[snr].shape[0]):  SNR.append(snr)
    X = np.vstack(X)
    Y = np.vstack(Y)
    SNR = np.array(SNR, dtype='int16')

    total_num = len(X)
    shuffle_idx = np.random.choice(range(0, total_num), size=int(total_num), replace=False)
    dataset = X[shuffle_idx]
    labelset = Y[shuffle_idx]
    SNR = SNR[shuffle_idx]

    # split the whole dataset with ratio 3:1:1 into training, validation and testing set
    train_num = int(total_num * 0.6)
    val_num = int(total_num * 0.2)

    x_train = dataset[0:train_num]
    y_train = labelset[0:train_num]
    x_val = dataset[train_num:train_num + val_num]
    y_val = labelset[train_num:train_num + val_num]
    x_test = dataset[train_num + val_num:]
    y_test = labelset[train_num + val_num:]
    val_SNRs = SNR[train_num:train_num + val_num]
    test_SNRs = SNR[train_num + val_num:]

    print("Training data:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Validation data:", x_val.shape)
    print("Validation labels:", y_val.shape)
    print("Testing data", x_test.shape)
    print("Testing labels", y_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs


def performance_evaluation(save_path, x_test, y_test, test_SNRs, model):

    '''
        Evaluate final model's performance
    '''

    pd_list = []
    pf_list = []
    acc_list = []
    md_list = []
    snrs = np.linspace(-20, 18, 20)
    snrs = np.array(snrs, dtype='int16')
    for snr in snrs:
        test_x_i = x_test[np.where(test_SNRs == snr)]
        test_y_i = y_test[np.where(test_SNRs == snr)]
        test_y_i_hat = model.predict(test_x_i, verbose=0)
        test_y_i_hat = (test_y_i_hat > 0.5).astype(int)
        test_y_i_hat = np.reshape(test_y_i_hat, (-1))
        test_y_i = np.reshape(test_y_i, (-1))
        cm = confusion_matrix(test_y_i, test_y_i_hat, labels=[1,0])
        np.seterr(divide='ignore', invalid='ignore')
        acc = (cm[0][0]+cm[1][1])/sum(map(sum, cm))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        pd_list.append(cm_norm[0][0])
        pf_list.append(cm_norm[1][0])
        md_list.append(cm_norm[0][1])
        acc_list.append(acc)
    output_excel = {'pd': pd_list, 'md': md_list, 'pf': pf_list, 'acc': acc_list}
    output = pd.DataFrame(list(output_excel.items()))
    output.to_excel(save_path, index=False, header=False)


def tl_data_processing(filename1, filename2, num):

    data = pickle.load(open(filename1, 'rb'), encoding='latin')
    label = pickle.load(open(filename2, 'rb'), encoding='latin')

    snrs = sorted(list(set(data.keys())))
    X = []
    Y = []
    SNR = []
    for snr in snrs:
        X.append(data[snr])
        Y.append(label[snr])
        for i in range(data[snr].shape[0]):  SNR.append(snr)
    X = np.vstack(X)
    Y = np.vstack(Y)
    SNR = np.array(SNR, dtype='int16')

    total_num = len(X)
    shuffle_idx = np.random.choice(range(0, total_num), size=int(total_num), replace=False)
    dataset = X[shuffle_idx]
    labelset = Y[shuffle_idx]
    SNR = SNR[shuffle_idx]

    # split the whole dataset with ratio 3:1:1 into training, validation and testing set
    train_num = int(total_num * 0.6)
    val_num = int(total_num * 0.2)

    x_train = dataset[0:train_num]
    y_train = labelset[0:train_num]
    x_val = dataset[train_num:train_num + val_num]
    y_val = labelset[train_num:train_num + val_num]
    val_SNRs = SNR[train_num:train_num + val_num]

    x_test = dataset[train_num + val_num:]
    y_test = labelset[train_num + val_num:]
    test_SNRs = SNR[train_num + val_num:]

    # Select the training set for transfer learning
    transfer_shuffle_idx = np.random.choice(range(0, len(x_train)), size=int(num), replace=False)
    x_train = x_train[transfer_shuffle_idx]
    y_train = y_train[transfer_shuffle_idx]

    print("Training data:", x_train.shape)
    print("Training labels:", y_train.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, val_SNRs, test_SNRs

