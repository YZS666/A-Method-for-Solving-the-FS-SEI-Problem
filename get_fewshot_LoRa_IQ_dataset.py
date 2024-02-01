import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
from scipy import signal
import math
import json
import h5py

def convert_to_I_Q_complex(data):
    '''Convert the loaded data to complex I and Q samples.'''
    num_row = data.shape[0]
    num_col = data.shape[1]
    data_complex = np.zeros([num_row, 2, round(num_col/2)])
    data_complex[:,0,:] = data[:,:round(num_col/2)]
    data_complex[:,1,:] = data[:,round(num_col/2):]

    return data_complex


def LoadDataset(file_path, dev_range, pkt_range):
    '''
    Load IQ sample from a dataset
    Input:
    file_path is the dataset path
    dev_range specifies the loaded device range
    pkt_range specifies the loaded packets range

    Return:
    data is the loaded complex IQ samples
    label is the true label of each received packet
    '''

    dataset_name = 'data'
    labelset_name = 'label'

    f = h5py.File(file_path, 'r')
    label = f[labelset_name][:]
    label = label.astype(int)
    label = np.transpose(label)
    label = label - 1

    label_start = int(label[0]) + 1
    label_end = int(label[-1]) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt/num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ',' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)

    data = f[dataset_name][sample_index_list]
    data = convert_to_I_Q_complex(data)
    # data = np.expand_dims(data, axis=1)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    # data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    # data = data.transpose(0, 3, 1, 2)
    # data = np.squeeze(data, axis=3)
    # data = convert_to_I_Q_complex(data)
    label = label[sample_index_list]

    f.close()
    return data, label

def Get_LoRa_ALLIQDataset(file_path, dev_range, pkt_range):
    X,Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    return X,Y

def Get_LoRa_IQDataset(file_path, dev_range, pkt_range):
    X,Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=30)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def get_num_class_Sourcetraindata(num):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Train/dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_spectrogram_Dataset(file_path, dev_range, pkt_range)
    train_index_shot = []
    val_index_shot = []
    for i in range(num):
        train_index_shot += [index for index, value in enumerate(Y_train) if value == i]
        val_index_shot += [index for index, value in enumerate(Y_val) if value == i]
    return X_train[train_index_shot], X_val[val_index_shot], Y_train[train_index_shot], Y_val[val_index_shot]

def get_num_class_Targettraindata(num):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_spectrogram_Dataset(file_path, dev_range, pkt_range)
    train_index_shot = []
    val_index_shot = []
    for i in range(num):
        train_index_shot += [index for index, value in enumerate(Y_train) if value == i]
        val_index_shot += [index for index, value in enumerate(Y_val) if value == i]
    return X_train[train_index_shot], X_val[val_index_shot], Y_train[train_index_shot], Y_val[val_index_shot]

def get_num_class_Sourcetestdata(num):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Train/dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    test_index_shot = []
    for i in range(num[0], num[1]):
        test_index_shot += [index for index, value in enumerate(Y_test) if value == i]
    return X_test[test_index_shot], Y_test[test_index_shot]

def get_num_class_Targettestdata(num):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Get_LoRa_IQDataset(file_path, dev_range, pkt_range)
    test_index_shot = []
    for i in range(num[0], num[1]):
        test_index_shot += [index for index, value in enumerate(Y_test) if value == i]
    return X_test[test_index_shot], Y_test[test_index_shot]

def get_num_class_Sourcetrainfinetunedata(num, k):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Train/dataset_training_no_aug.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 500, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_finetune_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_finetune_index_shot += random.sample(train_index_classi, k)

    X_fintune_train_val, Y_fintune_train_val = X_train_val[train_val_finetune_index_shot], Y_train_val[
        train_val_finetune_index_shot]
    X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val = train_test_split(X_fintune_train_val,
                                                                                      Y_fintune_train_val,
                                                                                      test_size=0.2, random_state=30)

    return X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val


def get_num_class_Targettrainfinetunedata(num, k):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_finetune_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_finetune_index_shot += random.sample(train_index_classi, k)

    X_fintune_train_val, Y_fintune_train_val = X_train_val[train_val_finetune_index_shot], Y_train_val[
        train_val_finetune_index_shot]
    X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val = train_test_split(X_fintune_train_val,
                                                                                      Y_fintune_train_val,
                                                                                      test_size=0.2, random_state=30)

    return X_fintune_train, X_fintune_val, Y_fintune_train, Y_fintune_val

def get_num_class_TargetSemitraindata(num, k):
    file_path = '/data/yaozs/dataset/LoRa_dataset/Test/dataset_seen_devices.h5'
    dev_range = np.arange(0, 30, dtype=int)
    pkt_range = np.arange(0, 400, dtype=int)
    X, Y = LoadDataset(file_path, dev_range, pkt_range)
    Y = Y.astype(np.uint8)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.1, random_state=30)
    train_val_label_index_shot = []
    train_val_unlabel_index_shot = []

    for i in range(num):
        train_index_classi = [index for index, value in enumerate(Y_train_val) if value == i]
        train_val_label_index_shot += random.sample(train_index_classi, k)

    for k in range(len(X_train_val)):
        if k not in train_val_label_index_shot:
            train_val_unlabel_index_shot = np.append(train_val_unlabel_index_shot, k)

    train_val_unlabel_index_shot = train_val_unlabel_index_shot.astype('int64')

    X_label_train_val, Y_label_train_val = X_train_val[train_val_label_index_shot], Y_train_val[
        train_val_label_index_shot]
    X_label_train, X_label_val, Y_label_train, Y_label_val = train_test_split(X_label_train_val,
                                                                                      Y_label_train_val,
                                                                                      test_size=0.2, random_state=30)

    X_unlabel_train_val, Y_unlabel_train_val = X_train_val[train_val_unlabel_index_shot], Y_train_val[
        train_val_unlabel_index_shot]

    return X_label_train, X_unlabel_train_val, X_label_val, \
           Y_label_train, Y_unlabel_train_val, Y_label_val


def rand_bbox(size,mask_ratio):
    length = size[2]
    cut_length = np.int(length*mask_ratio)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def MaskData(data, mask_ratio):
    bbx1, bbx2 = rand_bbox(data.size(), mask_ratio)
    data[:, :, bbx1: bbx2] = torch.zeros((data.size()[1],bbx2-bbx1)).cuda()
    return bbx1, bbx2, data


if __name__ == '__main__':
    num = 30
    X_label_train, X_unlabel_train_val, X_label_val, Y_label_train, Y_unlabel_train_val, Y_label_val=\
        get_num_class_TargetSemitraindata(num, 20)
    print('success')
