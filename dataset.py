import os
from random import shuffle, sample
import librosa
import numpy as np
from glob import glob


def sample_multiple(l: list, num: int):
    result = []
    while num > len(l):
        result += sample(l, len(l))
        num -= len(l)
    result += sample(l, num)
    return result


def get_file_list(dataset_path: str, label_dict: dict, file_type: str,  is_train=True,num=0):
    true_list = []
    all_file = []
    if num ==1 or num==2:
        for label, (is_pos, num) in label_dict.items():
            assert type(is_pos) is bool
            all_file += sample_multiple(
                glob(f'{dataset_path}/{label}/*({num}).{file_type}'), num)
            true_list += [is_pos] * num
    else:
        for label, (is_pos, num) in label_dict.items():
            assert type(is_pos) is bool
            all_file += sample_multiple(
                glob(f'{dataset_path}/{label}/*.{file_type}'), num)
            true_list += [is_pos] * num
    return all_file, true_list


def get_data_list_weighted(dataset_path: str, label_dict: dict,  is_train=True,num=0):
    '''
    label_dict : the key is string of label , word is (is_positive_class, number)
    returns:
        waves: numpy matrix, each row represents a wave, each column represents a time sample
        samples: the sample rate of waves
        true_list: whether the wave is positive class
    '''
    all_file, true_list = get_file_list(
        dataset_path, label_dict, file_type='wav', is_train=is_train,num=num)
    waves = []
    samples = []
    for file in all_file:
        w, s = librosa.load(file)
        # print(w.shape)
        waves.append(w)
        samples.append(s)
    waves = np.array(waves)
    print(f'waves.shape = {waves.shape}')
    samples = np.array(samples)
    true_list = np.array(true_list)

    p = np.random.permutation(len(true_list))
    return waves[p], samples[p], true_list[p]


def get_data_list_weighted_npy(dataset_path: str, label_dict: dict,  is_train=True):
    '''
    label_dict : the key is string of label , word is (is_positive_class, number)
    returns:
        waves: numpy matrix, each row represents a wave, each column represents a time sample
        samples: the sample rate of waves
        true_list: whether the wave is positive class
    '''
    all_file, true_list = get_file_list(
        dataset_path, label_dict, file_type='npy', is_train=is_train)
    mfccs = []
    for file in all_file:
        mfcc = np.load(file)
        mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    print(f'mfccs.shape = {mfccs.shape}')
    true_list = np.array(true_list)

    p = np.random.permutation(len(true_list))
    return mfccs[p], true_list[p]


def get_test_data_iter(num=None):
    # 读取测试集wave和sample，还有标签，返回迭代器

    dataset_path = './dataset/TestingData'
    all_data = glob(f'{dataset_path}/*.npy')
    # 打乱
    shuffle(all_data)
    if num is not None:
        all_data = all_data[:num]

    for file in all_data:
        mfcc = np.load(file)
        label_path = file[0:-7] + '.txt'
        txtfile = open(label_path, 'r')
        true_class = [x.strip() for x in txtfile]
        txtfile.close()
        yield mfcc, true_class


def get_test_data_whole(dataset_path: str, num=None):
    # 读取测试集wave和sample，还有标签，返回迭代器
    all_data = glob(f'{dataset_path}/*.npy')
    # 打乱
    # shuffle(all_data)
    if num is not None:
        all_data = all_data[:num]

    mfccs = []
    true_list = []
    for file in all_data:
        mfcc = np.load(file)
        # print(f'mfcc.shape = {mfcc.shape}')
        mfcc = np.hstack(
            (mfcc, np.zeros((mfcc.shape[0], 1723-mfcc.shape[1]))))
        mfccs.append(mfcc)

        label_path = file[0:-7] + '.txt'
        txtfile = open(label_path, 'r')
        true_class = [x.strip() for x in txtfile]
        txtfile.close()

        true_list.append(true_class)
    mfccs = np.array(mfccs)
    return mfccs, true_list


if __name__ == '__main__':
    for mfcc, true_class in get_test_data_whole():
        print(true_class)
