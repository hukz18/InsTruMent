import os
from random import *
import librosa
import numpy as np
from glob import glob


def get_data_list_weighted(dataset_path: str, label_dict: dict,  is_train=True):
    '''
    label_dict : the key is string of label , word is (is_positive_class, number)
    returns: 
        waves: numpy matrix, each row represents a wave, each column represents a time sample
        samples: the sample rate of waves
        true_list: whether the wave is positive class
    '''
    true_list = []
    all_file = []
    for label, (is_pos, num) in label_dict.items():
        assert type(is_pos) is bool
        all_file += sample(glob(f'{dataset_path}/{label}/*.wav'), num)
        true_list += [is_pos] * num

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
    true_list = []
    all_file = []
    for label, (is_pos, num) in label_dict.items():
        assert type(is_pos) is bool
        all_file += sample(glob(f'{dataset_path}/{label}/*.npy'), num)
        true_list += [is_pos] * num

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

    dataset_path = './dataset/IRMAS-TestingData-Part1/Part1'
    all_data = glob(f'{dataset_path}/*.wav')
    # 打乱
    shuffle(all_data)
    if num is not None:
        all_data = all_data[:num]
    for file in all_data:
        w, s = librosa.load(file)
        # print(w.shape)
        w = np.array(w)
        s = np.array(s)

        label_path = file[0:-3] + 'txt'
        txtfile = open(label_path, 'r')
        true_class = [x.strip() for x in txtfile]
        txtfile.close()
        yield w, s, true_class

def get_test_data_whole(num=None):
    # 读取测试集wave和sample，还有标签，返回迭代器

    dataset_path = './dataset/IRMAS-TestingData-Part1/Part1'
    all_data = glob.glob(f'{dataset_path}/*.wav')
    # 打乱
    shuffle(all_data)
    if num is not None:
        all_data = all_data[:num]

    waves = []
    samples = []
    true_list = []
    for file in all_data:
        w, s = librosa.load(file)
        # print(w.shape)
        w = np.array(w)
        s = np.array(s)

        label_path = file[0:-3] + 'txt'
        txtfile = open(label_path, 'r')
        true_class = [x.strip() for x in txtfile]
        txtfile.close()
        
        waves.append(w)
        samples.append(s)
        true_list.append(true_class)
    return waves, samples, true_list

if __name__ == '__main__':
    w, s, true_class = get_data_list(['voi'],10)
