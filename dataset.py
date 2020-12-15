import os
from random import *
import librosa
import numpy as np
import glob


def get_data(label, number: int, true_p=0.3, is_train=True):
    # 接受所需的标签，返回对应的正类及负类文件名
    # 加一个标签，决定从训练集还是测试集读取
    dataset_path = './IRMAS-TrainingData'
    all_pos_data = glob.glob(
        f'{dataset_path}/{label}/*.wav')
    pos_data_file = sample(all_pos_data, int(number * true_p))
    # print(f'getting pos data : {pos_data_file}')

    all_neg_file = glob.glob(
        f'{dataset_path}/[!{label[0]}][!{label[1]}][!{label[2]}]/*.wav')

    neg_data_file = sample(all_neg_file, int(number * (1-true_p)))
    # print(f'getting neg data : {neg_data_file}')

    all_file = pos_data_file + neg_data_file
    # print(all_file)

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
    true_list = [True] * int(number * true_p) + \
        [False] * int(number * (1-true_p))
    true_list = np.array(true_list)

    p = np.random.permutation(len(true_list))
    return waves[p], samples[p], true_list[p]

def get_data_list(label_list, number: int, true_p=0.3, is_train=True):
    # 接受所需的标签列表，返回对应的正类及负类文件名,true_p现在表示全部正类占比了
    # 加一个标签，决定从训练集还是测试集读取
    dataset_path = './IRMAS-TrainingData'

    all_pos_data = []
    for label in label_list:
        pos_data = glob.glob(
            f'{dataset_path}/{label}/*.wav')
        all_pos_data += pos_data
    pos_data_file = sample(all_pos_data, int(number * true_p))
    # print(f'getting pos data : {pos_data_file}')

    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
              'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器

    all_neg_file = []
    for label in labels:
        if label in label_list:
            continue
        neg_file = glob.glob(f'{dataset_path}/[!{label[0]}][!{label[1]}][!{label[2]}]/*.wav')
        all_neg_file += neg_file

    neg_data_file = sample(all_neg_file, int(number * (1-true_p)))
    # print(f'getting neg data : {neg_data_file}')

    all_file = pos_data_file + neg_data_file
    # print(all_file)

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
    true_list = [True] * int(number * true_p) + \
        [False] * int(number * (1-true_p))
    true_list = np.array(true_list)

    p = np.random.permutation(len(true_list))
    return waves[p], samples[p], true_list[p]

def get_data_list_weight(label_list, number: int, weight, is_train=True):
    # 接受所需的标签列表，返回对应的正类及负类文件名,true_p现在没了，用weight表示比例
    # 加一个标签，决定从训练集还是测试集读取
    dataset_path = './IRMAS-TrainingData'
    labels = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']  # 需要识别的乐器

    true_list = []

    pos_data_file = []
    for i in range(0,11):
        label = labels[i]
        if label in label_list:
            pos_file = glob.glob(f'{dataset_path}/{label}/*.wav')
            pos_data_file += sample(pos_file, int(number * weight[i]))
            true_list += [True] * int(number * weight[i])

    neg_data_file = []
    for i in range(0,11):
        label = labels[i]
        if label in label_list:
            continue
        neg_file = glob.glob(f'{dataset_path}/[!{label[0]}][!{label[1]}][!{label[2]}]/*.wav')
        neg_data_file += sample(neg_file, int(number * weight[i]))
        true_list += [False] * int(number * weight[i])

    all_file = pos_data_file + neg_data_file
    # print(all_file)

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

if __name__ == '__main__':
    waves,samples,true_list=get_data_list_weight(['cel','vio'],10,[0.5,0,0,0,0,0,0,0,0.1,0.4,0])