import os
from random import *
import librosa
import numpy as np
import glob


def get_data(label, number: int, true_p=0.3, is_train=True):
    # 接受所需的标签，返回对应的正类及负类文件名
    # 加一个标签，决定从训练集还是测试集读取
    dataset_path = './dataset/IRMAS-TrainingData'
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
