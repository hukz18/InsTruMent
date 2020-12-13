import os
from random import choice
import librosa
import numpy as np


def get_data(label, is_train=True):
    # 接受所需的标签，返回对应的正类及负类文件名
    # 加一个标签，决定从训练集还是测试集读取
    dataset_path = './IRMAS-TrainingData'
    all_pos_data = os.listdir(f'{dataset_path}/{label}')
    pos_data_file = f'{dataset_path}/{label}/{choice(all_pos_data)}'
    print(f'getting pos data : {pos_data_file}')
    files = os.listdir(dataset_path)
    files.remove(label)
    neg_label = choice(files)
    all_neg_data = os.listdir(f'{dataset_path}/{neg_label}')
    neg_data_file = f'{dataset_path}/{neg_label}/{choice(all_neg_data)}'
    print(f'getting neg data : {neg_data_file}')

    pos_wav, pos_sr = librosa.load(pos_data_file)
    neg_wav, neg_sr = librosa.load(neg_data_file)
    return (pos_wav, pos_sr), (neg_wav, neg_sr)
