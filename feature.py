import librosa
import numpy as np


def get_feature(pos_data, neg_data):
    # 接受文件名及其对应的标签，返回对应特征向量
    # 返回特征的维度为 N * (f+1), N为样本数，f为每条样本的特征维度，在特征的最后拼接该样本的label(1或0)
    pos_mfccs = librosa.feature.mfcc(y=pos_data[0], sr=pos_data[1], n_mfcc=24)
    print(f'pos data mfcc shape : {pos_mfccs.shape}')
    neg_mfccs = librosa.feature.mfcc(y=neg_data[0], sr=neg_data[1], n_mfcc=24)
    print(f'neg data mfcc shape : {neg_mfccs.shape}')
    return pos_mfccs, neg_mfccs
