import librosa
import numpy as np


def get_feature(waves, samples):
    # 接受文件名及其对应的标签，返回对应特征向量
    # 返回特征的维度为 N * (f+1), N为样本数，f为每条样本的特征维度，在特征的最后拼接该样本的label(1或0)
    mfccs = []
    for i in range(waves.shape[0]):
        mfcc = librosa.feature.mfcc(y=waves[i], sr=samples[i], n_mfcc=24)
        mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    print(f'mfccs.shape = {mfccs.shape}')
    return mfccs
