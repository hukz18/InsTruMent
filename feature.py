import librosa
import numpy as np
from glob import glob


def generate_features(data_path):
    all_files = glob(f'{data_path}/*/*.wav')
    # print(all_files)
    for file in all_files:
        print(file)
        w, s = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=w, sr=s, n_mfcc=40)
        print(mfcc.shape)
        np.save(file.replace('wav', 'npy'), mfcc)
        # break


def get_feature(waves, samples):
    # 接受文件名及其对应的标签，返回对应特征向量
    # 返回特征的维度为 N * (f+1), N为样本数，f为每条样本的特征维度，在特征的最后拼接该样本的label(1或0)
    mfccs = []
    for i in range(waves.shape[0]):
        mfcc = librosa.feature.mfcc(y=waves[i], sr=samples[i], n_mfcc=40)
        mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    print(f'mfccs.shape = {mfccs.shape}')
    return mfccs


def get_feature_npy(mfccs):
    '''
    input:
        mfccs : N * F * L , a set of mfccs
    return:
        mfcc_result : N * F
    '''
    freq_sum = np.sum(mfccs, axis=1)
    # print(freq_sum)
    sort_ind = np.argsort(freq_sum, axis=1)
    # print(sort_ind)
    sort_ind_ext = np.tile(sort_ind, (mfccs.shape[1], 1, 1))
    sort_ind_ext = np.moveaxis(sort_ind_ext, 1, 0)
    # print(sort_ind_ext)
    result = np.take_along_axis(mfccs, sort_ind_ext, axis=2)
    # print(result)
    avg_max_result = np.mean(result[:, :, 50:], axis=2)
    # print(avg_max_result)
    return avg_max_result


# x = np.random.randint(0, 9, (3, 4, 5))
# print(x)

# get_feature_npy(x)
