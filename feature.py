import librosa
from librosa.feature.spectral import mfcc
import numpy as np
from glob import glob
import shutil
from scipy.io import wavfile


def generate_features(data_path):
    all_files = glob(f'{data_path}/*/*.wav')
    # print(all_files)
    for file in all_files:
        print(file)
        s, w = wavfile.read(file)
        w = w.astype(float)
        mfcc1 = librosa.feature.mfcc(y=w[:, 0], sr=s, n_mfcc=43)
        mfcc2 = librosa.feature.mfcc(y=w[:, 1], sr=s, n_mfcc=43)
        # mfcc1 = librosa.feature.mfcc(y=w[::4, 0], sr=s // 4, n_mfcc=43)
        # mfcc2 = librosa.feature.mfcc(y=w[::4, 1], sr=s // 4, n_mfcc=43)
        print(mfcc1.shape)
        np.save(file.replace('.wav', '(1).npy'), mfcc1)
        np.save(file.replace('.wav', '(2).npy'), mfcc2)
        # break


def generate_dataset(data_path, num):
    all_files = glob(f'{data_path}/*/*.txt')
    for file in all_files:
        print(file)
        file = file.replace(f'\\', '/')
        with open(file, 'r') as txtfile:
            true_class = [x.strip() for x in txtfile]
        if 'voi' in true_class:
            continue
        is_test = 'TestingData' if np.random.random() > 0.8 else 'TrainingData'
        copy_file = file.replace(
            f'IRMAS-TestingData-Part{num}', is_test).replace(f'Part{num}/', '')
        shutil.copyfile(file, copy_file)
        shutil.copyfile(file.replace('.txt', '(1).npy'),
                        copy_file.replace('.txt', '(1).npy'))
        shutil.copyfile(file.replace('.txt', '(2).npy'),
                        copy_file.replace('.txt', '(2).npy'))


def get_feature(waves, samples):
    # 接受文件名及其对应的标签，返回对应特征向量
    # 返回特征的维度为 N * (f+1), N为样本数，f为每条样本的特征维度，在特征的最后拼接该样本的label(1或0)
    mfccs = []
    for i in range(waves.shape[0]):
        mfcc = librosa.feature.mfcc(y=waves[i], sr=samples[i], n_mfcc=40)
        mfccs.append(mfcc)
    mfccs = np.array(mfccs)
    # print(f'mfccs.shape = {mfccs.shape}')
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
    avg_max_result = np.mean(result[:, :, -200:], axis=2)
    # print(avg_max_result)
    return avg_max_result

# x = np.random.randint(0, 9, (3, 4, 5))
# print(x)

# get_feature_npy(x)
