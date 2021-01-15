import numpy as np
import librosa
from glob import glob
from random import shuffle, sample

def sample_multiple(l: list, num: int):
    result = []
    while num > len(l):
        result += sample(l, len(l))
        num -= len(l)
    result += sample(l, num)
    return result

def detect_strength(dataset_path: str):
    '''
    label_dict : the key is string of label , word is (is_positive_class, number)
    returns:
        waves: numpy matrix, each row represents a wave, each column represents a time sample
        samples: the sample rate of waves
        true_list: whether the wave is positive class
    '''
    all_file = []
    all_file += sample_multiple(glob(f'{dataset_path}/*.wav'), 10)

    waves = []
    samples = []
    for file in all_file:
        w, s = librosa.load(file)
        # print(w.shape)
        waves.append(w)
        samples.append(s)
        print(np.max(np.array(w)))
    waves = np.array(waves)
    samples = np.array(samples)

if __name__ == '__main__':
    detect_strength('./')