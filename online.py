import feature
from glob import glob
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
import pickle
import microphone

if __name__ == '__main__':
    file = 'oboe'
    microphone.Monitor_MIC(file)
    file += '.wav'
    w, s = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=w, sr=s, n_mfcc=40)
    print(mfcc.shape)

    mfcc = np.array([mfcc])
    res = feature.get_feature_npy(mfcc)
    # print(res)

    with open('regressor.pkl', 'rb') as f:
        ml_classifier = pickle.load(f)
    result = ml_classifier.regress_one(np.transpose(res))
    print(result)
