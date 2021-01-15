import feature
from glob import glob
import librosa
from dataset import get_test_data_iter
from librosa.feature.spectral import mfcc
from scipy.io import wavfile
import numpy as np
import pickle
import microphone
import threading

micro.Monitor_MIC('lala')
w, s = librosa.load('./lala.wav')
w = 0.8/np.max(np.array(w))*np.array(w)
mfcc = librosa.feature.mfcc(y=w, sr=s, n_mfcc=40)
print(mfcc.shape)
#np.save(file.replace('wav', 'npy'), mfcc)

mfcc = np.array([mfcc])
res = feature.get_feature_npy(mfcc)
print(res)

with open('classifier.pkl', 'rb') as f:
    ml_classifier = pickle.load(f)
result = ml_classifier.predict_one(np.transpose(res))
print(result)
