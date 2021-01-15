import feature
from glob import glob
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
import pickle
import micro

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
