import feature
from glob import glob
import librosa
from librosa.feature.spectral import mfcc
import numpy as np
import pickle
import micro

micro.Monitor_MIC('lala')
file = 'lala.wav'
print(file)
w, s = librosa.load(file)
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
