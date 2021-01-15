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

if __name__ == '__main__':
    with open('RF_classifier.pkl', 'rb') as f:
        ml_classifier = pickle.load(f)
    file = 'lala.wav'
    t = threading.Thread(target=microphone.Monitor_MIC, args=(file,))
    t.start()
    for i in range(10):
        t.join()
        t = threading.Thread(target=microphone.Monitor_MIC, args=(file,))
        t.start()
    s, w = wavfile.read(file)
    w = w.astype(float)
    w = np.array(w)
    w = 32666.0/np.max(w)*w
    mfcc1 = librosa.feature.mfcc(y=w[:, 0], sr=s, n_mfcc=43)
    mfcc2 = librosa.feature.mfcc(y=w[:, 1], sr=s, n_mfcc=43)
    # ml_classifier.evaluate(get_test_data_iter())
    meta_label1 = ml_classifier.predict_meta(mfcc1)
    meta_label2 = ml_classifier.predict_meta(mfcc2)
    result1 = ml_classifier.predict_one(mfcc1,meta_label1)
    result2 = ml_classifier.predict_one(mfcc2,meta_label2)
    for i in range(ml_classifier.class_num):
        if result1[i] or result2[i]:
            print(ml_classifier.labels[i])
    print('end analyzing ' + file)