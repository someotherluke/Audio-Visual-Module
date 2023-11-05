# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:48:25 2023

@author: haw17cju
"""
import numpy as np
import soundfile as sf                               #for testing purposes 
from VideoRec_demo import record_audio
from Feature_Extraction import extract_features
from Feature_Extraction import main
from Speech_DNN import create_model

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt                     #for testing purposes 


if __name__ == '__main__':
    record_audio()

    extract_features()

    name_list_old = ["Ben", "Bonney","Carlos", "Charlie",  "Chiedozie","El","Ethan",
             "Francesca","Jack","Jake","James","Lindon","Marc","Nischal",
             "Robin","Ryan","Sam","Seth","William","Yubo"]
    
    #record and load mfcc:
    seconds = 3
    fs= 44100
    max_frames = 700 #Be careful with this!
    #print("RECORDING...")S
    #r = sd.rec(seconds * fs, samplerate=fs, channels=1)
    #sd.wait()
    #print('PROCESSING...')
    r, fs = sf.read('Chiedozie_D.wav') #IMPORT TEST WAV HERE, CHANGE THE WAY RECORD AUDIO SAVE A FILE MAYBE MAKE A RECORD AUDIO_DEMO
    r= r.squeeze()
    test_name = main(r)
    test_name = np.pad(test_name, ((0,0), (0, max_frames-test_name.shape[1])))
    #max_frames = test_name.shape[1]
    mfcc_length = test_name.shape[0]
    
    model = create_model(max_frames, mfcc_length)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
    model.load_weights('possibly_good_batch_sixteen.h5') #BEST MODEL NAME GOES HERE!!!!!
    
    test_name = tf.reshape(test_name, (-1, mfcc_length, max_frames, 1))
    predict = model.predict(test_name)
    print(predict)
    predict = np.argmax(predict)
    
    print(name_list_old[predict])