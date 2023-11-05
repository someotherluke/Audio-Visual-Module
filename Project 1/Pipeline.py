# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:48:25 2023

@author: haw17cju
"""

from VideoRec import record_audio
from Feature Extraction import extract_features


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
    #print("RECORDING...")
    #r = sd.rec(seconds * fs, samplerate=fs, channels=1)
    #sd.wait()
    #print('PROCESSING...')
    r, fs = sf.read('Chiedozie_D.wav')
    r= r.squeeze()
    test_name = main(r)
    test_name = np.pad(test_name, ((0,0), (0, max_frames-test_name.shape[1])))
    #max_frames = test_name.shape[1]
    mfcc_length = test_name.shape[0]
    
    model = create_model(max_frames, mfcc_length)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
    model.load_weights('possibly_good_batch_sixteen.h5')
    
    test_name = tf.reshape(test_name, (-1, mfcc_length, max_frames, 1))
    predict = model.predict(test_name)
    print(predict)
    predict = np.argmax(predict)
    
    print(name_list_old[predict])