# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:17:13 2023

@author: yad23rju
"""

import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import glob 
from pathlib import Path
import os

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten, Conv2D, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics

name_list_old = ["Ben", "Bonney","Carlos", "Charlie",  "Chiedozie","El","Ethan",
             "Francesca","Jack","Jake","James","Lindon","Marc","Nischal",
             "Robin","Ryan","Sam","Seth","William","Yubo"]
name_list = ["Be","Bo","Crl","Chr","Cd","El","Et",
             "Fr","Jk","Je","Jm","L","M","N",
             "Ro","Ry","Sa","Se","W","Y"]


def create_model(max_frames, length):
    numClasses=20 
    model=Sequential()
    model.add(InputLayer(input_shape=(length, max_frames, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    
    return model


data = []
labels = []

size_mfcc = []
max_frames = 0


for mfcc_file in sorted(glob.glob('mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file)
    #print(mfcc_data.shape[1])
    if mfcc_data.shape[1] > max_frames:
        #print(mfcc_data.shape[1])
        max_frames = mfcc_data.shape[1]

    

print(max_frames)

for mfcc_file in sorted(glob.glob('mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0,0), (0, max_frames-mfcc_data.shape[1])))

    data.append(mfcc_data)
    
    #Creates variable '0_lucas_10' etc. not in numerical order(referring to last digit)!
    stemFilename = (Path(os.path.basename(mfcc_file)).stem)

    label = stemFilename.split('_')
    labels.append(label[0])

#LLOOOONG 1D array of all the numbers
labels = np.array(labels)

data = np.array(data)
    
#Creates a (500,10) array from the long 1d array. 10 columns. Each corresponding to 1 category. 
#1's for the assigned category, 0 for not
LE=LabelEncoder()
labels=to_categorical(LE.fit_transform(labels))

#data and labelling is correct because the data and labels are in the same order
X_train, X_tmp, y_train, y_tmp = train_test_split(data,labels, test_size=0.3, random_state=0)

X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,test_size=0.5, random_state=0)
#Stop the network if loss isnt minimising
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=2,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=4    
    )
#REMEMBER TO CREATE_MODEL EVERYTIME (SEE LAB)

mfcc_length = mfcc_data.shape[0]

#for i in range(adygaudwvb):    
model = create_model(max_frames, mfcc_length)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
model.summary()

num_epochs = 12
num_batch_size = 32




history = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=num_batch_size, epochs=num_epochs,verbose=1, callbacks=[callback])
model.save_weights('digit_classification.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

predicted_probs=model.predict(X_test,verbose=0)
predicted=np.argmax(predicted_probs,axis=1)
actual=np.argmax(y_test,axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')
confusion_matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =confusion_matrix, display_labels=name_list)
#cm_display = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted, labels=name_list, xticks_rotation='vertical')
cm_display.plot()

"""
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix)
ax.set_xticks(np.arange(len(name_list)))
ax.set_yticks(np.arange(len(name_list)))
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels(name_list, rotation = 90)
ax.set_yticklabels(name_list)
fig.tight_layout()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
"""
