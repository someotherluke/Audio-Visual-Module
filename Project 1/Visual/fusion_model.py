
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
from tensorflow.keras.layers import Dense, Activation,Flatten, Conv2D, InputLayer, MaxPooling2D, Input, concatenate, BatchNormalization, Average, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn import metrics

name_list_old = ["Ben", "Bonney","Carlos", "Charlie",  "Chiedozie","El","Ethan",
             "Francesca","Jack","Jake","James","Lindon","Marc","Nischal",
             "Robin","Ryan","Sam","Seth","William","Yubo"]
name_list = ["B","Bo","C","Ch","Cd","El","Et",
             "Fr","Jk","Je","Jm","L","M","N",
             "Ro","Ry","Sa","Se","W","Y"]

confusion_matrices = np.array([])
accuracies_matrix = np.array([])

def audio_model(max_frames, length):
    numClasses=20 
    model=Sequential()
    model.add(InputLayer(input_shape=(length, max_frames, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dense(32))
    
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    
    
    return model

#THIS IS BAD PROBABLY - NEEDS IMPROVEMENT
def visual_model(max_frames, length):
    num_classes = 20

    model = Sequential()

    model.add(InputLayer(input_shape=(length, max_frames, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))


    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.1)) #GET RID OF DROPOUTS PROBS

    model.add(Dense(num_classes, activation='softmax'))
    
    
    return model

#%%
def fusion_model(max_frames_visual, length_visual, max_frames_audio, length_audio):
    #Initiate models
    visual_input = Input(shape=(length_visual, max_frames_visual, 1))
    audio_input = Input(shape=(length_audio, max_frames_audio, 1))

    visual_model_instance = visual_model(max_frames_visual, length_visual)
    audio_model_instance = audio_model(max_frames_audio, length_audio)

    # Remove the softmax layer from the original models
    visual_model_instance.pop()
    audio_model_instance.pop()

    # Allow the weights of the original models to be updated
    for layer in visual_model_instance.layers:
        layer.trainable = True
    for layer in audio_model_instance.layers:
        layer.trainable = True

    # Get the output tensors of the original models
    visual_output = visual_model_instance(visual_input)
    audio_output = audio_model_instance(audio_input)

    # Add Dense layer to make the shapes compatible for averaging
    visual_output = Dense(20, activation='softmax')(Flatten()(visual_output))
    audio_output = Dense(20, activation='softmax')(Flatten()(audio_output))

    
    # Average the predictions from both models
    fusion_output = Average()([visual_output, audio_output])

    # Add a new softmax layer for the fusion model
    fusion_output = Dense(20, activation='softmax')(fusion_output)

    # Create the fusion model
    fusion_model = Model(inputs=[visual_input, audio_input], outputs=fusion_output)

    return fusion_model


#%%

total_visual_data = []
labels_visual = []

size_visual = []
max_frames_visual = 0


for visual_file in sorted(glob.glob('visual_only/*.npy')):#visuals_new for new total_visual_data
    visual_data = np.load(visual_file,allow_pickle=True)
    #print(visual_data.shape[1])
    if visual_data.shape[1] > max_frames_visual:
        #print(visual_data.shape[1])
        max_frames_visual = visual_data.shape[1]

print('Max frames:', max_frames_visual)

for visual_file in sorted(glob.glob('visual_only/*.npy')):
    visual_data = np.load(visual_file,allow_pickle=True)
    visual_data = np.pad(visual_data, ((0,0), (0, max_frames_visual-visual_data.shape[1])))

    total_visual_data.append(visual_data)
    
    #Creates variable '0_lucas_10' etc. not in numerical order(referring to last digit)!
    stemFilename = (Path(os.path.basename(visual_file)).stem)

    label = stemFilename.split('_')
    labels_visual.append(label[0])



labels_audio = []
total_audio_data = []

max_frames_audio = 0


for mfcc_file in sorted(glob.glob('mfccs/*.npy')):#mfccs_new for new total_audio_data
    mfcc_data = np.load(mfcc_file,allow_pickle=True)
    #print(mfcc_data.shape[1])
    if mfcc_data.shape[1] > max_frames_audio:
        #print(mfcc_data.shape[1])
        max_frames_audio = mfcc_data.shape[1]

print('Max frames:', max_frames_audio)

for mfcc_file in sorted(glob.glob('mfccs/*.npy')):
    mfcc_data = np.load(mfcc_file,allow_pickle=True)
    mfcc_data = np.pad(mfcc_data, ((0,0), (0, max_frames_audio-mfcc_data.shape[1])))

    total_audio_data.append(mfcc_data)
    
    #Creates variable '0_lucas_10' etc. not in numerical order(referring to last digit)!
    stemFilename = (Path(os.path.basename(mfcc_file)).stem)

    label = stemFilename.split('_')
    labels_audio.append(label[0])
    



















#LLOOOONG 1D array of all the numbers
labels_visual = np.array(labels_visual)

labels_audio= np.array(labels_audio)

total_visual_data = np.array(total_visual_data)
total_audio_data = np.array(total_audio_data)
#Creates a (420,20) array from the long 1d array. 10 columns. Each corresponding to 1 category. 
#1's for the assigned category, 0 for not
LE=LabelEncoder()
labels_visual=to_categorical(LE.fit_transform(labels_visual))
labels_audio=to_categorical(LE.fit_transform(labels_audio))


#%%

#SPLIT DATA FOR VISUAL - ignore testing data for now 
x_train_visual, x_tmp_visual, y_train_visual, y_tmp_visual = train_test_split(total_visual_data,labels_visual, test_size=0.2, random_state=0)

#x_val_visual, x_test_visual, y_val_visual, y_test_visual = train_test_split(x_tmp_visual, y_tmp_visual,test_size=0.5, random_state=0)


#SPLIT DATA FOR AUDIO
x_train_audio, x_tmp_audio, y_train_audio, y_tmp_audio = train_test_split(total_audio_data,labels_audio, test_size=0.2, random_state=0)

#x_val_audio, x_test_audio, y_val_audio, y_test_audio = train_test_split(x_tmp_audio, y_tmp_audio,test_size=0.5, random_state=0)


#Set dimensions
mfcc_length = mfcc_data.shape[0]
visual_length = visual_data.shape[0]


#%%
#SET INSTANCES OF AUDIO AND VISUAL MODELS
visual_model_instance = visual_model(max_frames_visual, visual_length)
audio_model_instance = audio_model(max_frames_audio, mfcc_length)

#SET THE FUSION MODEL
fusion_model_instance = fusion_model(max_frames_visual, visual_length, max_frames_audio, mfcc_length)

# Compile the fusion model with an appropriate optimizer and loss function
fusion_model_instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Concatenate the labels for visual and audio modalities
#y_train_fusion = np.concatenate([y_train_visual, y_train_audio], axis=1)
#y_val_fusion = np.concatenate([y_val_visual, y_val_audio], axis=1)
#Stop the network if loss isnt minimising
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.0001,
    patience=5,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=12
)
# Train the fusion model 
num_epochs = 300
batch_size = 4
learning_rate=0.00001
history = fusion_model_instance.fit(
    [x_train_visual, x_train_audio],
    y_train_visual,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=([x_tmp_visual, x_tmp_audio], y_tmp_visual), callbacks=[callback])




fusion_model_instance.save_weights('FUSED_WEIGHTS.h5')

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
"""
predicted_probs=fusion_model_instance.predict(X_test,verbose=0)
predicted=np.argmax(predicted_probs,axis=1)
actual=np.argmax(y_test,axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')
confusion_matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1), predicted)
"""
