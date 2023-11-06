# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:17:13 2023

@author: Dan and Luke
"""

import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import glob 
from pathlib import Path
import os

#ML 
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten, Conv2D, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics

#combinations
import itertools

#Save a list to txt file
def save_file(name, array):
    file = open(name,'w')
    for item in array:
    	file.write(item+"\n")
    file.close()

#Use this to quickly sort the accuracies list into size order
def sort_file_by_numbers(input_file, output_file):
    # Read the contents of the input file and split it into lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Split each line into a tuple (number, string) and convert the number to an integer
    data = [(line.split(',')[0], line) for line in lines]

    # Sort the data by numbers in descending order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)

    # Write the sorted data to the output file
    with open(output_file, 'w') as f:
        for _, line in sorted_data:
            f.write(line)



#Create a plot of the confusion matrix
def save_confusion_matrix_as_image(confusion_matrix, class_labels, output_folder, batch_lr):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    num_classes = len(class_labels)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_labels, rotation=90)
    plt.yticks(tick_marks, class_labels)

    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #Save to folder
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    picture_name = 'Confusion Matrix'+batch_lr
    file_path = os.path.join(output_folder, picture_name)
    plt.savefig(file_path)

#Define the possible variables
#REPLACE THIS WITH YOUR DESIRED ITERATION PARAMS-------------------------------
possible_filter_types = ["Mel"]
possible_filter_shapes = ["Square"]
possible_overlaps = [1/3]          #1/4 not included for time
possible_frame_sizes = [1024]      #2048 not included for time
possible_energy_included = [1]          #1 for dan, 0 for luke
possible_temporal_included = [1]        #1 for dan, 0 for luke
possible_no_bins = [64]
possible_truncate_no_bin = [16]      #8 not included for time

#Generate a matrix of all possible combinations
combinations = list(
    itertools.product(
        possible_filter_types,
        possible_filter_shapes,
        possible_overlaps,
        possible_frame_sizes,
        possible_energy_included,
        possible_temporal_included,
        possible_no_bins,
        possible_truncate_no_bin
    )
)

#Iterate through the combinations
def combination():
    file_paths = []
    for combination in combinations:
        try:
            filter_type = combination[0]
            filter_shape = combination[1]
            overlap = combination[2]
            frame_sizes = combination[3]
            energy_included = combination[4]
            temporal_included = combination[5]
            no_bins = combination[6]
            truncate_no_bins = combination[7]
            file_path= filter_type[0] + "_" + filter_shape[0]
            file_path += "_ol_" + str(overlap).replace(".","_") + "_fr_" + str(frame_sizes)+"_en_"+ str(energy_included)[0]+ "_temp_"+ str(temporal_included)[0]+ "_bins_"+ str(no_bins)+ "_trunc_bin_"+ str(truncate_no_bins)
            file_paths.append(file_path)
            print(filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)
        except:
            print("error occured with settings", filter_type, filter_shape, overlap, frame_sizes, energy_included, temporal_included, no_bins, truncate_no_bins)
    return file_paths

        
#Create list of all the combinations, shuffled
file_paths = combination()

#shuffle path
np.random.shuffle(file_paths)

#save list to file so we know the order
save_file('folder_list.txt', file_paths)

accuracies_matrix = np.array([])

#Define model
def create_model(max_frames, length):
    numClasses=20 
    model=Sequential()
    model.add(InputLayer(input_shape=(length, max_frames, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    
    return model

#name lists
name_list_old = ["Ben", "Bonney","Carlos", "Charlie",  "Chiedozie","El","Ethan",
             "Francesca","Jack","Jake","James","Lindon","Marc","Nischal",
             "Robin","Ryan","Sam","Seth","William","Yubo"]
name_list = ["B","Bo","C","Ch","Cd","El","Et",
             "Fr","Jk","Je","Jm","L","M","N",
             "Ro","Ry","Sa","Se","W","Y"]

accuracies = []
i=0
#iterate over all the files and put through the network
for file in file_paths:
    i+=1
    #print file and number so know how far along it is
    print(file, i)
    data = []
    labels = []
    
    size_mfcc = []
    max_frames = 0
    
    #Find max frames
    for mfcc_file in sorted(glob.glob(file +'/*.npy')):
        mfcc_data = np.load(mfcc_file)
        #print(mfcc_data.shape[1])
        if mfcc_data.shape[1] > max_frames:
            #print(mfcc_data.shape[1])
            max_frames = mfcc_data.shape[1]
    
    #important for testing later
    print('Max frames:', max_frames)
    
    #Sort and append mfccs
    for mfcc_file in sorted(glob.glob(file +'/*.npy')):
        mfcc_data = np.load(mfcc_file)
        mfcc_data = np.pad(mfcc_data, ((0,0), (0, max_frames-mfcc_data.shape[1])))
    
        data.append(mfcc_data)
        
        #Creates variable '0_lucas_10' etc. not in numerical order(referring to last digit)!
        stemFilename = (Path(os.path.basename(mfcc_file)).stem)
    
        label = stemFilename.split('_')
        labels.append(label[0])
    
    #Long 1D array of all the numbers
    labels = np.array(labels)
    
    data = np.array(data)
        
    #Creates a (500,10) array from the long 1d array. 10 columns. Each corresponding to 1 category. 
    #1's for the assigned category, 0 for not
    LE=LabelEncoder()
    labels=to_categorical(LE.fit_transform(labels))
    
    #data and labelling is correct because the data and labels are in the same order
    X_train, X_tmp, y_train, y_tmp = train_test_split(data,labels, test_size=0.3, random_state=0)
    
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,test_size=0.5, random_state=0)
        
    #For model creation
    mfcc_length = mfcc_data.shape[0]
    
    #HYPER PARA VARYING
    learning_rates = [0.001]
    batch_sizes = [16]

    #Iterate over the varying params
    for learning_rate in learning_rates:
        for num_batch_size in batch_sizes:
                #REMEMBER TO CREATE_MODEL EVERYTIME (SEE LAB)
                model = create_model(max_frames, mfcc_length)
                model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=Adam(learning_rate=learning_rates))
                model.summary()
                num_epochs = 25
                #Stop the network if val_accuracy isnt increasing
                callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    min_delta=0.01,
                    patience=3,
                    verbose=1,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True,
                    start_from_epoch=2
                )
    
                #Train model
                history = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=num_batch_size, epochs=num_epochs,verbose=1, callbacks=[callback])
                
                #Calc accuracy 
                predicted_probs=model.predict(X_test,verbose=0)
                predicted=np.argmax(predicted_probs,axis=1)
                actual=np.argmax(y_test,axis=1)
                accuracy = metrics.accuracy_score(actual, predicted)
                
                #Save accuracy to array
                accuracies.append((str(accuracy) + ',' + file))
                print(f'Accuracy: {accuracy * 100}%')
                
                #Determine where to save the files based on how good the accuracy is
                if accuracy >= 0.80:
                    accuracy_folder = 'Over_80'
                elif accuracy >= 0.70:
                    accuracy_folder = 'Over_70'
                elif accuracy >= 0.60:
                    accuracy_folder = 'Over_60'
                else:
                    accuracy_folder = 'Under_60'
                save_path = os.path.join('CNN_results', accuracy_folder, file)
                lr_batch_specification = '_LR_' + str(learning_rate).replace(".", "") + '_B_'+ str(num_batch_size)
                #Create confusion
                confusion_matrix = metrics.confusion_matrix(actual, predicted)
                accuracies_matrix = np.append(accuracies_matrix, accuracy)
                #Save confusion and create folder for it
                save_confusion_matrix_as_image(confusion_matrix, name_list_old, save_path,lr_batch_specification)
                
                #Save weights
                model.save_weights(save_path+'/'+file+'weights.h5')
                
                #plot model accuracy over time 
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'])
                ax.plot(history.history['val_accuracy'])
                ax.set_title('Model Accuracy - LR: {} BS:{}'.format(learning_rates[0], num_batch_size)) 
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Epoch')
                ax.legend(['Train', 'Validation'], loc='upper left')
                plt.savefig(save_path+'/'+'Accuracy graph'+lr_batch_specification)
                plt.show()
                
                #plot loss function
                fig2, ax2 = plt.subplots()
                ax2.plot(history.history['loss'])
                ax2.plot(history.history['val_loss'])
                ax2.set_title('Model Loss - LR: {} BS:{}'.format(learning_rates[0], num_batch_size)) 
                ax2.set_ylabel('Loss')
                ax2.set_xlabel('Epoch')
                ax2.legend(['Train', 'Validation'], loc='upper left')
                plt.savefig(save_path+'/'+'Loss graph'+lr_batch_specification)
                plt.show()
                
                save_file('CNN_results/accuracies'+lr_batch_specification+'.txt', accuracies)

print('Done')
