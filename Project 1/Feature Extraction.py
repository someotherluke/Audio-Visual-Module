# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:40:37 2023

@author: yad23rju

MAIN FEATURE EXTRACTION CODE. Reads .wav file in the folder 'Recordings' and outputs mfcc's to folder ''mfccs'
"""
#Mathematics modules
import numpy as np
from scipy.fftpack import dct    #discrete cosine transform library

#Plotting modules
import matplotlib.pyplot as plt

#Folder Access
import os
import glob 

#Audio modules
import soundfile as sf


#Set parameters

FRAME_LENGTH = 512 #Number of bins in each window (halved eventually). keep above 500 and even otherwise bin ranges are too small
OVER_LAP = 1/2 #Sets the overlap of our windows
STEP_LENGTH = int(FRAME_LENGTH * (OVER_LAP))
FILTER_NUMBER = 40 #Number of triangle peaks each window is being multiplied by

def mag_and_phase(section): #Calculate the fast forier transform and power (Variation of absolute magnitude)
    section = np.array(section)                    #Stores a section of speech as a numpy array

    #Takes hamming windows
    hamming = np.hamming(len(section))             #Create a hamming function based on section length
    scaled = np.multiply(hamming,section)          #Apply hamming window from function     
    
    #Performs fft on signal and calculates power
    fft_sample = np.fft.fft(scaled)                #Performs the Fast Forier transform on the hammed section os speech 
    absolute = np.abs(fft_sample)                  #Generates the absolute value of the signal (i.e. removes the complex component)
    power = absolute**2/len(section)               #Uses our absolute magnitude to generate a measure of signal power

    return power

def entire_utterance(speech): #Windows signal (Using above parameters) in overlapped steps (to proerly use hamming) and calls the fft function above to find the magnitude and power of the signal
    #Calculate number of frames total in the soundfile
    num_samples = len(speech) 
    num_frames = num_samples/FRAME_LENGTH * (FRAME_LENGTH/STEP_LENGTH)
    
    #ALWAYS ROUNDING DOWN TO REMOVE THE WINDOW WITH NO OR PARTIAL SIGNAL, AND SUBTRACT 1 TO REMOVE FINAL FRAME
    num_frames = int(np.floor(num_frames-1))

    all_power = np.array([])                                                                            #Create a power array for the loop
    energy_vals = np.array([])                                                                          #Create a energy array for the loop

    for i in range(0, num_frames):
        power = mag_and_phase(speech[STEP_LENGTH*i:STEP_LENGTH*i + FRAME_LENGTH])        #Find power from each window using funciton above

        energy = np.sum(speech[STEP_LENGTH*i:STEP_LENGTH*i + FRAME_LENGTH]**2)           #Sum the square of the window signal for the energy
        
        #ALWAYS TAKE HALF OF FFT (because of symmetry)
        all_power = np.append(all_power, power[0:int(FRAME_LENGTH/2)])                   #Append power array  
        energy_vals = np.append(energy_vals, energy)                                     #Append energy array
        
    return all_power, energy_vals

def mel_scale(frequency):                                                                #Creating a function from the real frequency to the mel frequency
    return 2595*np.log10(1+(frequency/700))

def inverse_mel(frequency):                                                              #Creating an inverse function from the mel frequency to true frequency
    return 700*(10**(frequency/2595) -1)
    
def plot_melscale(sample_frequency):
    #Take sampling rate and upper and lower bounds of desired frequency
    freq_high = sample_frequency/2                                                       #Maximum audible frequency 20khz, #So sample_frequency can be used provided smaple < 20khz, #Nyquist theorem means taking half of sample frequency
    freq_low = 20                                                                        #Sets our minimum frequency around the minimum audible frequency for humans (on the assumption that noise outside this isn't releveant to speech)
    
    #Convert frequency max/min to melscale
    mel_freq_high = mel_scale(freq_high)                                                 #Sets the maximum mel frequency
    mel_freq_low = mel_scale(freq_low)                                                   #Sets the minimum mel frequency
    
    mel_range = np.linspace(mel_freq_low, mel_freq_high, FILTER_NUMBER +2 )              #Evenly spaces the mel freqs between the high and low values - 2 added to filter_number so that the number of triangles is equal to our number of filters

    freq_range = inverse_mel(mel_range)                                                  #Find true frequencies from the mel frequencies
        
    bins = np.floor((FRAME_LENGTH + 1) * (freq_range / sample_frequency))                #Convert frequencies into bins - 1 addedto frame length so that the number of bins 
    
    fbank = np.zeros((FILTER_NUMBER, int(np.floor(FRAME_LENGTH / 2 ))))                  #Creates a 2 dimensional array with size FILTER_NUMBER (number of mfccs) by FRAME_LENGTH (divide by 2 because of the nyquist adjustment earlier)
    
    #Code is a variation of: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    #Define a left, middle and right bin
    for m in range(1, FILTER_NUMBER + 1):
        
        f_m_minus = int(bins[m - 1])  #left bin value
        f_m = int(bins[m])            #middle
        f_m_plus = int(bins[m + 1])   #right bin value
        
        #Iterate over the range of values between the left bin and the centre bin, adding to the fbank array
        for k in range(f_m_minus, f_m):
            #k here is the column number
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        #Iterate over the range of values between the centre bin and the right bin
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank  


def plot_filter(filter_index, window_index, power_vals, filter_bank, all_windows_filtered):#DEBUG
    #Plot the windowed power, filter and filter result
    fig, axs = plt.subplots(3)
    #axs = gs.subplots(sharex=True)
    axs[0].plot(all_windows_filtered[window_index,filter_index,0:int(FRAME_LENGTH/2)])
    axs[1].plot(power_vals[window_index])
    axs[2].plot(filter_bank[filter_index])
    axs[0].set_title("Result of filtering")
    axs[1].set_title("Original power spectrum")
    axs[2].set_title("Filter")
    fig.tight_layout()


def plot_dft(dft_mfccs):#DEBUG
    #Plot dft and spectrogram
    index = 10
    #Plot the dft just to check
    figgy, axes = plt.subplots(2)
    axes[0].plot(dft_mfccs[index])
    #plot spectrogram of mfcc
    #plt.figure(figsize=(15,5))
    axes[1].imshow(dft_mfccs, aspect='auto', origin='lower')
    #axes[2].specgram(dft_mfccs[3])


def plot_all_filter_banks(filter_bank):#DEBUG
    #Plot all of the filters in the filter bank
    for m in filter_bank:
         plt.plot(m)
         plt.xlim(0, FRAME_LENGTH/2 +3)    #Windowing our graph for presentation purposes
         plt.ylim(0,1)                     #As above

def plot_signals(r, file_name):#DEBUG
    fig10, ax10 = plt.subplots()
    ax10.plot(r)
    ax10.set_title(file_name)

def cepstral_lifter(mfcc):#NOT CURRENTLY IMPLEMENTED
    #'Ceptral lifting' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf and https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1165237
    #Smooths out formant peaks - not currently implemented
     num_filters = np.shape(mfcc)[1]
     i = 0
     lifter_coeff = 10 # arbitrary, good according to the source above
     lifter = 1 + (lifter_coeff/2)*np.sin(np.pi*i/lifter_coeff)
     for row in mfcc:
         lifter = 1 + (lifter_coeff/2)*np.sin(np.pi*i/lifter_coeff)
         row *= lifter
         i+=1
     return mfcc
 
def velocity_acceleration(dft_mfccs):
    #Begin by calculating velocities
    velocity_vals = np.array([])
    for i in range(0, dft_mfccs.shape[0]):      #Iterate over each frame of signal
        for j in range(1,dft_mfccs.shape[1]-1):     #Iterate over each mfcc, missing out first and last mfcc
            velocity = dft_mfccs[i, j+1] - dft_mfccs[i,j-1]     #Calculate velocity
            velocity_vals = np.append(velocity_vals, velocity)
    velocities_ordered = velocity_vals.reshape(dft_mfccs.shape[0], dft_mfccs.shape[1]-2)    #Reshape velocity into shape [FRAMES, MFCCS-2]
    
    #Before adding velocities to mfcc array, calculate accelerations
    acceleration_vals = np.array([])
    for i in range(0, velocities_ordered.shape[0]):      #Iterate over each velocity row corresponding to a frame of signal
        for j in range(1, velocities_ordered.shape[1]-1):     #Iterate over each velocity value, missing out first and last one
            acceleration = velocities_ordered[i, j+1] - velocities_ordered[i,j-1]     #Calculate acceleration
            acceleration_vals = np.append(acceleration_vals, acceleration)
    acceleration_ordered = acceleration_vals.reshape(velocities_ordered.shape[0], velocities_ordered.shape[1]-2)    #Reshape into shape [FRAMES, MFCCS-4]
    
    
    velocities_ordered = velocities_ordered.transpose()     #Transpose ordered velocities so it's easier to add each column to mfcc data
    acceleration_ordered = acceleration_ordered.transpose()     #Transpose ordered acceleration so it's easier to add each column to mfcc data
    for i in range(0, velocities_ordered.shape[0]):     #Iterate over each mfcc value
        dft_mfccs = np.insert(dft_mfccs, dft_mfccs.shape[1], velocities_ordered[i], axis=1)     #Append each column to the end of the mfcc array
        
    for i in range(0, acceleration_ordered.shape[0]):     #Iterate over each mfcc value
        dft_mfccs = np.insert(dft_mfccs, dft_mfccs.shape[1], acceleration_ordered[i], axis=1)    #Append each column to the end of the mfcc array
    return dft_mfccs
    
def save_to_file(file_name, mfcc): #Let's us save proccessed files to speed up using data for DNN
    #Saves file to the folder 'mfccs' in directory
    #allow pickle! To see why you may change this; https://stackoverflow.com/questions/41696360/numpy-consequences-of-using-np-save-with-allow-pickle-false
    directory = os.path.join('mfccs', file_name)
    np.save(directory, mfcc, allow_pickle=True)
    
    
def add_noise(signal, SNR):#NOT CURRENTLY IMPLEMENTED
    noise = np.random.normal(0,0.1, len(signal)) #Creates normally distributed noise (white noise)
    
    #Calculate the powers for the noise
    noise_power = np.mean(noise**2)

    speech_power = np.mean(signal**2)
    
    #Calculate scaling factor for desired SNR
    alpha = np.sqrt((speech_power/noise_power)*10**-(SNR/10))
    
    #Scale noise by alpha
    noise_power_alpha = (noise * alpha)**2
    
    #Add noise to the signal
    speech_power_noised = noise_power_alpha + signal
    
    fig29, ax29 = plt.subplots(3)
    ax29[0].plot(signal)
    ax29[1].plot(speech_power_noised)
    ax29[2].plot(noise)
    #noise_power_new = np.mean(noise_power_alpha)
        
        
def main(file_dir, file_name):
    r, fs2 = sf.read(file_dir, dtype='float32')                        #Read in file
    sample_frequency = 16000                                           #Somewhat arbitrary
    
    r = r[3000:-115000]                                                #Take window of signal initially to remove long periods of silence from recording

    #add noise to the signal, varied by the SNR coefficient
    #add_noise(r, 0)

    filter_bank = plot_melscale(sample_frequency)                      #Calculate the filterbanks, with the same sampling frequency as the FFT
    
    all_power_vals, energy_vals = entire_utterance(r)                  #Extract the power (and energy) values from each window of signal and add to big array
    
    
    power_vals_shaped = np.reshape(all_power_vals, 
                                   (int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, int(FRAME_LENGTH/2)))    #Reshape the power array such that each row corresponds to 1 frame of the FFT power values
    
    #print(power_vals_shaped.shape, "POWER_VALS SHAPE")                #DEBUG PRINT
    
    #Iterate over the power array and multiply by each mel filter triangle and sum up the result
    summed_filtered_array = np.array([])
    power_vals_filtered = np.array([])
    for row in power_vals_shaped:
        for filter_peak in filter_bank:
            
            summed_filtered = np.matmul(row, filter_peak)
            summed_filtered_array = np.append(summed_filtered_array, summed_filtered)
            
            """#UNCOMMENT TO SEE GRAPHS OF POWER, FILTER AND FILTER RESULT
            adjusted_power = np.multiply(row, filter_peak)
            summed_filtered = np.sum(adjusted_power)
            power_vals_filtered = np.append(power_vals_filtered,adjusted_power)
    
    #RESHAPE POWER ARRAY INTO [FRAME, FILTER, BIN] just to check everything is correct
    all_windows_filtered = np.reshape(power_vals_filtered, (power_vals_shaped.shape[0], FILTER_NUMBER, int(FRAME_LENGTH/2)))
    filter_index = 23
    window_index = 1
    plot_filter(filter_index, window_index, power_vals_shaped, filter_bank, all_windows_filtered)
    #######################
    """
    
    #Reshape the summed array into shape [WINDOW, FILTER]
    mfccs = summed_filtered_array.reshape(int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, FILTER_NUMBER)
    #print(mfccs.shape, "MFCC SHAPE") #DEBUG Check shape is correct
    
    #Take log of mfccs
    logged_mfccs = np.log10(mfccs)

    #PERFORM DFT ON LOGGED VALUES
    #Discard some MFCC's as the others represent fast changes in signal and don't contribute to recognition. 'Truncation' in the powerpoints
    keep_number = 39
    dft_mfccs = dct(logged_mfccs, axis=1, norm='ortho')[:, 0 : keep_number]   
    
    #Calculate velocity and acceleration and append to the dft_mfccs
    dft_mfccs = velocity_acceleration(dft_mfccs)
            
    #add energy values to the end of each mfcc
    dft_mfccs = np.insert(dft_mfccs,dft_mfccs.shape[1],energy_vals,axis=1)

    #TRANSPOSE MATRIX SO IT FITS INTO MODEL SEQUENTIALLY
    dft_mfccs = dft_mfccs.transpose()
    
    #save file to .npy in mfccs folder in directory
    save_to_file(file_name, dft_mfccs)

    #--------------------------------------------------------------------------
    #ALL EXTRA DEBUGGING GRAPHS BELOW
    #--------------------------------------------------------------------------

    #plot_dft(dft_mfccs)                 #Plots spectogram of processed signal

    #plt.plot(r)                         #Plots soundfile

    #plot_all_filter_banks(filter_bank)  #Plots all of our filter banks 

    #plot_signals(r, file_name)          #Plots graphs of signal for files being used

# for testing
#folder = debugging, for debugging
#folder = Recordings for actual test

def Feature_extraction(folder):
    folder_name =  folder + '/*.wav'
    for audio_file in sorted(glob.glob(folder_name)):
        #Remove file directory and .wav for naming purposes
        audio_file_name = audio_file[11:-4]
        
        main(audio_file, audio_file_name)
        
        print(audio_file_name, 'done')
    
    print('Done entire data set')

#TODO vary feature extraction methods:
#ENERGY, TEMPORAL DERIVATIVES: VELOCITY, ACCELERATION
#'SPECTRAL LIFTERING' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
