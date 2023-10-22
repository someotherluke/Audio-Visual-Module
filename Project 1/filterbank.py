# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:40:37 2023

@author: yad23rju

MAIN FEATURE EXTRACTION CODE. Reads .wav file in the folder 'Recordings' and outputs mfcc's'
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy.fftpack import dct
import glob 

#Set params
FRAME_LENGTH = 1024 #Number of bins in each window (halved eventually). IDEALLY KEEP AS A RESULT OF 2^x
STEP_LENGTH = FRAME_LENGTH * (1/2)
FILTER_NUMBER = 40 #Number of triangle peaks each window is being multiplied by

def mel_scale(data):
    return 2595*np.log10(1+(data/700))

def inverse_mel(data):

    return 700*(10**(data/2595) -1)
    
def plot_melscale(sample_frequency):
    #Take sampling rate and upper and lower bounds of desired frqeuency
    freq_high = sample_frequency/2 #Nyquist
    
    freq_low = 0 
    
    mel_freq_high = mel_scale(freq_high)

    mel_freq_low = mel_scale(freq_low)
    
    #Spread out the mel freqs between the high and low value
    mel_range = np.linspace(mel_freq_low, mel_freq_high, FILTER_NUMBER +2 )

    #find frequency of these mel numbers
    freq_range = inverse_mel(mel_range)
        
    #convert frequencies into bins
    bins = np.floor((FRAME_LENGTH + 1) * freq_range / sample_frequency)

    #set filterbank array to be filled
    fbank = np.zeros((FILTER_NUMBER, int(np.floor(FRAME_LENGTH / 2 ))))
    #code is a variation of: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    for m in range(1, FILTER_NUMBER + 1):
        #left bin value
        f_m_minus = int(bins[m - 1])  
        #middle
        f_m = int(bins[m])
        #right
        f_m_plus = int(bins[m + 1])
        #Iterate over the range of values between the left bin and the centre bin
        for k in range(f_m_minus, f_m):
            #k here is the column number
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        #Iterate over the range of values between the centre bin and the right bin
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    
    return fbank  

def mag_and_phase(section):
    #Calculate the fft and power
    
    section_squeezed = section.squeeze()
    section = np.array(section)
    #Perform and apply hamming window
    hamming = np.hamming(len(section))
    scaled = np.multiply(hamming,section) 
    fft_sample = np.fft.fft(scaled) 
    absolute = np.abs(fft_sample)
    #Take 'power' of spectrum
    power = absolute**2/len(section)

    return power

def entire_utterance(speech):
    #Window signal in steps and call the fft function above to find the mag and power
    num_samples = len(speech)
    
    step_length = STEP_LENGTH
    
    step_length = int(step_length)
    
    num_frames = num_samples/FRAME_LENGTH * (FRAME_LENGTH/step_length)
    
    num_frames = int(np.floor(num_frames)) #ALWAYS ROUND DOWN TO REMOVE THE WINDOW WITH NO SIGNAL
    
    all_power = np.array([])
    
    for i in range(0, num_frames-1):#miss out last digit because the last frame will have little audio anyway

        power = mag_and_phase(speech[step_length*i:step_length*i + FRAME_LENGTH])
        
        all_power = np.append( all_power, power[0:int(FRAME_LENGTH/2)]) # TAKE HALF OF FFT becuase symmetry
        
    return all_power

def plot_filter(filter_index, window_index, power_vals, filter_bank, all_windows_filtered):
    #Plot the windowed power, filter and filter result for debugging purposes
    fig, axs = plt.subplots(3)
    #axs = gs.subplots(sharex=True)
    axs[0].plot(all_windows_filtered[window_index,filter_index,0:int(FRAME_LENGTH/2)])
    axs[1].plot(power_vals[window_index])
    axs[2].plot(filter_bank[filter_index])
    axs[0].set_title("Result of filtering")
    axs[1].set_title("Original power spectrum")
    axs[2].set_title("Filter")
    fig.tight_layout()


def plot_dft(dft_mfccs):
    #Plot dft and spectrogram for debugging
    index = 10
    #Plot the dft just to check
    figgy, axes = plt.subplots()
    axes.plot(dft_mfccs[index])
    #plot spectrogram of mfcc
    plt.figure(figsize=(15,5))
    axes.imshow(dft_mfccs, aspect='auto', origin='lower');

    
def cepstral_lifter(mfcc):
    #'Ceptral lifting' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
    #Doesnt work currently
     num_filters = np.shape(mfcc)[1]
     i = np.linspace(0, num_filters)
     lifter_coeff = 22 # arbitrary, good according to the source above
     lifter = 1 + (lifter_coeff/2)*np.sin(np.pi*i/lifter_coeff)
     mfcc *= lifter
    
def save_to_file(file_name, mfcc):
    #allow pickle!
    directory = os.path.join('mfccs', file_name)
    np.save(directory, mfcc, allow_pickle=True)

def main(file_dir, file_name):

    r, fs2 = sf.read(file_dir, dtype='float32')
    
    sample_frequency = fs2 # Let the sound file decide frequency FOR NOW
    
    r = r[10000:-100000] # Take window of signal initially to remove long periods of silence   
    
    #CALC THE FILTER BANKS. GET THE POWER SPECTRUM OF EACH WINDOW AND MULTIPLY THEM
    filter_bank = plot_melscale(sample_frequency)
    
    all_power_vals = entire_utterance(r)
    
    #Reshape the power array such that each row corresponds to 1 frame of the FFT power values
    power_vals_shaped = np.reshape(all_power_vals, (int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, int(FRAME_LENGTH/2)))
    
    print(power_vals_shaped.shape, "POWER_VALS SHAPE")
    
    summed_filtered_array = np.array([])
    
    power_vals_filtered = np.array([])
    
    for row in power_vals_shaped:
        
        for filter_peak in filter_bank:
            
            summed_filtered = np.matmul(row, filter_peak)
            
            summed_filtered_array = np.append(summed_filtered_array, summed_filtered)
    
    mfccs = summed_filtered_array.reshape(int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, FILTER_NUMBER)
    print(mfccs.shape, "MFCC SHAPE")
    
    #TAKE LOG
    logged_mfccs = np.log10(mfccs)

    #PERFORM DFT ON LOGGED VALUES
    #Only take 1-13  of the MFCC's as the others represent fast changes in signal and don't contribute to recognition (?)
    keep_number = 13
    
    dft_mfccs = dct(logged_mfccs, axis=1, norm='ortho')[:, 1 : keep_number]

    #At this stage apply cepstral lifting if have time
    
    #TRANSPOSE MATRIX SO IT FITS INTO MODEL SEQUENTIALLY
    dft_mfccs = dft_mfccs.transpose()
    
    #save file to .npy
    save_to_file(file_name, dft_mfccs)
    
    #--------------------------------------------------------------------------
    #ALL DEBUGGING GRAPHS BELOW
    #--------------------------------------------------------------------------

    #plot_dft(dft_mfccs) #create spectrogram for debugging

    #plt.plot(r) #plot soundfile
    """
    #PLOT ALL THE FILTERS
    for m in filter_bank:
         plt.plot(m)
         plt.xlim(0, FRAME_LENGTH/2 +3)
         plt.ylim(0,1)
    """
    """
    #RESHAPE POWER ARRAY INTO [FRAME, FILTER, BIN] just to check everything is correct
    #all_windows_filtered = np.reshape(summed_filtered_array, (summed_filtered_array.shape[0], FILTER_NUMBER, int(FRAME_LENGTH/2)))
    print(all_windows_filtered.shape, " filter values reshaped")
    
    #PLOT THE POWER SPECTRUM, FILTER AND RESULT OF FILTER
    filter_index = 41
    window_index = 8
    plot_filter(filter_index, window_index, power_vals_shaped, filter_bank, all_windows_filtered)
    """

for audio_file in sorted(glob.glob('Recordings/*.wav')):
    #Remove file directory and .wav for naming purposes
    audio_file_name = audio_file[11:-4]
    
    main(audio_file, audio_file_name)
    
    print(audio_file_name, 'done')


#TODO vary feature extraction methods:
#ENERGY, TEMPORAL DERIVATIVES: VELOCITY, ACCELERATION
#'SPECTRAL LIFTERING' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
