# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:40:37 2023

@author: yad23rju

MAIN FEATURE EXTRACTION CODE. Reads .wav file in the folder 'Recordings' and outputs mfcc's to folder ''mfccs'
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy.fftpack import dct
import glob 

#Set params
FRAME_LENGTH = 512 #Number of bins in each window (halved eventually). keep above 500 and even otherwise bin ranges are too small
STEP_LENGTH = int(FRAME_LENGTH * (1/2))
FILTER_NUMBER = 39 #Number of triangle peaks each window is being multiplied by

def mel_scale(data):
    return 2595*np.log10(1+(data/700))

def inverse_mel(data):
    return 700*(10**(data/2595) -1)
    
def plot_melscale(sample_frequency):
    #Take sampling rate and upper and lower bounds of desired frequency
    freq_high = sample_frequency/2 #Nyquist
    
    freq_low = 0
    
    #convert frequency to melscale
    mel_freq_high = mel_scale(freq_high)

    mel_freq_low = mel_scale(freq_low)
    
    #Spread out the mel freqs between the high and low value - 2 added to filter number so bins either side can be taken
    mel_range = np.linspace(mel_freq_low, mel_freq_high, FILTER_NUMBER +2 )

    #find frequency of these mel numbers
    freq_range = inverse_mel(mel_range)
        
    #convert frequencies into bins
    bins = np.floor((FRAME_LENGTH + 1) * freq_range / sample_frequency)

    #set filterbank array to be filled, with columns equal to mfcc number and rows equal to frame length
    fbank = np.zeros((FILTER_NUMBER, int(np.floor(FRAME_LENGTH / 2 ))))
    #code is a variation of: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    #Define a left, middle and right bin
    for m in range(1, FILTER_NUMBER + 1):
        #left bin value
        f_m_minus = int(bins[m - 1])  
        #middle
        f_m = int(bins[m])
        #right
        f_m_plus = int(bins[m + 1])
        #Iterate over the range of values between the left bin and the centre bin, adding to the fbank array
        for k in range(f_m_minus, f_m):
            #k here is the column number
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        #Iterate over the range of values between the centre bin and the right bin
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank  

def mag_and_phase(section):
    #Calculate the fft and power
    #Turn section of speech into 1d array
    section_squeezed = section.squeeze()
    section = np.array(section)
    #Perform and apply hamming window
    hamming = np.hamming(len(section))
    scaled = np.multiply(hamming,section) 
    #perform fft on signal an
    fft_sample = np.fft.fft(scaled) 
    absolute = np.abs(fft_sample)
    #Take 'power' of spectrum
    power = absolute**2/len(section)

    return power

def entire_utterance(speech):
    #Window signal in steps and call the fft function above to find the mag and power
    num_samples = len(speech)
    
    #Calculate number of frames total in the soundfile
    num_frames = num_samples/FRAME_LENGTH * (FRAME_LENGTH/STEP_LENGTH)
    
    #ALWAYS ROUND DOWN TO REMOVE THE WINDOW WITH NO SIGNAL
    num_frames = int(np.floor(num_frames)) 
    
    all_power = np.array([])
    energy_vals = np.array([])
    #miss out last digit because the last frame will have little audio anyway
    for i in range(0, num_frames-1):

        #Find power from each window
        power = mag_and_phase(speech[STEP_LENGTH*i:STEP_LENGTH*i + FRAME_LENGTH])
        #sum the square of the window signal for the energy
        energy = np.sum(speech[STEP_LENGTH*i:STEP_LENGTH*i + FRAME_LENGTH]**2)
        energy_vals = np.append(energy_vals, energy)

        # TAKE HALF OF FFT because symmetry
        all_power = np.append( all_power, power[0:int(FRAME_LENGTH/2)]) 
        
    return all_power, energy_vals

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
    figgy, axes = plt.subplots(2)
    axes[0].plot(dft_mfccs[index])
    #plot spectrogram of mfcc
    #plt.figure(figsize=(15,5))
    axes[1].imshow(dft_mfccs, aspect='auto', origin='lower')
    #axes[2].specgram(dft_mfccs[3])

    
def cepstral_lifter(mfcc):
    #'Ceptral lifting' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
    #and https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1165237
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
    
def save_to_file(file_name, mfcc):
    #Saves file to the folder 'mfccs' in directory
    #allow pickle!
    directory = os.path.join('mfccs', file_name)
    np.save(directory, mfcc, allow_pickle=True)
    
    
def add_noise(signal, SNR):
    #Normally distribute the noise 
    noise = np.random.normal(0,0.1, len(signal))
    
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

    #Read in file
    r, fs2 = sf.read(file_dir, dtype='float32')
    
    sample_frequency = 16000 
    
    r = r[3000:-115000] # Take window of signal initially to remove long periods of silence from recording

    #add noise to the signal, varied by the SNR coefficient
    #add_noise(r, 0)

    #Calculate the filterbanks, with the same sampling frequency as the FFT
    filter_bank = plot_melscale(sample_frequency)
    
    #Extract the power values from each window of signal and add to big array
    all_power_vals, energy_vals = entire_utterance(r)
    
    #Reshape the power array such that each row corresponds to 1 frame of the FFT power values
    power_vals_shaped = np.reshape(all_power_vals, (int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, int(FRAME_LENGTH/2)))
    print(power_vals_shaped.shape, "POWER_VALS SHAPE")
    
    #Iterate over the power array and multiply by each mel filter triangle and sum up the result
    summed_filtered_array = np.array([])
    power_vals_filtered = np.array([])
    for row in power_vals_shaped:
        for filter_peak in filter_bank:
            
            summed_filtered = np.matmul(row, filter_peak)
            summed_filtered_array = np.append(summed_filtered_array, summed_filtered)
            
            """
            #UNCOMMENT TO SEE GRAPHS OF POWER, FILTER AND FILTER RESULT
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
    print(mfccs.shape, "MFCC SHAPE") #Check shape is correct
    
    #Take log of mfccs
    logged_mfccs = np.log10(mfccs)

    #PERFORM DFT ON LOGGED VALUES
    #Discard some MFCC's as the others represent fast changes in signal and don't contribute to recognition. 'Truncation' in the powerpoints
    keep_number = 39
    dft_mfccs = dct(logged_mfccs, axis=1, norm='ortho')[:, 0 : keep_number]   
            
    #add energy values to the end of each mfcc
    dft_mfccs = np.insert(dft_mfccs,dft_mfccs.shape[1],energy_vals,axis=1)

    #TRANSPOSE MATRIX SO IT FITS INTO MODEL SEQUENTIALLY
    dft_mfccs = dft_mfccs.transpose()
    
    #save file to .npy in mfccs folder in directory
    save_to_file(file_name, dft_mfccs)

    #--------------------------------------------------------------------------
    #ALL DEBUGGING GRAPHS BELOW
    #--------------------------------------------------------------------------

    #plot_dft(dft_mfccs) #plot dft

    #plt.plot(r) #plot soundfile

    #PLOT ALL THE FILTERS
    """
    for m in filter_bank:
         plt.plot(m)
         plt.xlim(0, FRAME_LENGTH/2 +3)
         plt.ylim(0,1)
    """
    
    #plot signal
    #fig10, ax10 = plt.subplots()
    #ax10.plot(r)
    #ax10.set_title(file_name)

#USE sorted(glob.glob('Recordings/*.wav')) for actual training. debugging for testing
for audio_file in sorted(glob.glob('debugging/*.wav')):
    #Remove file directory and .wav for naming purposes
    audio_file_name = audio_file[11:-4]
    
    main(audio_file, audio_file_name)
    
    print(audio_file_name, 'done')

print('Done entire data set')

#TODO vary feature extraction methods:
#ENERGY, TEMPORAL DERIVATIVES: VELOCITY, ACCELERATION
#'SPECTRAL LIFTERING' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
