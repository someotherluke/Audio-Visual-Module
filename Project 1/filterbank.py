# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:40:37 2023

@author: yad23rju
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import plotly.io as pio
import plotly.express as px
import soundfile as sf
from scipy.fftpack import dct


FRAME_LENGTH = 1024 #Number of bins in each window (halved eventually). IDEALLY KEEP AS A RESULT OF 2^x
STEP_LENGTH = FRAME_LENGTH * (1/2)
FILTER_NUMBER = 40 #Number of triangle peaks each window is being multiplied by
#SAMPLE_FREQUENCY = 16000 Currently let the sound file decide frequency

r, fs2 = sf.read('Dave0.wav', dtype='float32')
SAMPLE_FREQUENCY = fs2 # Let the sound file decide frequency
r = r[50000:80000] # Take window of signal just for test purposes

def mel_scale(data):
    return 2595*np.log10(1+(data/700))

def inverse_mel(data):

    return 700*(10**(data/2595) -1)
    
def plot_melscale():
    #Take sampling rate and upper and lower bounds of desired frqeuency
    freq_high = SAMPLE_FREQUENCY/2 #Nyquist
    freq_low = 0 
    mel_freq_high = mel_scale(freq_high)

    mel_freq_low = mel_scale(freq_low)
    #Spread out the mel freqs between the high and low value
    mel_range = np.linspace(mel_freq_low, mel_freq_high, FILTER_NUMBER +2 )

    freq_range = inverse_mel(mel_range)
    
    bins = np.floor((FRAME_LENGTH + 1) * freq_range / SAMPLE_FREQUENCY)

    #code is a variation of: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    fbank = np.zeros((FILTER_NUMBER, int(np.floor(FRAME_LENGTH / 2 ))))


    for m in range(1, FILTER_NUMBER + 1):
        #left
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
    array = np.array([])
    section_squeezed = section.squeeze()
    for element in section:
        array = np.append(array, element)

    #Perform and apply hamming window
    hamming = np.hamming(len(array))
    scaled = np.multiply(hamming,array) 
    fft_sample = np.fft.fft(scaled) 
    absolute = np.abs(fft_sample)
    phase = np.angle(fft_sample)
    #Take 'power' of spectrum
    power = absolute**2/len(array)

    return power
    
    """    PLOT POWER SPECTRUM
    f2,[ax4, ax5] = plt.subplots(nrows=2, ncols=1)
    ax4.plot(absolute)
    #ax5.plot(phase)#plot phase


    #ax5.plot(power)#plot power
    #ax4.set_title('Magnitude')
    #ax5.set_title('power')
    """


def entire_utterance(speech):

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
    gs = fig.add_gridspec(3, hspace=0)
    #axs = gs.subplots(sharex=True)
    axs[0].plot(all_windows_filtered[window_index,filter_index,0:int(FRAME_LENGTH/2)])
    axs[1].plot(power_vals_shaped[window_index])
    axs[2].plot(filter_bank[filter_index])
    axs[0].set_title("Result of filtering")
    axs[1].set_title("Original power spectrum")
    axs[2].set_title("Filter")
    fig.tight_layout()
    
def cepstral_lifter(mfcc):
    #'Ceptral lifting' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf
    #Doesnt work currently
     num_filters = np.shape(mfcc)[1]
     i = np.linspace(0, num_filters)
     lifter_coeff = 22
     lifter = 1 + (lifter_coeff/2)*np.sin(np.pi*i/lifter_coeff)
     mfcc *= lifter
    

#OPEN AND PLAY SOUNDFILE
sd.play(r, SAMPLE_FREQUENCY)

#CALC THE FILTER BANKS
filter_bank = plot_melscale()

#PLOT ALL THE FILTERS
for m in filter_bank:
     plt.plot(m)
     plt.xlim(0, FRAME_LENGTH/2 +3)
     plt.ylim(0,1)

""""" PLOT SPECIFIC FILTERBANK IF NEEDED
#plt.plot(filter_bank[3])
fig10, axs10 = plt.subplots(2)
axs10[0].plot(r)
axs10[1].plot(np.absolute(np.fft.fft(r))[0:1000])
"""""

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
        
        """THIS SECTION WILL PLOT THE FILTERED POWER VALUES - good for debugging
        #adjusted_power = np.multiply(row, filter_peak)
        #summed_filtered = np.sum(adjusted_power)
        #power_vals_filtered = np.append(power_vals_filtered,adjusted_power)

#Reshape into [FRAME, FILTER, BIN]
all_windows_filtered = np.reshape(power_vals_filtered, (power_vals_shaped.shape[0], FILTER_NUMBER, int(FRAME_LENGTH/2)))
print(all_windows_filtered.shape, " filter values reshaped")

#PLOT THE POWER SPECTRUM, FILTER AND RESULT OF FILTER
filter_index = 41
window_index = 8
plot_filter(filter_index, window_index, power_vals_shaped, filter_bank, all_windows_filtered)
"""""

mfccs = summed_filtered_array.reshape(int((len(r)/FRAME_LENGTH)*(FRAME_LENGTH/STEP_LENGTH))-1, FILTER_NUMBER)
print(mfccs.shape, "MFCC SHAPE")

#TAKE LOG THEN DFT
index = 10
logged_mfccs = np.log10(mfccs)
figgy1, axes1 = plt.subplots()
axes1.plot(logged_mfccs[index])
axes1.set_title("logged filtered values")

#PERFORM DFT ON LOGGED VALUES
#Only take 1-13  of the MFCC's as the others represent fast changes in signal and don't contribute to SR (?)
keep_number = 13
dft_mfccs = dct(logged_mfccs, axis=1, norm='ortho')[:, 1 : keep_number]

#Apply lifting function to reduce noise 

#Plot the dft just to check
figgy, axes = plt.subplots()
axes.plot(dft_mfccs[index])

plt.figure(figsize=(15,5))
plt.imshow(dft_mfccs, aspect='auto', origin='lower');







#TODO vary feature extraction methods:
#ENERGY, TEMPORAL DERIVATIVES: VELOCITY, ACCELERATION
#'SPECTRAL LIFTERING' see- https://maxwell.ict.griffith.edu.au/spl/publications/papers/euro99_kkp_fbe.pdf




