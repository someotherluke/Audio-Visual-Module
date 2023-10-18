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
#Maths Modules
import numpy as np

#PLotting Modules
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px

#Sound recording & playback Modules
import sounddevice as sd
import soundfile as sf


FRAME_LENGTH = 512 #Number of bins in each window (halved eventually)
FILTER_NUMBER = 12 
FILTER_NUMBER_PROXY = FILTER_NUMBER + 2 #Number of triangle peaks each window is being multiplyed by MINUS 2 (due to code)
SAMPLE_FREQUENCY = 16000

def mel_scale(data):
    return 2595*np.log10(1+(data/700))

def inverse_mel(data):

    return 700*(10**(data/2595) -1)
    
def plot_melscale(data):
    #Take sampling rate and upper and lower bounds of desired frqeuency

    freq_high = 8000    #Nyquist
    freq_low = 300      #Chosen at random
    
    mel_freq_high = mel_scale(freq_high)
    mel_freq_low = mel_scale(freq_low)

    mel_range = np.linspace(mel_freq_low, mel_freq_high, FILTER_NUMBER )
    
    freq_range = inverse_mel(mel_range)
    
    bins = np.floor((FRAME_LENGTH + 1) * freq_range / SAMPLE_FREQUENCY)
    print(mel_range)
    print(freq_range)
    print(bins)

    
    #CODE IS A VARIATION OF: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    fbank = np.zeros((FILTER_NUMBER-2, int(np.floor(FRAME_LENGTH / 2 ))))
    #fbank = [[]]


    #THE 8TH FILTER IS NOTHING 
    for m in range(1, FILTER_NUMBER -1):

        f_m_minus = int(bins[m - 1])   # left
        #print(f_m_minus)
        f_m = int(bins[m])    
        #print(f_m)         # center
        f_m_plus = int(bins[m + 1])    # right
        
        #Iterate over the range of values between the left bin and the centre bin
        for k in range(f_m_minus, f_m):
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

    hamming = np.hamming(len(array))
    #print("hamming done")
    scaled = np.multiply(hamming,array) # SHOULD BE ABSOLUTE VALUE OF ARRAY?
    fft_sample = np.fft.fft(scaled) #use rfft if a specific bin number is required
    
    absolute = np.abs(fft_sample)
    phase = np.angle(fft_sample)
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
    #plot_melscale(absolute[0:int(len(absolute)/2)]) # HERE WILL MAYBE MULITPLY THE FILTER


def entire_utterance(speech):

    num_samples = len(speech)
    #frame_length = 400 #at 16khz this is 25ms
    step_length = FRAME_LENGTH/2
    step_length = int(step_length)
    num_frames = num_samples/FRAME_LENGTH
    num_frames = int(num_frames)
    all_power = np.array([])


    
    for i in range(0, num_frames*2-1):

        power = mag_and_phase(speech[step_length*i:step_length*i + FRAME_LENGTH])

        #all_power = np.append(all_power, power, axis=0)
        all_power = np.append( all_power, power[0:int(FRAME_LENGTH/2)])
    return all_power

def plot_filter(filter_index, window_index, power_vals, filter_bank):
    
    fig, axs = plt.subplots(3)
    #fig.suptitle('Vertically stacked subplots')
    gs = fig.add_gridspec(3, hspace=0)
    #axs = gs.subplots(sharex=True)
    
    axs[0].plot(all_windows_filtered[window_index,filter_index,0:int(FRAME_LENGTH/2)])
    
    axs[1].plot(power_vals_shaped[window_index])
    axs[2].plot(filter_bank[filter_index])
    axs[0].set_title("Result of filtering")
    axs[1].set_title("Original power spectrum")
    axs[2].set_title("Filter")
    fig.tight_layout()
    

#OPEN AND PLAY SOUNDFILE
r, fs2 = sf.read('0_lucas_0.wav', dtype='float32')
sd.play(r, SAMPLE_FREQUENCY)

#CALC THE FILTER BANKS. GET THE POWER SPECTRUM OF EACH WINDOW AND MULTIPLY THEM
#Create the mel filterbanks in of shape (40,201), so for each FFT bin there are 40 mel filters (not sure why it's 201 and not 200)
filter_bank = plot_melscale(3)



#PLOT ALL THE FILTERS
for m in filter_bank:
    plt.plot(m)
    
#plt.plot(filter_bank_adjusted[2])

all_power_vals = entire_utterance(r)

#all_power_vals.reshape(int((len(r)/FRAME_LENGTH)*2)-2, FRAME_LENGTH)
#Reshape the power array such that each row corresponds to 1 frame of the FFT power values
power_vals_shaped = np.reshape(all_power_vals, (int((len(r)/FRAME_LENGTH)*2)-2, int(FRAME_LENGTH/2)))
print(power_vals_shaped.shape, "POWER_VALS SHAPE")
plt.plot(power_vals_shaped[0])
i = 0

power_vals_filtered = np.array([])
for row in power_vals_shaped:
    for filter_peak in filter_bank:
        adjusted_power = np.multiply(row, filter_peak)
        power_vals_filtered = np.append(power_vals_filtered, adjusted_power)
        
#Reshape into [FRAME, FILTER, BIN]
all_windows_filtered = np.reshape(power_vals_filtered, (power_vals_shaped.shape[0], FILTER_NUMBER-2, int(FRAME_LENGTH/2)))
print(all_windows_filtered.shape, " filter values reshaped")
    
#for i in range(all_windows_filtered.shape[1]):

#CURRENTLY FILTER 8 IS NOTHING AND 9 IS WEIRD
filter_index = 0
window_index = 0
plot_filter(filter_index, window_index, power_vals_shaped, filter_bank)
