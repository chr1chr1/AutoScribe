#use autocorrelation to estimate fundamental frequencies

# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py
    
import numpy as np
  
def AutoPitch(segment, low=10, high=200):
    """Use Autocorrelation to estimate pitch more accurately than spectrums"""
    #only use positive lags, and standardize correlation
    N = len(segment)
    corrs = np.correlate(segment.ys, segment.ys, mode='same')
    lags = np.arange(-N//2, N//2)
    N = len(corrs)
    lengths = range(N, N//2, -1)
    half = corrs[N//2:].copy()
    half /= lengths
    half /= half[0]
    
    #use argmax to find index of peak (# of frames lagged with highest correlation)
    lag = np.array(half[low:high]).argmax() + low

    #find frequency
    period = lag / segment.framerate
    frequency = 1 / period

    return frequency

def transcribe2(wave, length, tempo, subdivision, NumVoices):
    """wave: song to transcribe (.wav.) -- works for 1 or 2 voices
    length: length of song (in seconds) -- possible to get the length of the sound file without an argument?
    tempo: BPM of song
    subdivision: fastest note, e.g. 1/16 = sixteenth note, 1/4 = quarter note
    MaxVoices: max number of voices playing at any one time, defaults to 1
    Returns: Transcription"""
    
    #read sound file
    wave = read_wave(wave)
    
    transcription = []
    
    #subdivide song into even timesteps of length SamplePeriod
    QuarterNoteLength = 60 / tempo #length of quarter note in seconds e.g. 120 BPM tempo -> quarter note = .5 sec
    SamplePeriod = (QuarterNoteLength*4)*subdivision #duration of specifiednsample length to take (in seconds)
    SampleRate = 1 / SamplePeriod
    NumSamples = round(length * SampleRate)
    ts = np.arange(NumSamples) / SampleRate 

    for timestep in ts:
        segment = wave.segment(start=timestep, duration=SamplePeriod)
        
        #if max amp of segment peak is significantly below the average of the others, call it a rest
        spectrum = segment.make_spectrum()
        peak_amps = []
        peak_amp = spectrum.peaks()[0][0] # amplitude of this segment's dominant frequency
        peak_amps.append(peak_amp)
        mean_peak_amp = sum(peak_amps) / len(peak_amps) # mean of dominant freqs' amplitudes so far
        if peak_amp < mean_peak_amp*.25:
            transcription.append("Rest")
        
        elif NumVoices == 1:
            #guess fundamental frequency
            fundamental = AutoPitch(segment) # independent of segment size, since we look at specific range of lags?
            #Find note corresponding to frequency
            note = FindNote(fundamental)
            #Add note to transcription
            transcription.append(note)
            
        elif NumVoices == 2:
            chord = GuessChord3(segment, NumVoices)
            transcription.append(chord)
        
    return transcription
