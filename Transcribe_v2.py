#use autocorrelation to estimate fundamental frequencies

# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py
    
import numpy as np

def serial_corr(wave, lag=1):
    N = len(wave)
    y1 = wave.ys[lag:]
    y2 = wave.ys[:N-lag]
    corr = np.corrcoef(y1, y2)[0, 1]
    return corr

def autocorr(wave):
    """Computes and plots the autocorrelation function.

    wave: Wave
    
    returns: tuple of sequences (lags, corrs)
    """
    lags = np.arange(len(wave.ys)//2)
    corrs = [serial_corr(wave, lag) for lag in lags]
    return lags, corrs

def estimate_fundamental(segment, low=10, high=400):
    lags, corrs = autocorr(segment)
    lag = np.array(corrs[low:high]).argmax() + low
    period = lag / segment.framerate
    frequency = 1 / period
    return frequency
  
def SoloTranscribe(wave, length, tempo, subdivision):
    """wave: song to transcribe (.wav.) -- only works for solo instruments right now
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
        segment = wave.segment(timestep, SamplePeriod)
        
        #if max amp of segment peak is significantly below the average of the others, call it a rest
        spectrum = segment.make_spectrum()
        peak_amps = []
        peak_amp = spectrum.peaks()[0][0] # amplitude of this segment's dominant frequency
        peak_amps.append(peak_amp)
        mean_peak_amp = sum(peak_amps) / len(peak_amps) # mean of dominant freqs' amplitudes so far
        if peak_amp < mean_peak_amp*.25:
            transcription.append("Rest")
        
        else:
            #guess fundamental frequency
            fundamental = estimate_fundamental(segment)
            #Find note corresponding to frequency
            note = FindNote(fundamental)
            #Add note to transcription
            transcription.append(note)
        
    return transcription
