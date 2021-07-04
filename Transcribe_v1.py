# Get thinkdsp.py

import os

if not os.path.exists('thinkdsp.py'):
    !wget https://github.com/AllenDowney/ThinkDSP/raw/master/code/thinkdsp.py
    
import numpy as np

def FindNote(frequency):
    """find closest key to fundamental frequency and add corresponding value (note) to transcription"""
    EqualTemp = {4186.009: 'C8', 3951.066: 'B7', 3729.31: 'A#7', 3520.0: 'A7', 3322.438: 'G#7/Ab7', 3135.963: 'G7', 2959.955: 'F#7/Gb7', 2793.826: 'F7', 2637.02: 'E7', 2489.016: 'D#7/Eb7', 2349.318: 'D7', 2217.461: 'C#7/Db7', 2093.005: 'C7', 1975.533: 'B6', 1864.655: 'A#6/Bb6', 1760.0: 'A6', 1661.219: 'G#6/Ab6', 1567.982: 'G6', 1479.978: 'F#6/Gb6', 1396.913: 'F6', 1318.51: 'E6', 1244.508: 'D#6/Eb6', 1174.659: 'D6', 1108.731: 'C#6/Db6', 1046.502: 'C6', 987.7666: 'B5', 932.3275: 'A#5/Bb5', 880.0: 'A5', 830.6094: 'G#5/Ab5', 783.9909: 'G5', 739.9888: 'F#5/Gb5', 698.4565: 'F5', 659.2551: 'E5', 622.254: 'D#5/Eb5', 587.3295: 'D5', 554.3653: 'C#5/Db5', 523.2511: 'C5', 493.8833: 'B4', 466.1638: 'A♯4/Bb4', 440.0: 'A4', 415.3047: 'G♯4/Ab4', 391.9954: 'G4', 369.9944: 'F♯4/Gb4', 349.2282: 'F4', 329.6276: 'E4', 311.127: 'D♯4/Eb4', 293.6648: 'D4', 277.1826: 'C♯4/Db4', 261.6256: 'C4', 246.9417: 'B3', 233.0819: 'A♯3/Bb3', 220.0: 'A3', 207.6523: 'G♯3/Ab3', 195.9977: 'G3', 184.9972: 'F♯3/Gb3', 174.6141: 'F3', 164.8138: 'E3', 155.5635: 'D♯3/Eb3', 146.8324: 'D3', 138.5913: 'C#3/Db3', 130.8128: 'C3', 123.4708: 'B2', 116.5409: 'A#2/Bb2', 110.0: 'A2', 103.8262: 'G#2/Ab2', 97.99886: 'G2', 92.49861: 'F#2/Gb2', 87.30706: 'F2', 82.40689: 'E2', 77.78175: 'D#2/Eb2', 73.41619: 'D2', 69.29566: 'C#2/Db2', 65.40639: 'C2', 61.73541: 'B1', 58.27047: 'A#1/Bb1', 55.0: 'A1', 51.91309: 'G#1/Ab1', 48.99943: 'G1', 46.2493: 'F#1/Gb1', 43.65353: 'F1', 41.20344: 'E1', 38.89087: 'D#1/Eb1', 36.7081: 'D1', 34.64783: 'C#1/Db1', 32.7032: 'C1', 30.86771: 'B0', 29.13524: 'A#0/Bb0', 27.5: 'A0'}
    
    best_diff = 99999
    choice = 0
    for key in EqualTemp:
        diff = abs(frequency - key)
        if diff < best_diff:
            best_diff = diff
            choice = key
    return EqualTemp[choice]
    
    def GuessChord(spectrum, MaxVoices):
      """spectrum: Spectrum, MaxVoices: maximum number of possible voices in the chord (int), 
      Returns: list of notes comprising chord guess, and a list of their corresponding frequencies"""
      #Take all frequencies that are at least, say, 90% of the amplitude of the dominant frequency, 
      #cut out any duplicates, and return chord (so long as it's <= MaxVoices)
      notes = []
      for i in range(10):
          #in case there are duplicates of the same frequency, comparably dominant frequency and not the same note as one of the others already added to notes
          if len(notes) < MaxVoices and FindNote(spectrum.peaks()[i][1]) not in notes:
              notes.append(FindNote(spectrum.peaks()[i][1])
      return notes
  
def transcribe(wave, length, tempo, subdivision, MaxVoices):
    """wave: song to transcribe (.wav)
    length: length of song (in seconds) -- possible to get the length of the sound file without an argument?
    tempo: BPM of song
    subdivision: fastest note, e.g. 1/16 = sixteenth note, 1/4 = quarter note
    MaxVoices: max number of voices playing at any one time, defaults to 1
    Returns: Transcription (list of notes, grouped by subdivision if MaxVoices > 1)"""
    import numpy as np
    import thinkdsp
    
    #read sound file
    wave = read_wave(wave)
    
    transcription = []
    
    #subdivide song into even timesteps of length SamplePeriod
    QuarterNoteLength = 60 / tempo #length of quarter note in seconds e.g. 120 BPM tempo -> quarter note = .5 sec
    SamplePeriod = (QuarterNoteLength*4)*subdivision #duration of specifiednsample length to take (in seconds)
    SampleRate = 1 / SamplePeriod
    NumSamples = round(length * SampleRate)
    ts = np.arange(NumSamples) / SampleRate 

    peak_amps = []
    for timestep in ts:
        #produce harmonic spectrum for given segment and get its peaks
        segment = wave.segment(timestep, SamplePeriod) # would windowing help, or if segment contains a chord would it attenuate other notes?
        spectrum = segment.make_spectrum()
        peaks = spectrum.peaks()[:10]
        peak_amps.append(peaks[0][0])

        #if max amp of segment peak is significantly below the average of the others, call it a "rest" 
        peak_amp = spectrum.peaks()[0][0] # this segment's particular dominant freq's amplitude
        mean_amp = sum(peak_amps) / len(peak_amps) # mean of dominant freqs' amplitudes so far
        if peak_amp < mean_amp*.25:
            rest = []
            for i in range(MaxVoices):
                rest.append("Rest")
            transcription.append(rest)
        
        elif MaxVoices == 1:
            #guess fundamental frequency
            fundamental = min([freqs[1] for freqs in peaks])
            #Find note corresponding to frequency
            note = FindNote(fundamental)
            transcription.append(note)

        else:
            chord = GuessChord(spectrum, MaxVoices)
            transcription.append(chord)
        
    return transcription
