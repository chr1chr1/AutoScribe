#Cluster notes from transcription into individual voices (machine learning code adapted from MIT OCW course 6.0002)

import pylab
import random

def Dist(v1, v2):
    #Assumes v1 and v2 are equal length arrays of numbers
    dist = abs(v1 - v2)
    return dist

class Example(object):
    
    def __init__(self, name, features, label = None):
        #Assumes features is an array of floats
        self.name = name # note
        self.features = features #frequency 
        self.label = label
        
    def dimensionality(self):
        return 1
    
    def getFeatures(self):
        return self.features
    
    def getLabel(self):
        return self.label
    
    def getName(self):
        return self.name
    
    def distance(self, other):
        return Dist(self.features, other.getFeatures())
    
    def __str__(self):
        return self.name +':'+ str(self.features) + ':'\
               + str(self.label)

class Cluster(object):
    
    def __init__(self, examples):
        """Assumes examples a non-empty list of Examples"""
        self.examples = examples
        self.centroid = self.computeCentroid()
        
    def update(self, examples):
        """Assume examples is a non-empty list of Examples
           Replace examples; return amount centroid has changed"""
        oldCentroid = self.centroid
        self.examples = examples
        self.centroid = self.computeCentroid()
        return oldCentroid.distance(self.centroid)
    
    def computeCentroid(self):
        vals = pylab.array([0.0]*self.examples[0].dimensionality())
        for e in self.examples: #compute mean
            vals += e.getFeatures()
        centroid = Example('centroid', vals/len(self.examples))
        return centroid

    def getCentroid(self):
        return self.centroid

    def variability(self):
        totDist = 0
        for e in self.examples:
            totDist += (e.distance(self.centroid))**2
        return totDist
        
    def members(self):
        for e in self.examples:
            yield e

    def __str__(self):
        names = []
        for e in self.examples:
            names.append(e.getName())
        names.sort()
        result = 'Cluster with centroid '\
               + str(self.centroid.getFeatures()) + ' contains:\n  '
        for e in names:
            result = result + e + ', '
        return result[:-2] #remove trailing comma and space    
        
def dissimilarity(clusters):
    """Assumes clusters a list of clusters
       Returns a measure of the total dissimilarity of the
       clusters in the list"""
    totDist = 0
    for c in clusters:
        totDist += c.variability()
    return totDist

def kmeans(examples, k, verbose = False):
    #Get k randomly chosen initial centroids, create cluster for each
    initialCentroids = random.sample(examples, k)
    clusters = []
    for e in initialCentroids:
        clusters.append(Cluster([e]))
        
    #Iterate until centroids do not change
    converged = False
    numIterations = 0
    while not converged:
        numIterations += 1
        #Create a list containing k distinct empty lists
        newClusters = []
        for i in range(k):
            newClusters.append([])
            
        #Associate each example with closest centroid
        for e in examples:
            #Find the centroid closest to e
            smallestDistance = e.distance(clusters[0].getCentroid())
            index = 0
            for i in range(1, k):
                distance = e.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add e to the list of examples for appropriate cluster
            newClusters[index].append(e)
            
        for c in newClusters: #Avoid having empty clusters
            if len(c) == 0:
                raise ValueError('Empty Cluster')
        
        #Update each cluster; check if a centroid has changed
        converged = True
        for i in range(k):
            if clusters[i].update(newClusters[i]) > 0.0:
                converged = False
        if verbose:
            print('Iteration #' + str(numIterations))
            for c in clusters:
                print(c)
            print('') #add blank line
    return clusters

def trykmeans(examples, numClusters, numTrials, verbose = False):
    """Calls kmeans numTrials times and returns the result with the
          lowest dissimilarity"""
    best = kmeans(examples, numClusters, verbose)
    minDissimilarity = dissimilarity(best)
    trial = 1
    while trial < numTrials:
        try:
            clusters = kmeans(examples, numClusters, verbose)
        except ValueError:
            continue #If failed, try again
        currDissimilarity = dissimilarity(clusters)
        if currDissimilarity < minDissimilarity:
            best = clusters
            minDissimilarity = currDissimilarity
        trial += 1
    return best
  
#after transcribe, CLUSTER data, keep track of frequencies alongside notes
def GenSamples(transcription):
    FindFreq = {'C8': 4186.009,  'B7': 3951.066,  'A#7': 3729.31,  'A7': 3520.0,  'G#7/Ab7': 3322.438,  'G7': 3135.963,  'F#7/Gb7': 2959.955,  'F7': 2793.826,  'E7': 2637.02,  'D#7/Eb7': 2489.016,  'D7': 2349.318,  'C#7/Db7': 2217.461,  'C7': 2093.005,  'B6': 1975.533,  'A#6/Bb6': 1864.655,  'A6': 1760.0,  'G#6/Ab6': 1661.219,  'G6': 1567.982,  'F#6/Gb6': 1479.978,  'F6': 1396.913,  'E6': 1318.51,  'D#6/Eb6': 1244.508,  'D6': 1174.659,  'C#6/Db6': 1108.731,  'C6': 1046.502,  'B5': 987.7666,  'A#5/Bb5': 932.3275,  'A5': 880.0,  'G#5/Ab5': 830.6094,  'G5': 783.9909,  'F#5/Gb5': 739.9888,  'F5': 698.4565,  'E5': 659.2551,  'D#5/Eb5': 622.254,  'D5': 587.3295,  'C#5/Db5': 554.3653,  'C5': 523.2511,  'B4': 493.8833,  'A♯4/Bb4': 466.1638,  'A4': 440.0,  'G♯4/Ab4': 415.3047,  'G4': 391.9954,  'F♯4/Gb4': 369.9944,  'F4': 349.2282,  'E4': 329.6276,  'D♯4/Eb4': 311.127,  'D4': 293.6648,  'C♯4/Db4': 277.1826,  'C4': 261.6256,  'B3': 246.9417,  'A♯3/Bb3': 233.0819,  'A3': 220.0,  'G♯3/Ab3': 207.6523,  'G3': 195.9977,  'F♯3/Gb3': 184.9972,  'F3': 174.6141,  'E3': 164.8138,  'D♯3/Eb3': 155.5635,  'D3': 146.8324,  'C#3/Db3': 138.5913,  'C3': 130.8128,  'B2': 123.4708,  'A#2/Bb2': 116.5409,  'A2': 110.0,  'G#2/Ab2': 103.8262,  'G2': 97.99886,  'F#2/Gb2': 92.49861,  'F2': 87.30706,  'E2': 82.40689,  'D#2/Eb2': 77.78175,  'D2': 73.41619,  'C#2/Db2': 69.29566,  'C2': 65.40639,  'B1': 61.73541,  'A#1/Bb1': 58.27047,  'A1': 55.0,  'G#1/Ab1': 51.91309,  'G1': 48.99943,  'F#1/Gb1': 46.2493,  'F1': 43.65353,  'E1': 41.20344,  'D#1/Eb1': 38.89087,  'D1': 36.7081,  'C#1/Db1': 34.64783,  'C1': 32.7032,  'B0': 30.86771,  'A#0/Bb0': 29.13524,  'A0': 27.5}
    
    samples = []
    sample_names = []
    for subdivision in transcription:
        for note in subdivision:
            if note != "Rest":
                if note not in sample_names: #only add a note once (only works for voices that don't cross).
                    samples.append(Example(note, FindFreq[note]))
                    sample_names.append(note)
    return samples

def ClusterNotes(transcription, k, numTrials):
    """says which notes should belong to which voice"""
    samples = GenSamples(transcription)   
    best_clustering = trykmeans(samples, k, numTrials)
    return best_clustering

def getClusters(clusters):
    groups = []
    for c in clusters:
        names = []
        for e in c.examples:
            names.append(e.getName())
        groups.append(names)
    return groups

def SplitTranscription(transcription, groups):
    voices = [[] for i in range(len(groups))] # a list (transcription) for each voice
    #add note to  a voice based on which cluster it belongs to, one for each voice
    #slower? iterating through transcriptions len(groups) times (but simpler to code than once through transcription)
    for i in range(len(groups)):
        for subdivision in transcription:
            counter = 0
            for note in subdivision:
                if note in groups[i]:
                    voices[i].append(note)
                    counter += 1
                    break # should only be one matching note in chord per group by design
            if counter == 0:
                voices[i].append("Rest")
    return voices
