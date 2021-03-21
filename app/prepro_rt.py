'''
Handles audio read, clip segmentation and preprocessing for
CNN model.
'''

import math
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from exceptions import *
import soundfile as sf


class Preprocess:

    def __init__(self, mode='disk', audio_path='app/datasets/full-clips/001A.mp3'):
        '''
        Accepts mode argument to configure Preprocess in either
        'disk' mode, where real-time is simulated from a long-
        form audio file on the disk (audio_path being the file
        path to said file), in 'stream' mode where audio is
        recieved over a socket connection from the data colletion
        unit, in 'mic' mode, where audio is streamed from the
        PC audio input, or in 'fast' mode, where data is read
        from the disk, but at a rate much faster than real-time
        '''

        # configures module for disk mode
        if mode == 'disk':
            # import file from disk
            self.file, samp_rate = sf.read("app/datasets/full-clips/001A.mp3")
            # set index to zero
            self.index = 0

        # configures module for stream mode
        elif mode == 'stream':
            pass

        # configures module for mic mode
        elif mode == 'mic':
            pass

        # configures module for mic mode (might be unecessary/renundant)
        elif mode == 'fast':
            pass

    def read(self):
        '''
        Copies 1 second of audio from self.file to self.clip and
        increments self.index, returns EndOfFileError if less
        than 1 second of audio is available
        '''
        self.clip = self.file[self.index*samp_rate:self.index*samp_rate+samp_rate]
        self.index += 1

        if self.index > np.floor(len(self.file)/samp_rate):
            raise EndOfFileError

    def process(self):
        '''
        Processes self.clip audio segment to return clip_data,
        a ___DATATYPE-LIST?____ with an MFCC, ...
        '''
                dataset_path = "app/datasets/test"
        json_path = "app/datasets/test.json"
        SAMPLE_RATE = 16000
        TRACK_DURATION = 1 # measured in seconds
        SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
        int num_mfcc=13, 
        int n_fft=2048, 
        int hop_length=512, 
        int num_segments=1

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        # dictionary to store mapping, labels, and MFCCs
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
        }

        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        # loop through all genre sub-folder
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

            # ensure we're processing a genre sub-folder level
            if dirpath is not dataset_path:

                # save genre label (i.e., sub-folder name) in the mapping
                semantic_label = dirpath.split("\\")[-1]
                data["mapping"].append(semantic_label)
                print("\nProcessing: {}".format(semantic_label))

                # process all audio files in genre sub-dir
                for f in filenames:

            # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, d+1))

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)
            return self.model.predict(clip_data)

        #load data  
        with open(data_path,"r") as fp:
            data = json.load(fp)
        x = np.array(data["mfcc"])
        y = np.array(data["labels"])
        
        #create test data (discard train data using test_size of 1)
        x_train, x_data, y_train, y_data = train_test_split(x,y,test_size = 1)

        #3d array (number of time bins, nfcc values, channel)
        x_data = x_data[...,np.newaxis] #4d array > (number of samples, and the 3d array)

        #set final variables
        x = x_data[0]
        y = y_data[0]

        x = x[np.newaxis,...]
    
       return x, y
