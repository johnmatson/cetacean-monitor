'''
Handles audio read, clip segmentation and preprocessing for
CNN model.
'''

import math
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from exceptions import *


SAMPLE_RATE = 16000
TRACK_DURATION = 1 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


class Preprocess:

    def __init__(self, mode='disk', audio_path='app/data/datasets/full-clips/001A-short.wav'):
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

        # configures module to source audio from local audio file
        if mode == 'disk':
            # import file from disk
            self.file, self.samp_rate = librosa.load(audio_path)
            # set index to zero
            self.index = 0

        # configures module to source audio from socket connection
        elif mode == 'stream':
            # ADD SOCKET CODE HERE
            pass

        # configures module to source audio from PC audio input
        elif mode == 'mic':
            pass

    def read(self):
        '''
        Copies 1 second of audio from self.file to self.clip and
        increments self.index, returns EndOfFileError if less
        than 1 second of audio is available
        '''
        self.clip = self.file[self.index*self.samp_rate:self.index*self.samp_rate+self.samp_rate]
        self.index += 1

        if self.index > np.floor(len(self.file)/self.samp_rate):
            raise EndOfFileError

    def process(self, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
        '''
        Processes self.clip audio segment to return mfcc,
        a ___DATATYPE-LIST?____ with an MFCC, ...
        '''

        samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

        # process all segments of audio file
        for d in range(num_segments):

            # calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment

            # extract mfcc
            mfcc = librosa.feature.mfcc(self.clip[start:finish], self.samp_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

            # store only mfcc feature with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                a = np.array(mfcc.tolist())
                return a[np.newaxis,...,np.newaxis]
