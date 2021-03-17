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
        clip_data = 0
        return clip_data
