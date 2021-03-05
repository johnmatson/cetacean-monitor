import math
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class Preprocess:

    def __init__(self, mode='disk', audio_path='./datasets/full-clips/001A.mp3'):

        if mode == 'disk':
            # import file from disk
            # self.file = ...

            # set index to zero
            self.index = 0

        elif mode == 'stream':
            pass

    def read(self):
        pass

    def process(self):
        data = 0
        return data
