'''
Deploys the pre-trained CNN model. Model is loaded from disk
during initialization.
'''

import math
import librosa
import json
import os
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class Model:

    def __init__(self, path='app/saved_model'):
        '''
        Initializes module by loading saved module parameters
        located on disk at a location given by path variable
        '''
        self.model = keras.models.load_model(path)
        # self.model.summary()

    def predict(self, clip_data):
        '''
        Accepts clip_data argument and uses module ML model
        to return a prediction
        '''
       return self.model.predict(data)

model = Model()
