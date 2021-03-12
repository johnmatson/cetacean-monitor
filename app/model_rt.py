import math
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class Model:

    def __init__(self, path='app/saved_model'):
        self.model = keras.models.load_model(path)
        self.model.summary()

    def predict(self, data):
        return self.model.predict(data)
        

model = Model()
