import math
import librosa
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

class Model:

    def __init__(self, path='app/saved_model'):
        model = keras.models.load_model(path)
        model.summary()

    def predict(self, data):
        return 0
        

model = Model()
