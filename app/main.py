'''

'''


import math
import time
import threading
import queue
import concurrent.futures
import librosa
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


# MODE OPTIONS
# 'DISK'    : sources audio from local audio file
# 'STREAM'  : sources audio from socket connection
# 'MIC'     : sources audio from PC audio input

MODE = 'DISK'

AUDIO_PATH = 'app/data/full-clips/001A-short.wav'
MODEL_PATH = 'app/saved_model'

SAMPLE_RATE = 16e3
NUM_MFCC = 13
NUM_FFT = 2048
HOP_LENGTH = 512


def read(clip_queue, event):
    '''
    '''

    if MODE == 'DISK':

        print('Audio file laoding')
        audio_file, fs = librosa.load(AUDIO_PATH)
        print('Audio file loaded')

        for i in range(math.floor(len(audio_file)/SAMPLE_RATE)):
            start = time.time()
            if event.is_set():
                break
            clip_queue.put(audio_file[int(i*SAMPLE_RATE):int(i*SAMPLE_RATE+SAMPLE_RATE)])
            end = time.time()
            time.sleep(1 - (end - start))


def process(clip_queue, event):
    '''
    '''

    print('CNN model loading')
    model = keras.models.load_model(MODEL_PATH)
    print('CNN model loaded')

    while not event.is_set():
    # while True:

        # extract mfcc
        clip = clip_queue.get()
        mfcc_stage1 = librosa.feature.mfcc(clip, SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=NUM_FFT, hop_length=HOP_LENGTH)
        mfcc_stage2 = np.array((mfcc_stage1.T).tolist())
        mfcc_stage3 = mfcc_stage2[np.newaxis,...,np.newaxis]

        # make prediction using CNN
        prediction = model.predict(mfcc_stage3) # ADD PREDICTION TO OUTPUT CUE
        print(np.argmax(prediction, axis=1))


if __name__ == '__main__':
    
    pipeline = queue.Queue(maxsize=10)
    event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(read, pipeline, event)
        executor.submit(process, pipeline, event)
        # time.sleep(30)
        # event.set()
