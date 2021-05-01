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
    Continuously reads data from either DISK, STREAM or MIC. Adds
    audio data to 'clip_queue' FIFO pipeline in 1-second segments.
    Exits when 'event' is set.
    '''

    if MODE == 'DISK':

        print('Audio file laoding...')
        audio_file, fs = librosa.load(AUDIO_PATH)
        print('Audio file loaded.')

        for i in range(math.floor(len(audio_file)/SAMPLE_RATE)):
            
            start = time.time()

            # add 1 second of audio to the queue
            clip_queue.put(audio_file[int(i*SAMPLE_RATE):int(i*SAMPLE_RATE+SAMPLE_RATE)])

            if event.is_set():
                break

            end = time.time()
            time.sleep(1 - (end - start))


def process(clip_queue, event):
    '''
    Processes audio data from 'clip_queue' as soon as it is
    available. MFCC is calculated for each audio clip, which is
    fed to CNN, which makes a prediction. Exits when 'event' is
    set.
    '''

    print('CNN model loading...')
    model = keras.models.load_model(MODEL_PATH)
    print('CNN model loaded.')

    while not event.is_set() or not clip_queue.empty():

        # extract mfcc
        clip = clip_queue.get()
        mfcc_stage1 = librosa.feature.mfcc(clip, SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=NUM_FFT, hop_length=HOP_LENGTH)
        mfcc_stage2 = np.array((mfcc_stage1.T).tolist())
        mfcc_stage3 = mfcc_stage2[np.newaxis,...,np.newaxis]

        # make prediction using CNN
        prediction = model.predict(mfcc_stage3)
        print(np.argmax(prediction, axis=1))


if __name__ == '__main__':
    
    pipeline = queue.Queue(maxsize=10) # input audio queue
    event = threading.Event() # exit event flag

    try:
        # run read() & process() concurrently
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(read, pipeline, event)
        executor.submit(process, pipeline, event)

    except KeyboardInterrupt:
        event.set()
