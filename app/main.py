'''
Call file from command line with a mode configuration argument of
'DISK', 'STREAM', or 'MIC' or with no arguments to run in disk mode by
default.
'''


import math
import time
import threading
import queue
import concurrent.futures
import socket
import struct
import pickle
import os
import sys
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

SAMPLE_RATE = 16000
NUM_MFCC = 13
NUM_FFT = 2048
HOP_LENGTH = 512

HOST = '127.0.0.1'
PORT = 6829


def read(clip_queue, event):
    '''
    Continuously reads data from either DISK, STREAM or MIC. Adds audio
    data to 'clip_queue' FIFO pipeline in 1-second segments. Exits when
    'event' is set.
    '''

    if MODE == 'DISK':

        print('Audio file laoding...')
        audio_file, fs = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)
        print('Audio file loaded.')

        end_time = time.time()
        
        for i in range(math.floor(len(audio_file)/SAMPLE_RATE)):

            # add 1 second of audio to the queue
            start_sample = int(i*SAMPLE_RATE)
            end_sample = int((i+1)*SAMPLE_RATE)
            clip_queue.put(audio_file[start_sample:end_sample])

            if event.is_set():
                break

            # sleep remainder of 1-second cycle
            time.sleep(1 - (time.time() - end_time))
            end_time = time.time()

    if MODE == 'STREAM':

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:

            client_socket.connect((HOST, PORT)) 
            print("Connected to server",(HOST, PORT))

            data = b""
            payload_size = struct.calcsize("Q")

            while not event.is_set():
                
                try:
                    while len(data) < payload_size:
                        packet = client_socket.recv(4*1024) # 4K
                        if not packet: break
                        data+=packet
                        
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q",packed_msg_size)[0]

                    while len(data) < msg_size:
                        data += client_socket.recv(4*1024)

                    frame_data = data[:msg_size]
                    data  = data[msg_size:]
                    frame = pickle.loads(frame_data)

                    if len(frame) >= SAMPLE_RATE:
                        x = np.array(frame[:SAMPLE_RATE])
                        clip_queue.put(x)

                except Exception as e:
                    print(e)
                    break


def process(clip_queue, event):
    '''
    Processes audio data from 'clip_queue' as soon as it is available.
    MFCC is calculated for each audio clip, which is fed to CNN, which
    makes a prediction. Exits when 'event' is set.
    '''

    print('CNN model loading...')
    model = keras.models.load_model(MODEL_PATH)
    print('CNN model loaded.')

    while not event.is_set() or not clip_queue.empty():

        # extract mfcc
        clip = clip_queue.get()
        mfcc_stage1 = librosa.feature.mfcc(
            clip, SAMPLE_RATE, n_mfcc=NUM_MFCC,n_fft=NUM_FFT,
            hop_length=HOP_LENGTH)
        mfcc_stage2 = np.array((mfcc_stage1.T).tolist())
        mfcc_stage3 = mfcc_stage2[np.newaxis,...,np.newaxis]

        # make prediction using CNN
        prediction = model.predict(mfcc_stage3)
        print(np.argmax(prediction, axis=1))


if __name__ == '__main__':

    if len(sys.argv) == 2:
        MODE = sys.argv[1]
    
    pipeline = queue.Queue(maxsize=10) # input audio queue
    event = threading.Event() # exit event flag

    try:
        # run read() & process() concurrently
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(read, pipeline, event)
        executor.submit(process, pipeline, event)

    except KeyboardInterrupt:
        event.set()
