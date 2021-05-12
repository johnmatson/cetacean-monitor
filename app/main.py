'''
Call file from command line with an argument of 'DISK', 'STREAM', or
'MIC' to configure operating mode. Programs runs in 'DISK' mode by
default. Program accepts an additional command line argument for the
IPv4 address of the server, for 'STREAM' mode. By default, program uses
the loopback address (127.0.0.1).
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
from PyQt5.QtWidgets import QApplication
from interface import AlertUI


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

# output smoothing filter weights, should sum to 1
X_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2])
Y_WEIGHTS = np.array([0.2])


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

    if MODE == 'MIC':
        pass


def process(clip_queue, predict_pipe, alert_pipe, new_data, event):
    '''
    Processes audio data from 'clip_queue' as soon as it is available.
    MFCC is calculated for each audio clip, which is fed to CNN, which
    makes a prediction. Exits when 'event' is set.
    '''

    model = keras.models.load_model(MODEL_PATH)

    # prediction filter input and output buffers
    x = np.full(len(X_WEIGHTS), 0.0)
    y = np.full(len(Y_WEIGHTS), 0.0)

    # alert = [False] * 4

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

        # apply smoothing filter to model output
        x = np.roll(x, 1)
        x[0] = prediction[0][0]
        y[0] = np.sum(x * X_WEIGHTS) + np.sum(y * Y_WEIGHTS)

        # alert level assignment
        # alert = [False] * 4
        # if y[0] > 0.7:
        #     alert[:] = [True] * 4
        # elif y[0] > 0.5:
        #     alert[:3] = [True] * 3
        # elif y[0] > 0.3:
        #     alert[:2] = [True] * 2
        # elif y[0] > 0.1:
        #     alert[0] = True
        alert = y[0]

        # print(prediction, x, y, alert)
        predict_pipe.put(prediction)
        alert_pipe.put(alert)
        new_data.set()



if __name__ == '__main__':

    if len(sys.argv) == 2:
        MODE = sys.argv[1]
    elif len(sys.argv) == 3:
        MODE = sys.argv[1]
        HOST = sys.argv[2]

    audio_pipe = queue.Queue(maxsize=10) # input audio queue
    predict_pipe = queue.Queue(maxsize=10)
    alert_pipe = queue.Queue(maxsize=10)

    update_event = threading.Event()
    event = threading.Event() # exit event flag

    try:
        # run read() & process() concurrently
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        executor.submit(read, audio_pipe, event)
        executor.submit(process, audio_pipe, predict_pipe, alert_pipe, update_event, event)

        app = QApplication(sys.argv)
        win = AlertUI(update_event, predict_pipe, alert_pipe)
        app.exec()
        event.set()

    except KeyboardInterrupt:
        event.set()
