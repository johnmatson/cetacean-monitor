'''
Designed to run on the data-processing PC. Program establishes audio
read (audio source dependant on MODE argument), audio processing – using
CNN model, and user interface – as specified in interface.py. Audio is
read and processed concurrently using a queue system, while the GUI
updates the display with a risk indicator based on the filtered CNN
model predictions.

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
MODEL_PATH = 'app/models/model3-noise'

SAMPLE_RATE = 16000
NUM_MFCC = 13
NUM_FFT = 2048
HOP_LENGTH = 512

HOST = '127.0.0.1'
PORT = 6829
CHUNK = 1024 * 4

# output smoothing filter weights, should sum to 1
X_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2])
Y_WEIGHTS = np.array([0.2])


def read(audio_pipe, exit_event):
    '''
    Continuously reads data from either DISK, STREAM or MIC. Adds audio
    data to 'audio_pipe' FIFO pipeline in 1-second segments. Exits when
    'exit_event' is set.
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
            audio_pipe.put(audio_file[start_sample:end_sample])

            if exit_event.is_set():
                break

            # sleep remainder of 1-second cycle
            time.sleep(1 - (time.time() - end_time))
            end_time = time.time()

    if MODE == 'STREAM':

        # establish connection to socket server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((HOST, PORT))
            print('Connected to server',(HOST, PORT))

            data = b''
            payload_size = struct.calcsize('Q')

            while not exit_event.is_set():
                
                try:
                    # recieve payload_size number of bytes
                    while len(data) < payload_size:
                        packet = client_socket.recv(CHUNK)
                        if not packet: break
                        data+=packet
                        
                    # unpack bytes to determine message size
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack('Q',packed_msg_size)[0]

                    # recieve remainder of message if neccesary
                    while len(data) < msg_size:
                        data += client_socket.recv(CHUNK)

                    # de-serialize data from stream
                    frame_data = data[:msg_size]
                    data  = data[msg_size:]
                    frame = pickle.loads(frame_data)

                    # write 1 second of audio to queue
                    if len(frame) >= SAMPLE_RATE:
                        x = np.array(frame[:SAMPLE_RATE])
                        audio_pipe.put(x)

                except Exception as e:
                    print(e)
                    break

    if MODE == 'MIC':
        pass


def process(audio_pipe, predict_pipe, risk_pipe, update_event, exit_event):
    '''
    Processes audio data from 'audio_pipe' as soon as it is available.
    MFCC is calculated for each audio clip, which is fed to CNN, which
    makes a prediction. Exits when 'exit_event' is set.
    '''

    # load CNN
    model = keras.models.load_model(MODEL_PATH)

    # prediction filter input and output buffers
    x = np.full(len(X_WEIGHTS), 0.0)
    y = np.full(len(Y_WEIGHTS), 0.0)

    while not exit_event.is_set() or not audio_pipe.empty():

        # extract mfcc
        clip = audio_pipe.get()
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
        risk = y[0]

        # send data to GUI
        predict_pipe.put(prediction)
        risk_pipe.put(risk)
        update_event.set()


if __name__ == '__main__':

    # get MODE argument
    if len(sys.argv) == 2:
        MODE = sys.argv[1].upper()

    # get server IPv4 address argument
    elif len(sys.argv) == 3:
        MODE = sys.argv[1].upper()
        HOST = sys.argv[2]

    audio_pipe = queue.Queue(maxsize=10) # input audio queue
    predict_pipe = queue.Queue(maxsize=10) # output model prediction queue
    risk_pipe = queue.Queue(maxsize=10) # output risk indicator queue

    update_event = threading.Event() # update display event flag
    exit_event = threading.Event() # exit program event flag

    try:
        # run read() & process() concurrently
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        executor.submit(read, audio_pipe, exit_event)
        executor.submit(process, audio_pipe, predict_pipe, risk_pipe, update_event, exit_event)

        # load user interface
        app = QApplication(sys.argv)
        win = AlertUI(update_event, predict_pipe, risk_pipe)
        app.exec()

        # set exit flag once GUI is closed (after app.exec() returns)
        exit_event.set()

    except KeyboardInterrupt:
        exit_event.set()
