'''
Runs on the Raspberry Pi. Reads audio from external USB audio interface
or audio file on disk. Establishes socket connection with remote client
and streams audio over connection. Call file from command line with an
argument of 'DISK' or 'MIC' to configure operating mode. Programs runs
in 'DISK' mode by default.
'''


import socket
import wave
import pyaudio
import pickle
import struct
import librosa
import time
import math
import sys
import numpy as np
import scipy.signal as sps


# MODE OPTIONS
# 'DISK'    : sources audio from local audio file
# 'MIC'     : sources audio from PC audio input

MODE = 'DISK'

AUDIO_PATH = 'app/data/full-clips/001A-short.wav'

HOST = ''
PORT = 6829

FORMAT = pyaudio.paFloat32
CHANNELS = 1
SAMPLE_RATE = 16000
USB_SAMPLE_RATE = 48000
DEVICE_INDEX = 2


def transmit():
    '''
    '''
    
    with socket.socket() as server_socket:

        server_socket.bind((HOST, PORT))
        server_socket.listen()

        if MODE == 'DISK':

            print('Audio file laoding...')
            audio_file, fs = librosa.load(AUDIO_PATH, sr=USB_SAMPLE_RATE)
            print('Audio file loaded.')

            print('Server listening at',(HOST, PORT))
            client_socket,addr = server_socket.accept()
            print('Connected to clinet',addr)

            end_time = time.time()

            for i in range(math.floor(len(audio_file)/USB_SAMPLE_RATE)):

                if not client_socket:
                    break

                # parse 1 second of audio to stream
                start_sample = int(i*USB_SAMPLE_RATE)
                end_sample = int((i+1)*USB_SAMPLE_RATE)
                data = audio_file[start_sample:end_sample]

                # pack data for socket transmission
                a = pickle.dumps(data)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)

                # sleep remainder of 1-second cycle
                time.sleep(1 - (time.time() - end_time))
                end_time = time.time()

        if MODE == 'MIC':

            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                input_device_index=DEVICE_INDEX,
                rate=USB_SAMPLE_RATE,
                input=True,
                frames_per_buffer=SAMPLE_RATE)

            print('Server listening at',(HOST, PORT))
            client_socket,addr = server_socket.accept()
            print('Connected to clinet',addr)

            while client_socket:

                # read, convert & resample bytes from audio stream
                raw_data = stream.read(USB_SAMPLE_RATE, exception_on_overflow=False)
                np_data = np.frombuffer(raw_data, dtype=np.float32)
                data = sps.decimate(np_data, int(USB_SAMPLE_RATE/SAMPLE_RATE))

                # pack data for socket transmission
                a = pickle.dumps(data)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        MODE = sys.argv[1]
        
    transmit()
