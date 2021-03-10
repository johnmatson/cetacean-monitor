from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import random
from skimage.measure import block_reduce

#To find the duration of wave file in seconds
import wave
import contextlib

#Keras imports
import tensorflow.keras as keras
import tensorflow as tf

import time
import datetime

imwidth                             = 50
imheight                            = 34
num_classes                         = 10
test_rec_folder                     = "./testrecs"
num_test_files                      = 1

THRESHOLD                           = 1000
CHUNK_SIZE                          = 512
FORMAT                              = pyaudio.paInt16
RATE                                = 16000
WINDOW_SIZE                         = 50
CHECK_THRESH                        = 3
SLEEP_TIME                          = 0.5 #(seconds)
IS_PLOT                             = 1

#Check for silence
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    return sample_width, r

def get_bounds(ds):
    """
    Extract relevant signal from the captured audio
    """    
    np.array(ds)
    lds = len(ds)
    count = 0
    ll=-1
    ul=-1

    #Lower Limit
    for i in range(0,lds,WINDOW_SIZE):
        sum = 0
        for k in range(i,(i+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum>THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ll = i - WINDOW_SIZE * CHECK_THRESH
            break
        
    #Upper Limit
    count = 0
    for j in range(i,lds,WINDOW_SIZE):
        sum = 0
        for k in range(j,(j+WINDOW_SIZE)%lds):
            sum = sum + np.absolute(ds[k])
        if(sum<THRESHOLD):
            count +=1
        if(count>CHECK_THRESH):
            ul = j - WINDOW_SIZE * CHECK_THRESH


        if(ul>0 and ll >0):
            break
    return ll, ul 


def record_to_file(path):
    """
    Records from the microphone and outputs the resulting data to 'path'
    """
    sample_width, data = record()
    ll, ul = get_bounds(data)
    print(ll,ul)
    if(ul-ll<100):
        return 0
    #nonz  = np.nonzero(data)
    ds = data[ll:ul]
    if(IS_PLOT):
        plt.plot(data)
        plt.axvline(x=ll)
        #plt.axvline(x=ll+5000)
        plt.axvline(x=ul)
        plt.show()

    #data = pack('<' + ('h'*len(data)), *data)
    fname = "0.wav"
    if not os.path.exists(path):
        os.makedirs(path)
    wf = wave.open(os.path.join(path,fname), 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(ds)
    wf.close()
    return 1

def findDuration(fname):
    """
    Function to find the duration of the wave file in seconds
    """
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration


def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    """
    Plot Spectrogram
    """
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)
    #print("")
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    #fig.savefig('sp_xyz.png', dpi=300, frameon='false')
    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    #print(size_inches, dpi, width, height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray

def rgb2gray(rgb):
    """
    Convert color image to grayscale
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def normalize_gray(array):
    """
    Normalize Gray colored image
    """
    return (array - array.min())/(array.max() - array.min())

def get_wav_data(path):
    input_wav           = path
    spectrogram         = graph_spectrogram( input_wav )
    graygram            = rgb2gray(spectrogram)
    normgram            = normalize_gray(graygram)
    norm_shape          = normgram.shape
    #print("Spec Shape->", norm_shape)
    if(norm_shape[0]>100):
        redgram             = block_reduce(normgram, block_size = (26,26), func = np.mean)
    else:
        redgram             = block_reduce(normgram, block_size = (3,3), func = np.mean)
    redgram             = redgram[0:imheight,0:imwidth]
    red_data            = redgram.reshape(imheight,imwidth, 1)
    empty_data          = np.empty((1,imheight,imwidth,1))
    empty_data[0,:,:,:] = red_data
    new_data            = empty_data
    return new_data

def load_model_from_disk():
    """
    Load saved model
    """
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

if __name__ == '__main__':
    while(1):
        time.sleep(SLEEP_TIME)
        if(os.path.isfile('model.json')):
            print("please speak a word into the microphone")
            success = record_to_file(test_rec_folder)
            if(not success):
                print(" Speak Again Clearly")
                continue

        model = load_model_from_disk()

        for i in range(num_test_files):
            fname = str(i)+".wav"
            new_data    = get_wav_data(os.path.join(test_rec_folder,fname))    
            predictions = np.array(model.predict(new_data))
            maxpred = predictions.argmax()
            normpred = normalize_gray(predictions)*100
            predarr = np.array(predictions[0])
            sumx = predarr.sum()
            print("TestFile Name: ", fname, " The Model Predicts:", maxpred)
            for nc in range(num_classes):
                confidence = np.round(100*(predarr[nc]/sumx))
                print("Class ", nc, " Confidence: ", confidence)
            #print("TestFile Name: ",fname, " Values:", predictions)
            print("_____________________________\n")