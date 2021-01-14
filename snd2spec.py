import librosa, librosa.display 
import numpy as np
import matplotlib.pyplot as plt



audio_test = "audio_test.wav"


audio_sig, sr = librosa.load(audio_test,sr=44100)
librosa.display.waveplot(audio_sig, sr=sr) 
plt.xlabel("Time")
plt.xlabel("Amplitude")
plt.show()



























