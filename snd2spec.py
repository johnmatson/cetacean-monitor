import librosa, librosa.display 
import numpy as np
import matplotlib.pyplot as plt



audio_test = "audio_test.wav"

#capture waveform of signal
audio_sig, sr = librosa.load(audio_test,sr=22050)
librosa.display.waveplot(audio_sig, sr=sr) 
plt.figure(1)
plt.xlabel("Time")
plt.xlabel("Amplitude")
plt.show()

#fft: capture spectrum of signal
fft = np.fft.fft(audio_sig)

mag = np.abs(fft)
freq = np.linspace(0,sr,len(mag))

#remove frequency symmetry
lmag = mag[:int(len(freq)/2)]
lfreq = freq[:int(len(freq)/2)]


plt.figure(2)
plt.plot(lfreq,lmag)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#stft: spectrogram
nfft = 2048
hop_len = 512

stft = librosa.core.stft(audio_sig, hop_length=hop_len, n_fft=nfft)
spec = np.abs(stft)

log_spec = librosa.amplitude_to_db(spec)

plt.figure(3)
librosa.display.specshow(log_spec,sr=sr,hop_length=hop_len)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

#mel-frequency cepstral coefficients (MFFCs): used for dataset matching
plt.figure(4)
MFCCs = librosa.feature.mfcc(audio_sig,sr=sr,hop_length=hop_len)
librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_len)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()



















