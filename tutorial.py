import torch
import torchaudio
import requests
import matplotlib.pyplot as plt
import numpy as np


waveforms, spectrograms, mfccs = [], [], []
fig, ax = plt.subplots(nrows=3, ncols=10, gridspec_kw = {'wspace':0, 'hspace':0})
for i in range(10):
    filename = "data/{}-11-11.wav".format(i)
    waveform, sample_rate = torchaudio.load(filename)
    spectro = torchaudio.transforms.MelSpectrogram()(waveform)
    mfcc = torchaudio.transforms.MFCC()(waveform)
    

    ax[0][i].plot(waveform.t().numpy())
    ax[1][i].plot(spectro.log2()[0,:,:].numpy())
    ax[2][i].plot(mfcc[0,:,:].t().numpy())

    waveforms.append(waveform)
    spectrograms.append(spectro)
    mfccs.append(mfcc)

fig.tight_layout()
plt.show()


