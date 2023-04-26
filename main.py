from pathlib import Path

import numpy as np
import scipy.fft as fft
from scipy.signal import resample
import torch

from layer.resnet1d import ResNet1d

def get_model():
    ckpt_dir = Path('checkpoints')
    ckpt = torch.load(ckpt_dir / 'model.pth')

    seq_length = 4096
    net_filter_size = [64,128,196,256,320]
    net_seq_length = [4096,1024,256,64,16]
    N_CLASSES = 1
    N_LEADS = 12
    kernel_size = 17
    dropout_rate = 0.8

    model = ResNet1d(input_dim=(N_LEADS, seq_length),
        blocks_dim=list(zip(net_filter_size, net_seq_length)),
        n_classes=N_CLASSES,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(ckpt['model'])
    return model

def get_data():
    '''
    The cell fluorescense data is a timeseries. The model was trained on a
    somewhat similar timeseries (EKG) but at different Hz and with each input
    consisting of 12 time series (one for each lead).

    We attain the cell data, perform some signal preprocessing, and format it
    to be ingestible by the model.
    '''
    data_dir = Path('data')
    dwl = np.load(data_dir / 'data-with-labels.npz')

    preproc_filename = data_dir / 'preprocessed_fluorescence_intensities.npz'
    if preproc_filename.exists():
        print(f'Loading preprocessed data from "{preproc_filename}"... ', end='')
        fl = np.load(preproc_filename)['fluorescence_intensities']
        print(f'Done!')
    else:
        print(f'Preprocessing dataset... ', end='')
        fl = dwl['fluorescence_intensities']
        fl = preprocess(fl)
        np.savez_compressed(
            preproc_filename,
            fluorescence_intensities=fl,
            allow_pickle=False
        )
        print(f'Done! Saved to "{preproc_filename}"')

    return torch.from_numpy(fl), torch.from_numpy(dwl['labels'])

def preprocess(a):
    # Model was originally trained on sequences of 10 seconds at about 400 Hz.
    # Fluorescence cell data is 10 seconds at about 100 Hz.
    # Since the time period that each was recorded at are both 10 seconds,
    # we can simply resample the Hz with no additional processing.
    a = resample(a, 4096, axis=1)

    # One of the datasets that the model was validated on was "Sami-Trop"
    # https://zenodo.org/record/4905618#.ZEQ2mS2B30o
    # I (Tim) noticed that the median across all samples was exactly 0.
    # This is strong evidence that the data was median-centered (as opposed to
    # no preprocessing altogether or mean-centered) before being passed to the
    # model. I decided to median-center each time series by its median as
    # opposed to center all time series by the same median computed across all
    # samples.
    a -= np.median(a, axis=1)[:, None]

    # Utilize Discrete Cosine Transform Type II to smooth out waveform
    # https://en.wikipedia.org/wiki/Discrete_cosine_transform#Informal_overview
    a_fft = fft.dct(a, type=2, norm='ortho', axis=1)
    a_fft[:, 200:] = 0  # manually chosen frequency cutoff
    a_smooth = fft.idct(a_fft, type=2, norm='ortho', axis=1)
    return a_smooth

def main():
    model = get_model()
    inputs, outputs = get_data()

if __name__ == '__main__':
    main()
