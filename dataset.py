from pathlib import Path

import numpy as np
import scipy.fft as fft
from scipy.signal import resample
import torch

def preprocess(a):
    # Model was originally trained on sequences of 10 seconds at about 400 Hz.
    # Fluorescence cell data is 10 seconds at about 100 Hz.
    # Since the time period that each was recorded at are both 10 seconds,
    # we can simply resample the Hz.
    a = resample(a, 4096, axis=1)

    # One of the datasets that the model was validated on was "Sami-Trop"
    # https://zenodo.org/record/4905618#.ZEQ2mS2B30o
    # I (Tim) noticed that the median across all samples was exactly 0.
    # This is strong evidence that the data was median-centered (as opposed to
    # mean-centered or no centering altogether) before being passed to the
    # model. I decided to median-center each time series by its median (as
    # opposed to center all time series by the same median computed across all
    # samples which is likely what "Sami-Trap" did).
    a -= np.median(a, axis=1)[:, None]

    # Utilize Discrete Cosine Transform Type II to smooth out waveform
    # https://en.wikipedia.org/wiki/Discrete_cosine_transform#Informal_overview
    a_fft = fft.dct(a, type=2, norm='ortho', axis=1)
    a_fft[:, 200:] = 0  # this is a manually chosen frequency cutoff
    a_smooth = fft.idct(a_fft, type=2, norm='ortho', axis=1)
    return a_smooth

class FluorescenceTimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, data_root):
        '''
        The cell fluorescense data is a timeseries. The model was trained on a
        somewhat similar timeseries (EKG) but at different Hz and with each
        input consisting of 12 time series (one for each lead).

        We attain the cell data, perform some signal preprocessing, and format
        it to be ingestible by the model.
        '''
        data_root = Path(data_root)
        dwl = np.load(data_root / 'data-with-labels.npz')

        preproc_filename = data_root / 'preprocessed_fluorescence_intensities.npz'
        if preproc_filename.exists():
            print(f'Loading data from "{preproc_filename}"... ', end='')
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

        self.inputs = torch.from_numpy(fl)
        self.outputs = torch.from_numpy((
            dwl['labels'] / dwl['labels'].sum(axis=1)[:, None]
        ).astype(np.float32))
        self.padding = torch.zeros(
            (11, self.inputs.shape[-1]), dtype=self.inputs.dtype
        )

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Required: specify what each iteration in dataloader yields

        The model expects 12 time series per input, but each cell's
        fluorescence data is only a single time series. We expand the array to
        have 12 time series but 11 of them will be 0's.
        """
        inps = torch.cat([self.inputs[index].reshape(1, -1), self.padding])
        return inps, self.outputs[index]

