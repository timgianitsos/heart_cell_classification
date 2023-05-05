from pathlib import Path

import numpy as np
import scipy.fft as fft
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, random_split

def _preprocess(a):
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

def get_train_dev_datasets(data_root, ratio_train_set_to_whole):
    '''
    The cell fluorescense data is a timeseries. The model was trained on a
    somewhat similar timeseries (EKG) but at different Hz and with each
    input consisting of 12 time series (one for each lead).

    We attain the cell data, perform some signal preprocessing, and format
    it to be ingestible by the model.

    We then output a train and dev set.
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
        fl = _preprocess(fl)
        np.savez_compressed(
            preproc_filename,
            fluorescence_intensities=fl,
        )
        print(f'Done! Saved to "{preproc_filename}"')

    inputs = torch.from_numpy(fl)
    targets = torch.from_numpy((
        dwl['labels'] / dwl['labels'].sum(axis=1)[:, None]
    ).astype(np.float32))

    dataset = FluorescenceTimeSeriesDataset(inputs, targets)
    # TODO consider stratified sampling. This might be difficult since there
    # are not merely 12 possible labels but 2^12 because each waveform can
    # be assigned multiple of the 12 labels. However, still worth looking into.
    return random_split(dataset, [
        ratio_train_set_to_whole, 1 - ratio_train_set_to_whole
    ])

class FluorescenceTimeSeriesDataset(Dataset):

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Required: specify what each iteration in dataloader yields

        The model expects 12 time series per input, but each cell's
        fluorescence data is only a single time series. We reshape
        array to have a channel dimension of length one.
        """
        # TODO this class is only necessary because of the reshaping here.
        # Without it, we could just use torch.utils.data.TensorDataset
        # Consider a more elegant way to handle this reshaping upstreadm
        # to obviate the need for this class.
        return self.inputs[index].reshape(1, -1), self.targets[index]
