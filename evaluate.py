from pathlib import Path
from sys import argv, stderr
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import preprocess, FluorescenceTimeSeriesDataset
from model import ResNet1d

def main():
    if len(argv) <= 2:
        print(f'Usage: python3 {__file__} path/to/model.pth path/to/data.npz', file=stderr)
        return
    if not Path(argv[1]).exists():
        print(f'The provided path "{argv[1]}" does not exist')
    if not Path(argv[2]).exists():
        print(f'The provided path "{argv[2]}" does not exist')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Load Model and manually override input layer and output layer specs
    # because the original spec was for EKG data, not cell fluorescence data
    model_info = torch.load(argv[1], map_location='cpu')
    model_args = model_info['model_args']
    model_args['input_dim'] = (1, model_args['input_dim'][1])
    model_args['n_classes'] = 12
    model = ResNet1d(**model_args)
    model.load_state_dict(model_info['model_state'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Load the data
    data = np.load(argv[2])
    loader = torch.utils.data.DataLoader(
        FluorescenceTimeSeriesDataset(inputs=torch.from_numpy(
            preprocess(data['fluorescence_intensities'])
        )),
        batch_size=450,
        num_workers=4,
        shuffle=False,
    )

    with torch.inference_mode():
        label_guesses_from_model = torch.concatenate([
            F.softmax(model(inp.to(device)), dim=1) for inp in tqdm(loader)
        ]).cpu().numpy()

    # We will compute accuracy. Since some samples have multiple labels,
    # we will only consider the samples with single labels.
    idx = data['labels'].sum(axis=1) == 1
    single_label_perf = data['labels'][idx].argmax(axis=1) == np.isclose(
        label_guesses_from_model, np.max(
            label_guesses_from_model, axis=1
        )[:, None]
    ).astype(np.int8)[idx].argmax(axis=1)
    print(
        f'Accuracy on single labels: {single_label_perf.sum()} / '
        f'{len(single_label_perf)} = {single_label_perf.mean():.3f}'
    )

if __name__ == '__main__':
    main()
