from pathlib import Path
from sys import argv, stderr
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import preprocess, FluorescenceTimeSeriesDataset
from model import ResNet1d

def save_model_guesses(model, device, fluorescence_intensities, output_file):
    print('Preprocessing dataset...')
    loader = torch.utils.data.DataLoader(
        FluorescenceTimeSeriesDataset(inputs=torch.from_numpy(
            preprocess(fluorescence_intensities)
        )),
        batch_size=450,
        num_workers=4,
        shuffle=False,
    )

    with torch.inference_mode():
        label_guesses = torch.concatenate([
            F.softmax(model(inp.to(device)), dim=1)
            for inp in tqdm(
                loader, desc=f'Generating predictions for {output_file}'
            )
        ]).cpu().numpy()

    np.save(output_file, label_guesses)

def main():
    if len(argv) <= 1:
        print(f'Usage: python3 {__file__} path/to/model.pth', file=stderr)
        return
    if not Path(argv[1]).exists():
        print(f'The provided path "{argv[1]}" does not exist')
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
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    data_root_dir = Path('data')
    save_model_guesses(
        model,
        device,
        np.load(
            data_root_dir / 'data-with-labels.npz'
        )['fluorescence_intensities'],
        data_root_dir / 'model-guesses-with-labels.npy'
    )
    save_model_guesses(
        model,
        device,
        np.load(
            data_root_dir / 'data-without-labels.npz'
        )['fluorescence_intensities'],
        data_root_dir / 'model-guesses-without-labels.npy'
    )

if __name__ == '__main__':
    main()
