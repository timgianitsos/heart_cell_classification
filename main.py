from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arg_parser import ArgParser
from dataset import FluorescenceTimeSeriesDataset
from layer.resnet1d import ResNet1d
from logger import TrainLogger

def get_model():
    '''
    Model arguments found here:
    https://github.com/antonior92/ecg-age-prediction/blob/f9801bbe7eb2ce8c5416f5d3d4182c7302813dec/train.py#L182-L183

    and here:
    https://www.dropbox.com/s/thvqwaryeo8uemo/model.zip?file_subpath=%2Fmodel%2Fconfig.json
    '''
    seq_length = 4096
    net_filter_size = [64,128,196,256,320]
    net_seq_length = [4096,1024,256,64,16]
    N_CLASSES = 1
    N_LEADS = 12
    kernel_size = 17
    dropout_rate = 0.8

    ckpt_dir = Path('checkpoints')
    ckpt = torch.load(ckpt_dir / 'model.pth')

    model = ResNet1d(input_dim=(N_LEADS, seq_length),
        blocks_dim=list(zip(net_filter_size, net_seq_length)),
        n_classes=N_CLASSES,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(ckpt['model'])
    return model

def main():
    args = ArgParser().parse_args()

    model = get_model().to(args._derived['devices'][0])
    if args._derived['devices'][0] != 'cpu':
        model = nn.DataParallel(model, device_ids=args._derived['devices'], output_device=args._derived['devices'][0])

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = torch.utils.data.DataLoader(
        FluorescenceTimeSeriesDataset(args.dataset_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    ckpt_paths = deque()
    logger = TrainLogger(args, len(loader), phase=None)
    logger.log_hparams(args)

if __name__ == '__main__':
    main()
