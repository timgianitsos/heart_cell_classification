from collections import deque

import numpy as np
import torch
import torch.nn as nn

from arg_parser import ArgParser
from dataset import FluorescenceTimeSeriesDataset
from model import get_model
from logger import TrainLogger

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
