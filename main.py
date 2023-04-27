from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arg_parser import ArgParser
from dataset import FluorescenceTimeSeriesDataset
from model import get_model
from logger import TrainLogger

def main():
    args = ArgParser().parse_args()

    model, model_args = get_model(args.model_load_path)
    model = model.to(args._derived['devices'][0])
    if args._derived['devices'][0] != 'cpu':
        model = nn.DataParallel(
            model,
            device_ids=args._derived['devices'],
            output_device=args._derived['devices'][0]
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = torch.utils.data.DataLoader(
        FluorescenceTimeSeriesDataset(args.dataset_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    ckpt_paths = deque()
    logger = TrainLogger(args, len(loader), phase=None)
    logger.log_hparams(args)

    for epoch in range(args.num_epochs):
        logger.start_epoch()
        for inp, target in tqdm(loader, dynamic_ncols=True):
            logger.start_iter()
            opt.zero_grad()

            inp = inp.to(args._derived['devices'][0])
            target = target.to(args._derived['devices'][0])
            out = model(inp)
            loss = F.cross_entropy(out, target)
            loss.backward()

            opt.step()
            logger.log_iter_classifier(args.batch_size, loss)
            logger.end_iter()
        logger.end_epoch()

    if args.save_dir_root and epoch % args.epochs_per_model_save == 0:
        samples_processed = (epoch + 1) * len(loader)
        m = model.module if args._derived['devices'][0] != 'cpu' else m
        ckpt_dict = {
            'ckpt_info': {'samples_processed': samples_processed},
            'model_name': m.__class__.__name__,
            'model_state': m.state_dict(),
            'optimizer': opt.state_dict(),
            'model_args': model_args
        }
        ckpt_path = (
            Path(args._derived['ckpt_dir']) / f'step_{images_processed}.pth'
        )
        torch.save(ckpt_dict, ckpt_path)
        ckpt_paths.append(ckpt_path)
        if len(ckpt_paths) > args.max_ckpts:
            oldest_ckpt = ckpt_paths.popleft()
            os.remove(oldest_ckpt)

if __name__ == '__main__':
    main()
