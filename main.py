from collections import deque
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arg_parser import ArgParser
from dataset import get_train_dev_datasets
from model import get_model
from logger import TrainLogger

def main():
    args = ArgParser().parse_args()

    model, model_args = get_model(args.model_load_path)
    model = model.to(args._derived['devices'][0])
    model.train()
    if args._derived['devices'][0] != 'cpu':
        model = nn.DataParallel(
            # TODO consider using DistributedDataParallel in the future
            model,
            device_ids=args._derived['devices'],
            output_device=args._derived['devices'][0]
        )

    # TODO allow for loading optimizer from checkpoint
    opt_cls = getattr(torch.optim, args.optimizer)
    if opt_cls is torch.optim.Adam:
        opt_kwargs = {'betas': (args.adam_beta1, args.adam_beta2)}
    else:
        opt_kwargs = {}
    opt = opt_cls(model.parameters(), lr=args.lr, **opt_kwargs)

    train_dataset, dev_dataset = get_train_dev_datasets(
        args.dataset_root, args.ratio_train_set_to_whole
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    ckpt_paths = deque()
    logger = TrainLogger(args, len(train_loader), phase=None)
    logger.log_hparams(args)

    for epoch in range(1, args.num_epochs + 1):
        logger.start_epoch()
        for inp, target in tqdm(train_loader, dynamic_ncols=True):
            logger.start_iter()
            opt.zero_grad()

            inp = inp.to(args._derived['devices'][0])
            target = target.to(args._derived['devices'][0])
            out = model(inp)
            loss = F.cross_entropy(out, target)
            loss.backward()
            opt.step()

            logger.log_iter(len(inp), loss)
            logger.end_iter()
        logger.end_epoch()

        # TODO put this logic inside the .end_epoch() function?
        if args.save_dir_root and epoch % args.epochs_per_model_save == 0:
            samples_processed = (epoch + 1) * len(train_loader)
            m = model.module if args._derived['devices'][0] != 'cpu' else m
            ckpt_dict = {
                'ckpt_info': {'samples_processed': samples_processed},
                'model_name': m.__class__.__name__,
                'model_state': m.state_dict(),
                'model_args': model_args
            }
            ckpt_path = Path(
                args._derived['ckpt_dir']
            ) / f'step_{samples_processed}.pth'
            torch.save(ckpt_dict, ckpt_path)
            ckpt_paths.append(ckpt_path)
            if len(ckpt_paths) > args.max_ckpts:
                oldest_ckpt = ckpt_paths.popleft()
                os.remove(oldest_ckpt)

if __name__ == '__main__':
    main()
