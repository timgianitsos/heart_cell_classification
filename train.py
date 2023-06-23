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

def main(multiprocess_id, args, train_dataset, dev_dataset):
    '''
    The `multiprocess_id` is automatically supplied if this function is
    provided to `torch.multiprocessing.spawn`. We assume the ids start at 0 and
    increment for every new process.

    If this function is not being invoked by `torch.multiprocessing.spawn` (and
    thus only running from a single process), this function should manually
    suppled `multiprocess_id`  with a value of `0`.
    '''
    if not 0 <= multiprocess_id < len(args._derived['devices']):
        raise ValueError(
            f'A process id of {multiprocess_id} was passed but there are only '
            f'{len(args._derived["devices"])} possible '
            f'devices to distribute the model between, so any id outside the '
            f'range of 0-{len(args._derived["devices"]) - 1} '
            f'inclusive are not permitted.'
        )
    if args._derived['devices'] != ['cpu'] and not all(
            e[:5] == 'cuda:' and 0 <= int(e[5:]) < torch.cuda.device_count()
            for e in args._derived['devices']):
        raise ValueError(
            f'Acceptable values for `args._derived["devices"]` are ["cpu"] or '
            f'a list like ["cuda:0", "cuda:1", ...] where each id is a number '
            f'less than the number of available GPUs, '
            f'{torch.cuda.device_count()}'
        )

    using_multiprocess = len(args._derived['devices']) > 1
    device = args._derived['devices'][multiprocess_id]
    model, model_args = get_model(args.model_load_path)
    model = model.to(device)
    model.train()
    if device != torch.device('cpu'):
        # Set default GPU id for `tensor.to('cuda')`
        # (the default is normally gpu 0)
        torch.cuda.set_device(device)
        if using_multiprocess:
            torch.distributed.init_process_group(
                backend='nccl', world_size=len(args._derived['devices']),
                rank=multiprocess_id, init_method='tcp://127.0.0.1:12345'
            )
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device],
            )

    # TODO allow for loading optimizer from checkpoint
    # TODO Consider `ZeroRedundancyOptimizer`
    opt_cls = getattr(torch.optim, args.optimizer)
    if opt_cls is torch.optim.Adam:
        opt_kwargs = {'betas': (args.adam_beta1, args.adam_beta2)}
    else:
        opt_kwargs = {}
    opt = opt_cls(model.parameters(), lr=args.lr, **opt_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, # TODO shuffle should be True, but this is incompatible with the sampler, so we should shuffle the data on disk instead before it is loaded
        sampler=(
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if using_multiprocess else None
        ),
        # TODO Consider `drop_last` and `pin_memory
    )
    # TODO ensure that the steps per evaluation is less than the number
    # of steps in a batch
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=(
            torch.utils.data.distributed.DistributedSampler(dev_dataset)
            if using_multiprocess else None
        ),
        # TODO Consider `drop_last` and `pin_memory
    )

    ckpt_paths = deque()
    # TODO with multiple processes, the loggin about how many batches have been
    # processed will be incorrect
    logger = TrainLogger(args, len(train_loader), multiprocess_id, phase=None)
    logger.log_hparams(args)

    for epoch in range(1, args.num_epochs + 1):
        logger.start_epoch()
        # TODO should `tqdm` take other processes into account?
        for inp, target in tqdm(train_loader, dynamic_ncols=True, disable=(
                multiprocess_id != 0)):
            logger.start_iter()
            opt.zero_grad()

            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)
            # TODO cross entropy is only the same thing as KL divergence when
            # the reference distribution has entropy 0 (i.e. single label).
            # Should KL divergence itself be used for samples with multiple
            # labels?
            loss = F.cross_entropy(out, target)
            loss.backward()
            logger.log_iter(len(inp), loss, model, dev_loader, args)
            opt.step()

            logger.end_iter()
        logger.end_epoch()

        # TODO put this logic inside the .end_epoch() function?
        if (args.save_dir_root and epoch % args.epochs_per_model_save == 0 and
                multiprocess_id == 0):
            samples_processed = (epoch + 1) * len(train_loader)
            m = model.module if using_multiprocess else m
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

    if using_multiprocess:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    args = ArgParser().parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args, *get_train_dev_datasets(
            args.dataset_root, args.ratio_train_set_to_whole
        )),
        nprocs=len(args._derived['devices']),
        join=True,
    )
