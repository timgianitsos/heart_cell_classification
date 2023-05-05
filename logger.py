import os
from pathlib import Path
from datetime import datetime
from time import time

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
from torchnet.meter import AverageValueMeter
import torch.nn.functional as F
import torchvision.utils as vutils

class BaseLogger(object):
    def __init__(self, args, dataset_len):

        self.name = args._derived['full_name']
        self.save_dir = args._derived.get('save_dir_current', None)
        
        self.dataset_len = dataset_len

        if self.save_dir:
            tb_dir = Path('/'.join(self.save_dir.split('/')[:-1])) / 'tb' / self.name
            os.makedirs(tb_dir, exist_ok=True)
            self.log_filepath = os.path.join(self.save_dir, f'{self.name}.log')
        else:
            tb_dir = None
        self.summary_writer = SummaryWriter(log_dir=tb_dir, write_to_disk=self.save_dir is not None)

        self.epoch = 0
        self.iter = 0

        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = 0
        self.epoch_start_time = None

    def _log_text(self, text_dict):
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)

    def _log_scalars(self, scalar_dict, print_to_stdout=True, unique_id=None):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write(f'[{k}: {v}]')
            k = k.replace('_', '/')  # Group in TensorBoard by split.
            if unique_id is not None:
                k = f'{k}/{unique_id}'
            self.summary_writer.add_scalar(k, v, self.global_step)

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        if self.save_dir:
            with open(self.log_filepath, 'a') as log_file:
                log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError

class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""

    def __init__(self, args, dataset_len, phase=None):
        super(TrainLogger, self).__init__(args, dataset_len)
       
        # Tag suffix used for indicating training phase in loss + viz
        self.tag_suffix = phase
        
        self.num_epochs = args.num_epochs

        self.train_loss_meter = AverageValueMeter()
        self.dev_loss_meter = AverageValueMeter()
        
        self.steps_per_dev_eval = args.steps_per_dev_eval

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def start_iter(self):
        """Log info for start of an iteration."""
        pass

    # TODO function takes too many arguments
    def log_iter(self, batch_size, train_loss, model, dev_loader, args):
        train_loss = train_loss.item()
        self.train_loss_meter.add(train_loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_dev_eval == 0:

            model.eval()
            with torch.inference_mode():
                for inp, target in dev_loader:
                    inp = inp.to(args._derived['devices'][0])
                    target = target.to(args._derived['devices'][0])
                    out = model(inp)
                    dev_loss = F.cross_entropy(out, target).item()
                    self.dev_loss_meter.add(dev_loss, len(inp))
            model.train()

            # Write a header for the log entry
            message = f"[epoch: {self.epoch}, iter: {self.iter} / {self.dataset_len}, train_loss: {self.train_loss_meter.mean:.3g}, test_loss: {self.dev_loss_meter.mean:.3g}]"
            self.write(message)

            # Write all errors as scalars to the graph
            # TODO consider plotting loss std
            self._log_scalars({
                'TrainLoss': self.train_loss_meter.mean,
                'DevLoss': self.dev_loss_meter.mean,
            }, print_to_stdout=False, unique_id=self.tag_suffix)
            self.train_loss_meter.reset()
            self.dev_loss_meter.reset()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += 1
        self.global_step += 1

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write(f'[start of epoch {self.epoch}]')

    def end_epoch(self):
        """Log info for end of an epoch."""
        epoch_time = time() - self.epoch_start_time
        self.write(f'[end of epoch {self.epoch}, epoch time: {epoch_time:.2g} seconds]')
        self.epoch += 1
