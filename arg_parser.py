import argparse
from datetime import datetime
import json
import os
from os.path import dirname, join
import random
import subprocess
from sys import argv

import torch
import numpy as np

class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Model arguments')

        # Basic runtime settings
        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible outputs.')

        # System configurations
        self.parser.add_argument('--gpu_ids', type=str, default='0' if torch.cuda.is_available() else '-1', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=1, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--dataset_root', type=str, default=join(dirname(argv[0]), 'data'), help='The root of the dataset directory')

        # Model hyperparameters
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')

        # Checkpointing
        self.parser.add_argument('--save_dir_root', type=lambda x: None if x == 'None' else x, default=None, help='Directory for results, prefix. Use `None` to neglect outputs (for debugging)')
        self.parser.add_argument('--model_load_path', type=str, default=join(dirname(argv[0]), 'checkpoints', 'model.pth'), help='Load from a previous checkpoint.')
        self.parser.add_argument('--num_visuals', type=str, default=10, help='Number of visual examples to show per batch on Tensorboard.')
        self.parser.add_argument('--max_ckpts', type=int, default=15, help='Max ckpts to save.')
        self.parser.add_argument('--steps_per_print', type=int, default=50, help='Steps taken for each print of logger')
        self.parser.add_argument('--steps_per_visual', type=int, default=400, help='Steps for each visual to be printed by logger in tensorboard')
        self.parser.add_argument('--epochs_per_model_save', type=int, default=10, help='Epochs for a model checkpoint to be saved')

    def parse_args(self):
        args = self.parser.parse_args()
        args._derived = {}

        # Get version control hash for record-keeping
        args._derived['commit_hash'] = subprocess.run(
            ['git', '-C', join('.', dirname(__file__)), 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout.strip()

        # This appends, if necessary, a message about there being uncommitted
        # changes (i.e. if there are uncommitted changes, you can't be sure
        # exactly what the code looks like, whereas if there are no uncommitted
        # changes, you know exactly what the code looked like).
        args._derived['commit_hash'] += ' (with uncommitted changes)' if bool(
            subprocess.run(
                [
                    'git', '-C', join('.', dirname(__file__)),
                    'status', '--porcelain'
                ],
                stdout=subprocess.PIPE,
                universal_newlines=True,
            ).stdout.strip()
        ) else ''

        # Create save dir for run
        args._derived['full_name'] = (
            f'{args.name}_{datetime.now().strftime("%b%d_%H%M%S")}'
            f'_{os.getlogin()}'
        )
        if args.save_dir_root:
            # Create sub directories
            save_dir_current = join(
                args.save_dir_root, args._derived['full_name']
            )
            args._derived['save_dir_current'] = save_dir_current
            os.makedirs(save_dir_current, exist_ok=False)
            args._derived['ckpt_dir'] = join(save_dir_current, 'ckpts')
            os.makedirs(args._derived['ckpt_dir'], exist_ok=False)

        # Convert comma-separated arguments to a list
        args_to_list = lambda csv, arg_type=int: [
            arg_type(d) for d in str(csv).split(',')
        ]

        gpu_ids = args_to_list(args.gpu_ids)
        if gpu_ids != [-1] and not all(
            0 <= v < torch.cuda.device_count() for v in gpu_ids
        ):
            raise Exception(
                f'The option --gpu_ids={gpu_ids} is invalid. '
                f'The only valid options for `--gpu_ids` are -1 (indicating '
                f'the use of CPU) or comma separated value(s) in the range '
                f'[0, {torch.cuda.device_count()}) (where the right bound is '
                f'the number of available CUDA devices). '
            )
        elif gpu_ids == [-1]:
            args._derived['devices'] = ['cpu']
        else:
            args._derived['devices'] = [f'cuda:{i}' for i in gpu_ids]
            # Set default GPU id for `tensor.to('cuda')`
            # (the default is normally `0`)
            torch.cuda.set_device(args._derived['devices'][0])

        # Save args to a JSON file
        if args.save_dir_root:
            with open(join(save_dir_current, 'args.json'), 'w') as fh:
                json.dump(vars(args), fh, indent=4, sort_keys=True)
                fh.write('\n')

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(json.dumps(vars(args), indent=4, sort_keys=True))
        return args
