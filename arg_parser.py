import argparse
from datetime import datetime
import json
import os
from os.path import dirname, join
import random
import subprocess
from sys import argv, stderr

import torch
import numpy as np

class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Model arguments')

        # Basic runtime settings
        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible outputs.')

        # System configurations
        self.parser.add_argument('--gpu_ids', type=str, default='0,1' if torch.cuda.is_available() else '-1', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--dataset_root', type=str, default=join(dirname(argv[0]), 'data'), help='The root of the dataset directory')

        # Model hyperparameters
        self.parser.add_argument('--batch_size', type=int, default=450, help='Batch size per device')
        self.parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
        self.parser.add_argument('--ratio_train_set_to_whole', type=float, default=0.975, help='Ratio of training-set to training-set + dev-set')
        self.parser.add_argument('--optimizer', type=str, default='RMSprop', choices=[
            k for k, v in vars(torch.optim).items() if type(v) == type and issubclass(v, torch.optim.Optimizer)
        ], help='Available optimizers')
        if self.parser.parse_known_args()[0].optimizer == 'Adam':
            self.parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta 1, only applies to Adam optimizer')
            self.parser.add_argument('--adam_beta2', type=float, default=0.999, help='Beta 2, only applies to Adam optimizer')

        # Checkpointing
        self.parser.add_argument('--model_load_path', type=str, default=join(dirname(argv[0]), 'checkpoints', 'model-pretrained.pth'), help='Load from a previous checkpoint.')
        self.parser.add_argument('--steps_per_dev_eval', type=int, default=15, help='Batches processed for each print of logger and evaluation of dev step.')
        self.parser.add_argument('--save_dir_root', type=lambda x: None if x == 'None' else x, default=None, help='Directory for results, prefix. Use `None` to neglect outputs (for debugging)')
        if self.parser.parse_known_args()[0].save_dir_root:
            self.parser.add_argument('--max_ckpts', type=int, default=3, help='Max ckpts to save.')
            self.parser.add_argument('--epochs_per_model_save', type=int, default=100, help='Epochs for a model checkpoint to be saved')

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
            f'{os.getlogin()}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f'_{args.name}'
        )

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

        # Save args to a JSON file
        if args.save_dir_root:
            if args.num_epochs % args.epochs_per_model_save:
                raise ValueError(
                    f'The total number of epochs {args.num_epochs} is not '
                    f'divisible by the number of epochs that happen per '
                    f'model save checkpoint which is '
                    f'{args.epochs_per_model_save}. This means that the last '
                    f'{args.num_epochs % args.epochs_per_model_save} epoch(s) '
                    f'would be wasted effort since there is no checkpoint '
                    f'at the end.'
                )
            # Create sub directories
            save_dir_current = join(
                args.save_dir_root, args._derived['full_name']
            )
            args._derived['save_dir_current'] = save_dir_current
            os.makedirs(save_dir_current, exist_ok=False)
            args._derived['ckpt_dir'] = join(save_dir_current, 'ckpts')
            os.makedirs(args._derived['ckpt_dir'], exist_ok=False)
            with open(join(save_dir_current, 'args.json'), 'w') as fh:
                json.dump(vars(args), fh, indent=4, sort_keys=True)
                fh.write('\n')
        else:
            print(
                '\nWARNING: Since --save_dir_root is not set, neither the model '
                'weights nor the model metrics will be saved\n',
                file=stderr
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(json.dumps(vars(args), indent=4, sort_keys=True))
        return args
