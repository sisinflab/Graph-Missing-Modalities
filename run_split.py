import os
import shutil

from elliot.run import run_experiment

import argparse

parser = argparse.ArgumentParser(description="Run split.")
parser.add_argument('--data', type=str, default='Beauty')
parser.add_argument('--final', type=str, default='no')
args = parser.parse_args()

if args.final == 'yes':
    if not (os.path.exists(f'./data/{args.data}/train_final.tsv') and os.path.exists(f'./data/{args.data}/val_final.tsv') and os.path.exists(f'./data/{args.data}/test_final.tsv')):
        run_experiment(f"config_files/split_final_{args.data}.yml")
        shutil.move(f'./data/{args.data}_splits/0/test.tsv', f'./data/{args.data}/test_final.tsv')
        shutil.move(f'./data/{args.data}_splits/0/0/train.tsv', f'./data/{args.data}/train_final.tsv')
        shutil.move(f'./data/{args.data}_splits/0/0/val.tsv', f'./data/{args.data}/val_final.tsv')
        shutil.rmtree(f'./data/{args.data}_splits/')
else:
    if not (os.path.exists(f'./data/{args.data}/train.tsv') and os.path.exists(f'./data/{args.data}/val.tsv') and os.path.exists(f'./data/{args.data}/test.tsv')):
        run_experiment(f"config_files/split_{args.data}.yml")
        shutil.move(f'./data/{args.data}_splits/0/test.tsv', f'./data/{args.data}/test.tsv')
        shutil.move(f'./data/{args.data}_splits/0/0/train.tsv', f'./data/{args.data}/train.tsv')
        shutil.move(f'./data/{args.data}_splits/0/0/val.tsv', f'./data/{args.data}/val.tsv')
        shutil.rmtree(f'./data/{args.data}_splits/')
