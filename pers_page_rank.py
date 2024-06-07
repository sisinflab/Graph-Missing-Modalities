#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import datetime

import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Digital_Music', help='choose the dataset')
parser.add_argument('--model', type=str, default='vbpr', help='project name')

args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "--a": [0.1]
})


def summary(configuration):
    final_list = [('%s=%s' % (k[2:], v)) for (k, v) in configuration.items()]
    return '_'.join(final_list)


def to_cmd(c):
    command = ' '.join([f'{k}={v}' for k, v in c.items()])
    return command


def to_logfile(c):
    outfile = "{}.log".format(summary(c).replace("/", "_"))
    return outfile


def main():
    logs_path = 'logs'

    if not os.path.exists(logs_path + f'/{args.data}/pers_page_rank/{args.model}/'):
        os.makedirs(logs_path + f'/{args.data}/pers_page_rank/{args.model}/')

    command_lines = set()

    script = 'run_multimodal.py'

    for hyperparam in hyperparams:
        logfile = to_logfile(hyperparam)
        completed = False
        if os.path.isfile(f'{logs_path}/{args.data}/pers_page_rank/{args.model}/{logfile}'):
            with open(f'{logs_path}/{args.data}/pers_page_rank/{args.model}/{logfile}', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = ('Best Model params' in content)

        if not completed:
            command_line = (
                f'CUBLAS_WORKSPACE_CONFIG=:4096:8 python {script} {to_cmd(hyperparam)} '
                f'--data={args.data} '
                f'--method=pers_page_rank '
                f'--model={args.model}')
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    with open(f'run_multimodal_all_pers_page_rank_{args.data}_{args.model}.sh', 'w') as f:
        print(f'#!/bin/bash', file=f)
        for command_line in sorted_command_lines:
            print(f'{command_line}', file=f)


if __name__ == '__main__':
    main()
