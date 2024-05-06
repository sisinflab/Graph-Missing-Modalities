#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import datetime

import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--model', type=str, default='vbpr', help='choose the model')
parser.add_argument('--batch_size_jobs', type=int, default=5, help='batch size for jobs')
parser.add_argument('--cluster', type=str, default='mesocentre', help='cluster name')

args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})


def summary(configuration):
    final_list = [('%s=%s' % (k[2:], v)) for (k, v) in configuration.items()]
    return '_'.join(final_list)


def to_cmd(c):
    command = ' '.join([f'{k} {v}' for k, v in c.items()])
    return command


def to_logfile(c):
    outfile = "{}.log".format(summary(c).replace("/", "_"))
    return outfile


def main():
    logs_path = 'logs'
    scripts_path = 'scripts'

    if not os.path.exists(logs_path + f'/{args.dataset}/{args.model}/'):
        os.makedirs(logs_path + f'/{args.dataset}/{args.model}/')

    if not os.path.exists(scripts_path + f'/{args.dataset}/{args.model}/'):
        os.makedirs(scripts_path + f'/{args.dataset}/{args.model}/')

    command_lines = set()

    for hyperparam in hyperparams:
        logfile = to_logfile(hyperparam)
        completed = False
        if os.path.isfile(f'{logs_path}/{args.dataset}/{args.model}/{logfile}'):
            with open(f'{logs_path}/{args.dataset}/{args.model}/{logfile}', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Best Model params' in content

        if not completed:
            command_line = (f'python run_multimodal_cluster.py {to_cmd(hyperparam)} '
                            f'--dataset {args.dataset} '
                            f'--method feat_prop '
                            f'--model {args.model} > {logs_path}/{args.dataset}/{args.model}/{logfile} 2>&1')
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    if args.batch_size_jobs == -1:
        args.batch_size_jobs = nb_jobs

    if args.cluster == 'margaret':
        header = None
    else:
        header = """#!/bin/bash -l

#SBATCH --output=/workdir/%u/slogs/missing_multimod-%A_%a.out
#SBATCH --error=/workdir/%u/slogs/missing_multimod-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB # memory in Mb
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH --time=4:00:00 # time requested in days-hours:minutes:seconds
#SBATCH --array=1-{0}%100

echo "Setting up bash environment"
source ~/.bashrc
set -x

# Modules
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Conda environment
source activate missing_multimod

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/Graph-Missing-Modalities

"""

    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    if header:
        for index, offset in enumerate(range(0, nb_jobs, args.batch_size_jobs), 1):
            offset_stop = min(offset + args.batch_size_jobs, nb_jobs)
            with open(scripts_path + f'/{args.dataset}/{args.model}/' + date_time + f'__{index}.sh', 'w') as f:
                print(header.format(offset_stop - offset), file=f)
                current_command_lines = sorted_command_lines[offset: offset_stop]
                for job_id, command_line in enumerate(current_command_lines, 1):
                    print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}', file=f)


if __name__ == '__main__':
    main()
