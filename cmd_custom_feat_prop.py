#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import datetime

import argparse
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Digital_Music', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose the gpu id')
parser.add_argument('--batch_size_jobs', type=int, default=5, help='batch size for jobs')
parser.add_argument('--cluster', type=str, default='cineca', help='cluster name')
parser.add_argument('--mail_user', type=str, default='', help='your email')
parser.add_argument('--account', type=str, default='', help='project name')
parser.add_argument('--model', type=str, default='vbpr', help='project name')
parser.add_argument('--partition', type=str, default='', help='partition name')

args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
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
    scripts_path = 'scripts'

    if not os.path.exists(logs_path + f'/{args.dataset}/feat_prop/{args.model}/'):
        os.makedirs(logs_path + f'/{args.dataset}/feat_prop/{args.model}/')

    if not os.path.exists(scripts_path + f'/{args.dataset}/feat_prop/{args.model}/'):
        os.makedirs(scripts_path + f'/{args.dataset}/feat_prop/{args.model}/')

    command_lines = set()

    for hyperparam in hyperparams:
        logfile = to_logfile(hyperparam)
        completed = False
        if os.path.isfile(f'{logs_path}/{args.dataset}/feat_prop/{args.model}/{logfile}'):
            with open(f'{logs_path}/{args.dataset}/feat_prop/{args.model}/{logfile}', 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = ('Best Model params' in content) and ('queue.Full' not in content)

        if not completed:
            if args.cluster == 'cineca':
                command_line = (f'CUBLAS_WORKSPACE_CONFIG=:4096:8 python run_multimodal.py {to_cmd(hyperparam)} '
                                f'--dataset={args.dataset} '
                                f'--method=feat_prop '
                                f'--model={args.model} > {logs_path}/{args.dataset}/feat_prop/{args.model}/{logfile} 2>&1')
            elif args.cluster == 'margaret':
                command_line = (f'CUBLAS_WORKSPACE_CONFIG=:4096:8 $HOME/.conda/envs/missing/bin/python run_multimodal.py {to_cmd(hyperparam)} '
                                f'--dataset={args.dataset} '
                                f'--method=feat_prop '
                                f'--model={args.model} > {logs_path}/{args.dataset}/feat_prop/{args.model}/{logfile} 2>&1')
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    if args.batch_size_jobs == -1:
        args.batch_size_jobs = nb_jobs

    if args.cluster == 'cineca':
        header = """#!/bin/bash -l
#SBATCH --job-name=SisInf_Missing_Multimod
#SBATCH --time=24:00:00                                   ## format: HH:MM:SS
#SBATCH --nodes=1
#SBATCH --mem=20GB                                       ## memory per node out of 494000MB (481GB)
#SBATCH --output=../../../../../slogs/SisInf_Missing_Multimod_output-%A_%a.out
#SBATCH --error=../../../../../slogs/SisInf_Missing_Multimod_error-%A_%a.err
#SBATCH --account={1}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={2}
#SBATCH --gres=gpu:1                                    ##    1 out of 4 or 8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-{0}

source ~/.bashrc
set -x

module load gcc/12.2.0-cuda-12.1
module load python/3.10.8--gcc--11.3.0

cd $HOME/workspace/Graph-Missing-Modalities

source venv/bin/activate

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

echo "Run experiments"
"""
    elif args.cluster == 'margaret':
        header = """#!/bin/bash -l
#SBATCH --output=../../../../../slogs/missing-%A_%a.out
#SBATCH --error=../../../../../slogs/missing-%A_%a.err
#SBATCH --partition={1}
#SBATCH --job-name=missing
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB # memory in Mb
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH --time=8:00:00 # time requested in days-hours:minutes:seconds
#SBATCH --array=1-{0}

echo "Setting up bash environment"
source ~/.bashrc
set -x

# Modules
module load conda/4.9.2

cd $HOME/projects/Graph-Missing-Modalities/

# Conda environment
conda activate missing

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
"""

    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    if header:
        for index, offset in enumerate(range(0, nb_jobs, args.batch_size_jobs), 1):
            offset_stop = min(offset + args.batch_size_jobs, nb_jobs)
            with open(scripts_path + f'/{args.dataset}/feat_prop/{args.model}/' + date_time + f'__{index}.sh', 'w') as f:
                if args.cluster == 'cineca':
                    print(header.format(offset_stop - offset, args.account, args.mail_user), file=f)
                elif args.cluster == 'margaret':
                    print(header.format(offset_stop - offset, args.partition), file=f)
                current_command_lines = sorted_command_lines[offset: offset_stop]
                for job_id, command_line in enumerate(current_command_lines, 1):
                    print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}', file=f)


if __name__ == '__main__':
    main()
