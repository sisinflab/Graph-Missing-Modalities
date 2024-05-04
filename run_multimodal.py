from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='Digital_Music')
parser.add_argument('--model', type=str, default='freedom')
parser.add_argument('--method', type=str, default='zeros')
args = parser.parse_args()

run_experiment(f"config_files/{args.model}_{args.method}_{args.dataset}.yml")
