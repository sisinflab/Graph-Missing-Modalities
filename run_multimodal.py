from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='Digital_Music')
parser.add_argument('--model', type=str, default='freedom')
parser.add_argument('--method', type=str, default='zeros')
parser.add_argument('--layers', type=str, default='3')
parser.add_argument('--top_k', type=str, default='20')
args = parser.parse_args()

if args.method == 'feat_prop':
    run_experiment(f"config_files/{args.model}_{args.method}_{args.layers}_{args.top_k}_{args.dataset}.yml")
else:
    run_experiment(f"config_files/{args.model}_{args.method}_{args.dataset}.yml")
