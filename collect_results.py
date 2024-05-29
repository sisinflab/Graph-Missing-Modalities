import ast
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Run collect results.")
parser.add_argument('--dataset', type=str, default='Office_Products')
parser.add_argument('--num_layers', type=int, default=20)
parser.add_argument('--num_top_k', type=int, default=10)
parser.add_argument('--metric', type=str, default='nDCG')
parser.add_argument('--model', type=str, default='ngcfm')
args = parser.parse_args()

file_path = f"feat_prop_{args.dataset}_{args.model}_collected.out"
best_model_results = np.empty((args.num_layers, args.num_top_k))

with open(file_path, 'r') as file:
    for _, line in enumerate(file):
        dictionary = ast.literal_eval(line.split('Best Model results:\t')[-1].split('\n')[0])
        layer = int(line.split('layers=')[-1].split('_')[0])
        top_k = int(line.split('top_k=')[-1].split('.log')[0])
        best_model_results[layer - 1, top_k // 10 - 1] = dictionary[20][args.metric]

for l in range(args.num_layers):
    print(f'Best {args.metric} for layer {l + 1}: {best_model_results[l, :].max()}')

for t in range(args.num_top_k):
    print(f'Best {args.metric} for top_k {(t + 1) * 10}: {best_model_results[:, t].max()}')

print(f'Best {args.metric} overall: {best_model_results.max()}')

np.save(f'feat_prop_{args.dataset}_{args.model}_{args.metric}.npy', best_model_results)
