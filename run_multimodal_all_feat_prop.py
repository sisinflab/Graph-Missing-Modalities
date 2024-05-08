from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--model', type=str, default='vbpr', help='choose model')
args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"python run_multimodal.py "
             f"--dataset {args.dataset} "
             f"--model {args.model} "
             f"--layers {hyp['--layers']} "
             f"--top_k {hyp['--top_k']} "
             f"--method feat_prop\n")

with open(f"run_multimodal_all_feat_prop_{args.dataset}.sh", 'w') as f:
    f.write(bash)
