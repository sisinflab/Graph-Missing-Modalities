from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')
args = parser.parse_args()

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"python impute.py "
             f"--data {args.dataset} "
             f"--gpu {args.gpu_id} "
             f"--layers {hyp['--layers']} "
             f"--top_k {hyp['--top_k']} "
             f"--method feat_prop\n")

with open(f"impute_all_feat_prop_{args.dataset}.sh", 'w') as f:
    f.write(bash)
