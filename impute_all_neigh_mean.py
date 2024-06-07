from sklearn.model_selection import ParameterGrid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose gpu id')
args = parser.parse_args()

hyperparams = ParameterGrid({
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"python impute.py "
             f"--data {args.data} "
             f"--gpu {args.gpu_id} "
             f"--top_k {hyp['--top_k']} "
             f"--method neigh_mean\n")

with open(f"impute_all_neigh_mean_{args.data}.sh", 'w') as f:
    f.write(bash)
