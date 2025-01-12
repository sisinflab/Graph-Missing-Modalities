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
    bash += (f"CUBLAS_WORKSPACE_CONFIG=:16:8 python impute_autoencoder.py "
             f"--data {args.data} "
             f"--top_k {hyp['--top_k']} "
             f"--method gae\n")

with open(f"impute_all_gae_{args.data}.sh", 'w') as f:
    f.write(bash)
