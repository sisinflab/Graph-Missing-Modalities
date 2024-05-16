from sklearn.model_selection import ParameterGrid

hyperparams = ParameterGrid({
    "--layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "--top_k": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
})

bash = "#!/bin/bash\n"

for hyp in hyperparams:
    bash += (f"python run_multimodal.py "
             f"--dataset $1 "
             f"--model $2 "
             f"--layers {hyp['--layers']} "
             f"--top_k {hyp['--top_k']} "
             f"--method feat_prop\n")

with open(f"run_multimodal_all_feat_prop.sh", 'w') as f:
    f.write(bash)
