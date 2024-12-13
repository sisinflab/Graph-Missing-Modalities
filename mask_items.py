import pandas as pd
import random

import argparse

parser = argparse.ArgumentParser(description="Run mask items.")
parser.add_argument('--data', type=str, default='Office_Products')
args = parser.parse_args()

data = args.data

seed = 42
num_repeats = 5

train = pd.read_csv(f'./data/{data}/train_indexed.tsv', sep='\t', header=None)
items = list(range(train[1].nunique()))

for n in range(num_repeats):
    random.seed(seed + n)
    for p in range(10, 100, 10):
        sampled = random.sample(items, int((p / 100) * len(items)))
        pd.DataFrame([pd.Series(sampled)]).transpose().sort_values(by=0).to_csv(
            f'./data/{data}/sampled_{p}_{n + 1}.txt', sep='\t', header=None, index=None)
