import pandas as pd
import random

data = 'baby'
seed = 42
num_repeats = 5
num_items = {
    'baby': 7050,
    'toys': 11924,
    'sports': 18357
}

num_items = num_items[data]

for n in range(num_repeats):
    random.seed(seed + n)
    for p in range(10, 100, 10):
        sampled = random.sample(list(range(num_items)), int((p / 100) * num_items))
        pd.DataFrame([pd.Series(sampled)]).transpose().sort_values(by=0).to_csv(
            f'./MMSSL/data/{data}/sampled_{p}_{n + 1}.txt', sep='\t', header=None, index=None)