import argparse
import os
import pandas as pd
import numpy as np
import shutil

np.random.seed(42)

parser = argparse.ArgumentParser(description="Run imputation.")
parser.add_argument('--data', type=str, default='Digital_Music')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=str, default='1')
parser.add_argument('--method', type=str, default='zeros')
args = parser.parse_args()

visual_folder = f'data/{args.data}/visual_embeddings/torch/ResNet50/avgpool'
textual_folder = f'data/{args.data}/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

output_visual = f'data/{args.data}/visual_embeddings_{args.method}'
output_textual = f'data/{args.data}/textual_embeddings_{args.method}'

try:
    missing_visual = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_visual.tsv'), sep='\t', header=None)
    missing_visual = set(missing_visual[0].tolist())
except pd.errors.EmptyDataError:
    missing_visual = set()

try:
    missing_textual = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_textual.tsv'), sep='\t', header=None)
    missing_textual = set(missing_textual[0].tolist())
except pd.errors.EmptyDataError:
    missing_textual = set()

visual_shape = np.load(os.path.join(visual_folder, os.listdir(visual_folder)[0])).shape
textual_shape = np.load(os.path.join(textual_folder, os.listdir(textual_folder)[0])).shape

if not os.path.exists(output_visual):
    os.makedirs(output_visual)
if not os.path.exists(output_textual):
    os.makedirs(output_textual)

if args.method == 'zeros':
    for miss in missing_visual:
        np.save(os.path.join(output_visual, f'{miss}.npy'), np.zeros(visual_shape, dtype=np.float32))

    for miss in missing_textual:
        np.save(os.path.join(output_textual, f'{miss}.npy'), np.zeros(textual_shape, dtype=np.float32))

elif args.method == 'random':
    for miss in missing_visual:
        np.save(os.path.join(output_visual, f'{args.method}/{miss}.npy'), np.random.rand(*visual_shape))

    for miss in missing_textual:
        np.save(os.path.join(output_textual, f'{args.method}/{miss}.npy'), np.random.rand(*textual_shape))

elif args.method == 'mean':
    num_items_visual = len(os.listdir(visual_folder))
    num_items_textual = len(os.listdir(textual_folder))

    visual_features = np.empty((num_items_visual, visual_shape[-1])) if num_items_visual else None
    textual_features = np.empty((num_items_textual, textual_shape[-1])) if num_items_textual else None

    if visual_features is not None:
        visual_items = os.listdir(visual_folder)
        for idx, it in enumerate(visual_items):
            visual_features[idx, :] = np.load(os.path.join(visual_folder, it))
        mean_visual = visual_features.mean(axis=0, keepdims=True)
        for miss in missing_visual:
            np.save(os.path.join(output_visual, f'{miss}.npy'), mean_visual)

    if textual_features is not None:
        textual_items = os.listdir(textual_folder)
        for idx, it in enumerate(textual_items):
            textual_features[idx, :] = np.load(os.path.join(textual_folder, it))
        mean_textual = textual_features.mean(axis=0, keepdims=True)
        for miss in missing_textual:
            np.save(os.path.join(output_textual, f'{miss}.npy'), mean_textual)

visual_items = os.listdir(visual_folder)
textual_items = os.listdir(textual_folder)

for it in visual_items:
    shutil.copy(os.path.join(visual_folder, it), os.path.join(output_visual, args.method))

for it in textual_items:
    shutil.copy(os.path.join(textual_folder, it), os.path.join(output_textual, args.method))
