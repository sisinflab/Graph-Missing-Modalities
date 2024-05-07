from elliot.run import run_experiment
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='Digital_Music')
parser.add_argument('--model', type=str, default='freedom')
parser.add_argument('--method', type=str, default='zeros')
parser.add_argument('--layers', type=str, default='3')
parser.add_argument('--top_k', type=str, default='20')
args = parser.parse_args()

visual_folder_original_indexed = f'./data/{args.dataset}/visual_embeddings_indexed'
textual_folder_original_indexed = f'./data/{args.dataset}/textual_embeddings_indexed'

if args.method == 'feat_prop':
    visual_folder_imputed_indexed = f'./data/{args.dataset}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    textual_folder_imputed_indexed = f'./data/{args.dataset}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    visual_folder_complete = f'./data/{args.dataset}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed'
    textual_folder_complete = f'./data/{args.dataset}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed'

    if not os.path.exists(visual_folder_complete):
        os.makedirs(visual_folder_complete)

    if not os.path.exists(textual_folder_complete):
        os.makedirs(textual_folder_complete)

    for it in os.listdir(visual_folder_original_indexed):
        shutil.copy(os.path.join(visual_folder_original_indexed, it), visual_folder_complete)
    for it in os.listdir(visual_folder_imputed_indexed):
        shutil.copy(os.path.join(visual_folder_imputed_indexed, it), visual_folder_complete)

    for it in os.listdir(textual_folder_original_indexed):
        shutil.copy(os.path.join(textual_folder_original_indexed, it), textual_folder_complete)
    for it in os.listdir(textual_folder_imputed_indexed):
        shutil.copy(os.path.join(textual_folder_imputed_indexed, it), textual_folder_complete)

    run_experiment(f"config_files/{args.model}_{args.method}_{args.layers}_{args.top_k}_{args.dataset}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)
    os.remove(f"config_files/{args.model}_{args.method}_{args.layers}_{args.top_k}_{args.dataset}.yml")
else:
    visual_folder_imputed_indexed = f'./data/{args.dataset}/visual_embeddings_{args.method}_indexed'
    textual_folder_imputed_indexed = f'./data/{args.dataset}/textual_embeddings_{args.method}_indexed'
    visual_folder_complete = f'./data/{args.dataset}/visual_embeddings_{args.method}_complete_indexed'
    textual_folder_complete = f'./data/{args.dataset}/textual_embeddings_{args.method}_complete_indexed'

    if not os.path.exists(visual_folder_complete):
        os.makedirs(visual_folder_complete)

    if not os.path.exists(textual_folder_complete):
        os.makedirs(textual_folder_complete)

    for it in os.listdir(visual_folder_original_indexed):
        shutil.copy(os.path.join(visual_folder_original_indexed, it), visual_folder_complete)
    for it in os.listdir(visual_folder_imputed_indexed):
        shutil.copy(os.path.join(visual_folder_imputed_indexed, it), visual_folder_complete)

    for it in os.listdir(textual_folder_original_indexed):
        shutil.copy(os.path.join(textual_folder_original_indexed, it), textual_folder_complete)
    for it in os.listdir(textual_folder_imputed_indexed):
        shutil.copy(os.path.join(textual_folder_imputed_indexed, it), textual_folder_complete)

    run_experiment(f"config_files/{args.model}_{args.method}_{args.dataset}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)

    os.remove(f"config_files/{args.model}_{args.method}_{args.dataset}.yml")
