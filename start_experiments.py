from elliot.run import run_experiment
import argparse
import itertools
import random

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='baby')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--model', type=str, default='freedom')
parser.add_argument('--layers', type=str, default='1')
args = parser.parse_args()

strategies = ['feat_prop']
perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
rounds = [1, 2, 3, 4, 5]

visual = list(itertools.product(strategies, perc, rounds))
textual = list(itertools.product(strategies, perc, rounds))
final = list(zip(visual, textual))

if args.model == 'freedom':
    config = """experiment:
  backend: pytorch
  path_output_rec_result: ./results/{0}/folder/recs/
  path_output_rec_weight: ./results/{0}/folder/weights/
  path_output_rec_performance: ./results/{0}/folder/performance/
  data_config:
    strategy: fixed
    train_path: ../data/{0}/train.tsv
    validation_path: ../data/{0}/val.tsv
    test_path: ../data/{0}/test.tsv
    side_information:
      - dataloader: VisualAttribute
        visual_features: ../data/{0}/image_feat
        masked_items_path: ../data/{0}/visual_sampled_perc_round.txt
        strategy: strategy_name_visual
        feat_prop: co
        prop_layers: propagation_layers
      - dataloader: TextualAttribute
        textual_features: ../data/{0}/text_feat
        masked_items_path: ../data/{0}/textual_sampled_perc_round.txt
        strategy: strategy_name_textual
        feat_prop: co
        prop_layers: propagation_layers
  dataset: dataset_name
  top_k: 50
  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Recall, nDCG, Precision]
  gpu: gpu_id
  external_models_path: ../external/models/__init__.py
  models:
    external.FREEDOM:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: 0.001
      factors: 64
      epochs: 200
      l_w: 1e-5
      n_layers: 1
      n_ui_layers: 2
      top_k: 10
      factors_multimod: 64
      modalities: ('visual', 'textual')
      loaders: ('VisualAttribute','TextualAttribute')
      mw: (0.1,0.9)
      drop: 0.8
      lr_sched: (1.0,50)
      batch_size: 1024
      early_stopping:
        patience: 5
        mode: auto
        monitor: Recall@20
        verbose: True
"""
elif args.model == 'vbpr':
    config = """experiment:
  backend: pytorch
  path_output_rec_result: ./results/{0}/folder/recs/
  path_output_rec_weight: ./results/{0}/folder/weights/
  path_output_rec_performance: ./results/{0}/folder/performance/
  data_config:
    strategy: fixed
    train_path: ../data/{0}/train.tsv
    validation_path: ../data/{0}/val.tsv
    test_path: ../data/{0}/test.tsv
    side_information:
      - dataloader: VisualAttribute
        visual_features: ../data/{0}/image_feat
        masked_items_path: ../data/{0}/visual_sampled_perc_round.txt
        strategy: strategy_name_visual
        feat_prop: co
        prop_layers: propagation_layers
      - dataloader: TextualAttribute
        textual_features: ../data/{0}/text_feat
        masked_items_path: ../data/{0}/textual_sampled_perc_round.txt
        strategy: strategy_name_textual
        feat_prop: co
        prop_layers: propagation_layers
  dataset: dataset_name
  top_k: 50
  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Recall, nDCG, Precision]
  gpu: gpu_id
  external_models_path: ../external/models/__init__.py
  models:
    external.VBPR:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      lr: 0.005
      modalities: ('visual', 'textual')
      epochs: 200
      factors: 64
      batch_size: 1024
      l_w: 1e-2
      comb_mod: concat
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: Recall@20
        verbose: True
    """
else:
    config = """experiment:
  backend: pytorch
  path_output_rec_result: ./results/{0}/folder/recs/
  path_output_rec_weight: ./results/{0}/folder/weights/
  path_output_rec_performance: ./results/{0}/folder/performance/
  data_config:
    strategy: fixed
    train_path: ../data/{0}/train.tsv
    validation_path: ../data/{0}/val.tsv
    test_path: ../data/{0}/test.tsv
    side_information:
      - dataloader: VisualAttribute
        visual_features: ../data/{0}/image_feat
        masked_items_path: ../data/{0}/visual_sampled_perc_round.txt
        strategy: strategy_name_visual
        feat_prop: co
        prop_layers: propagation_layers
      - dataloader: TextualAttribute
        textual_features: ../data/{0}/text_feat
        masked_items_path: ../data/{0}/textual_sampled_perc_round.txt
        strategy: strategy_name_textual
        feat_prop: co
        prop_layers: propagation_layers
  dataset: dataset_name
  top_k: 50
  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Recall, nDCG, Precision]
  gpu: gpu_id
  external_models_path: ../external/models/__init__.py
  models:
    external.LATTICE:
      meta:
        hyper_opt_alg: grid
        verbose: True
        save_weights: False
        save_recs: False
        validation_rate: 10
        validation_metric: Recall@20
        restore: False
      batch_size: 1024
      factors: 64
      lr: 0.001
      l_w: 1e-5
      n_layers: 1
      n_ui_layers: 2
      top_k: 20
      l_m: 0.7
      factors_multimod: 64
      modalities: ('visual', 'textual')
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: Recall@20
        verbose: True
    """

random.seed(42)

for idx, c in enumerate(final):
    visual_strategy = c[0][0]
    visual_perc = c[0][1]
    visual_round = c[0][2]
    textual_strategy = c[1][0]
    textual_perc = c[1][1]
    textual_round = c[1][2]
    folder = f'{visual_strategy}_{visual_perc}_{visual_round}_{textual_strategy}_{textual_perc}_{textual_round}_{args.layers}'
    with open(f'config_files/{args.data}_visual-strategy={visual_strategy}_visual-perc={visual_perc}_'
              f'visual-round={visual_round}_textual-strategy={textual_strategy}_'
              f'textual-perc={textual_perc}_textual-round={textual_round}_layers={args.layers}.yml', 'w') as f:
        f.write(config.replace(
            'folder', folder
        ).replace(
            'strategy_name_visual', visual_strategy
        ).replace(
            'strategy_name_textual', textual_strategy
        ).replace(
            'visual_sampled_perc_round', f'sampled_{visual_perc}_{visual_round}'
        ).replace(
            'textual_sampled_perc_round', f'sampled_{textual_perc}_{textual_round}'
        ).replace('gpu_id', args.gpu).replace('dataset_name', args.data).replace('propagation_layers', args.layers))
    print(f'*********START: {idx + 1}*********')
    run_experiment(f'config_files/{args.data}_visual-strategy={visual_strategy}_visual-perc={visual_perc}_'
                   f'visual-round={visual_round}_textual-strategy={textual_strategy}_'
                   f'textual-perc={textual_perc}_textual-round={textual_round}_layers={args.layers}.yml')
    print(f'*********END: {idx + 1}*********')
