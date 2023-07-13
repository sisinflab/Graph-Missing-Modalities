from elliot.run import run_experiment
import argparse
import itertools
import random

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='office')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--colab', type=bool, default=False)
args = parser.parse_args()

strategies = ['zeros', 'mean', 'random', 'feat_prop']
perc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
rounds = [1, 2, 3, 4, 5]

visual = list(itertools.product(strategies, perc))
textual = list(itertools.product(strategies, perc))
final = list(zip(visual, textual))

if args.colab:
    config = """experiment:
      backend: pytorch
      path_output_rec_result: ../Graph-Missing-Modalities/results/{0}/folder/recs/
      path_output_rec_weight: ../Graph-Missing-Modalities/results/{0}/folder/weights/
      path_output_rec_performance: ../Graph-Missing-Modalities/results/{0}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../../Graph-Missing-Modalities/data/{0}/train.txt
        validation_path: ../../Graph-Missing-Modalities/data/{0}/val.txt
        test_path: ../../Graph-Missing-Modalities/data/{0}/test.txt
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../../Graph-Missing-Modalities/data/{0}/image_feat
            masked_items_path: ../../Graph-Missing-Modalities/data/{0}/visual_sampled_perc_round.txt
            strategy: strategy_name_visual
            feat_prop: co
            prop_layers: 3
          - dataloader: TextualAttribute
            textual_features: ../../Graph-Missing-Modalities/data/{0}/text_feat
            masked_items_path: ../../Graph-Missing-Modalities/data/{0}/textual_sampled_perc_round.txt
            strategy: strategy_name_textual
            feat_prop: co
            prop_layers: 3
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
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          modalities: ('visual', 'textual')
          epochs: 200
          factors: 64
          batch_size: 1024
          l_w: [ 1e-5, 1e-2 ]
          comb_mod: concat
          seed: 123
        external.MMGCN:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 10
            validation_metric: Recall@20
            restore: False
          lr: [ 0.00001, 0.00003, 0.0001, 0.001, 0.01 ]
          epochs: 200
          num_layers: 3
          factors: 64
          factors_multimod: (256, None)
          batch_size: 1024
          aggregation: mean
          concatenation: False
          has_id: True
          modalities: ('visual', 'textual')
          l_w: [ 1e-5, 1e-2 ]
          seed: 123
        external.GRCN:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 10
            validation_metric: Recall@20
            restore: False
          lr: [0.0001, 0.001, 0.01, 0.1, 1]
          epochs: 200
          num_layers: 2
          num_routings: 3
          factors: 64
          factors_multimod: 128
          batch_size: 1024
          aggregation: add
          weight_mode: confid
          pruning: True
          has_act: False
          fusion_mode: concat
          modalities: ('visual', 'textual')
          l_w: [1e-5, 1e-2]
          seed: 123
        external.LATTICE:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 10
            validation_metric: Recall@20
            restore: False
          epochs: 200
          batch_size: 1024
          factors: 64
          lr: [0.0001, 0.0005, 0.001, 0.005, 0.01]
          l_w: [1e-5, 1e-2]
          n_layers: 1
          n_ui_layers: 2
          top_k: 20
          l_m: 0.7
          factors_multimod: 64
          modalities: ('visual', 'textual')
          seed: 123
    """
else:
    config = """experiment:
          backend: pytorch
          path_output_rec_result: ./results/{0}/folder/recs/
          path_output_rec_weight: ./results/{0}/folder/weights/
          path_output_rec_performance: ./results/{0}/folder/performance/
          data_config:
            strategy: fixed
            train_path: ../data/{0}/train.txt
            validation_path: ../data/{0}/val.txt
            test_path: ../data/{0}/test.txt
            side_information:
              - dataloader: VisualAttribute
                visual_features: ../data/{0}/image_feat
                masked_items_path: ../data/{0}/visual_sampled_perc_round.txt
                strategy: strategy_name_visual
                feat_prop: co
                prop_layers: 3
              - dataloader: TextualAttribute
                textual_features: ../data/{0}/text_feat
                masked_items_path: ../data/{0}/textual_sampled_perc_round.txt
                strategy: strategy_name_textual
                feat_prop: co
                prop_layers: 3
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
              lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
              modalities: ('visual', 'textual')
              epochs: 200
              factors: 64
              batch_size: 1024
              l_w: [ 1e-5, 1e-2 ]
              comb_mod: concat
              seed: 123
            external.MMGCN:
              meta:
                hyper_opt_alg: grid
                verbose: True
                save_weights: False
                save_recs: False
                validation_rate: 10
                validation_metric: Recall@20
                restore: False
              lr: [ 0.00001, 0.00003, 0.0001, 0.001, 0.01 ]
              epochs: 200
              num_layers: 3
              factors: 64
              factors_multimod: (256, None)
              batch_size: 1024
              aggregation: mean
              concatenation: False
              has_id: True
              modalities: ('visual', 'textual')
              l_w: [ 1e-5, 1e-2 ]
              seed: 123
            external.GRCN:
              meta:
                hyper_opt_alg: grid
                verbose: True
                save_weights: False
                save_recs: False
                validation_rate: 10
                validation_metric: Recall@20
                restore: False
              lr: [0.0001, 0.001, 0.01, 0.1, 1]
              epochs: 200
              num_layers: 2
              num_routings: 3
              factors: 64
              factors_multimod: 128
              batch_size: 1024
              aggregation: add
              weight_mode: confid
              pruning: True
              has_act: False
              fusion_mode: concat
              modalities: ('visual', 'textual')
              l_w: [1e-5, 1e-2]
              seed: 123
            external.LATTICE:
              meta:
                hyper_opt_alg: grid
                verbose: True
                save_weights: False
                save_recs: False
                validation_rate: 10
                validation_metric: Recall@20
                restore: False
              epochs: 200
              batch_size: 1024
              factors: 64
              lr: [0.0001, 0.0005, 0.001, 0.005, 0.01]
              l_w: [1e-5, 1e-2]
              n_layers: 1
              n_ui_layers: 2
              top_k: 20
              l_m: 0.7
              factors_multimod: 64
              modalities: ('visual', 'textual')
              seed: 123
        """

random.seed(42)

for idx, c in enumerate(final):
    visual_strategy = c[0][0]
    visual_perc = c[0][1]
    visual_round = random.sample(rounds, 1)[0]
    textual_strategy = c[1][0]
    textual_perc = c[1][1]
    textual_round = random.sample(rounds, 1)[0]
    folder = f'{visual_strategy}_{visual_perc}_{visual_round}_{textual_strategy}_{textual_perc}_{textual_round}'
    with open(f'config_files/{args.data}_visual-strategy={visual_strategy}_visual-perc={visual_perc}_'
              f'visual-round={visual_round}_textual-strategy={textual_strategy}_'
              f'textual-perc={textual_perc}_textual-round={textual_round}.yml', 'w') as f:
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
        ).replace('gpu_id', args.gpu).replace('dataset_name', args.data))
    print(f'*********START: {idx + 1}*********')
    run_experiment(f'config_files/visual-strategy={visual_strategy}_visual-perc={visual_perc}_'
                   f'visual-round={visual_round}_textual-strategy={textual_strategy}_'
                   f'textual-perc={textual_perc}_textual-round={textual_round}.yml')
    print(f'*********END: {idx + 1}*********')
