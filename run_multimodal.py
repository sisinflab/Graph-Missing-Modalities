from elliot.run import run_experiment
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='Office_Products')
parser.add_argument('--model', type=str, default='lightgcnm')
parser.add_argument('--method', type=str, default='zeros')
parser.add_argument('--layers', type=str, default='3')
parser.add_argument('--top_k', type=str, default='20')
args = parser.parse_args()

visual_folder_original_indexed = f'./data/{args.dataset}/visual_embeddings_indexed'
textual_folder_original_indexed = f'./data/{args.dataset}/textual_embeddings_indexed'

if args.method in ['pers_page_rank', 'feat_prop']:

    if not os.path.exists('./config_files/'):
        os.makedirs('./config_files/')

    if args.model == 'vbpr':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.layers}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.layers}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.layers}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.VBPR:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [ 1e-5, 1e-2 ]
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          batch_size: 1024
          comb_mod: concat
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
    """
    elif args.model == 'ngcfm':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.layers}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.layers}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.layers}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.NGCFM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          epochs: 200
          n_layers: 3
          factors: 64
          weight_size: 64
          node_dropout: 0.1
          message_dropout: 0.1
          batch_size: 1024
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          normalize: True
          l_w: [ 1e-5, 1e-2 ]
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == 'lightgcnm':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.layers}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.layers}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.layers}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.LightGCNM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          epochs: 200
          n_layers: 3
          factors: 64
          normalize: True
          l_w: [ 1e-5, 1e-2 ]
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          batch_size: 1024
          aggregation: concat
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == "freedom":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.layers}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.layers}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.layers}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.FREEDOM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [1e-5, 1e-2]
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
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == "bm3":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.layers}_{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.layers}_{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.layers}_{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_{args.method}_{args.layers}_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.BM3:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          multimod_factors: 64
          reg_weight: [0.1, 0.01]
          cl_weight: 2.0
          dropout: 0.3
          n_layers: 2
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          epochs: 200
          factors: 64
          lr_sched: (1.0,50)
          batch_size: 1024
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True        
            """
    else:
        raise NotImplemented

    with open(f'./config_files/{args.model}_feat_prop_{args.layers}_{args.top_k}_{args.dataset}.yml', 'w') as f:
        f.write(config.format(args.dataset).replace('dataset_name', args.dataset))

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

elif args.method == 'neigh_mean':

    if not os.path.exists('./config_files/'):
        os.makedirs('./config_files/')

    if args.model == 'vbpr':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.VBPR:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [ 1e-5, 1e-2 ]
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          batch_size: 1024
          comb_mod: concat
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
    """
    elif args.model == 'ngcfm':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.NGCFM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          epochs: 200
          n_layers: 3
          factors: 64
          weight_size: 64
          node_dropout: 0.1
          message_dropout: 0.1
          batch_size: 1024
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          normalize: True
          l_w: [ 1e-5, 1e-2 ]
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == 'lightgcnm':
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.LightGCNM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr:  [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          epochs: 200
          n_layers: 3
          factors: 64
          normalize: True
          l_w: [ 1e-5, 1e-2 ]
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          batch_size: 1024
          aggregation: concat
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == "freedom":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.FREEDOM:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          factors: 64
          epochs: 200
          l_w: [1e-5, 1e-2]
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
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True
            """
    elif args.model == "bm3":
        config = f"""experiment:
      backend: pytorch
      path_output_rec_result: ./results/{0}/{args.top_k}/folder/recs/
      path_output_rec_weight: ./results/{0}/{args.top_k}/folder/weights/
      path_output_rec_performance: ./results/{0}/{args.top_k}/folder/performance/
      data_config:
        strategy: fixed
        train_path: ../data/{0}/train_indexed.tsv
        validation_path: ../data/{0}/val_indexed.tsv
        test_path: ../data/{0}/test_indexed.tsv
        side_information:
          - dataloader: VisualAttribute
            visual_features: ../data/{0}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed
          - dataloader: TextualAttribute
            textual_features: ../data/{0}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed
      dataset: dataset_name
      top_k: 50
      evaluation:
        cutoffs: [10, 20, 50]
        simple_metrics: [Recall, nDCG, Precision]
      gpu: 0
      external_models_path: ../external/models/__init__.py
      models:
        external.BM3:
          meta:
            hyper_opt_alg: grid
            verbose: True
            save_weights: False
            save_recs: False
            validation_rate: 1
            validation_metric: Recall@20
            restore: False
          lr: [ 0.0001, 0.0005, 0.001, 0.005, 0.01 ]
          multimod_factors: 64
          reg_weight: [0.1, 0.01]
          cl_weight: 2.0
          dropout: 0.3
          n_layers: 2
          modalities: ('visual', 'textual')
          loaders: ('VisualAttribute','TextualAttribute')
          epochs: 200
          factors: 64
          lr_sched: (1.0,50)
          batch_size: 1024
          seed: 123
          early_stopping:
            patience: 5
            mode: auto
            monitor: Recall@20
            verbose: True        
            """
    else:
        raise NotImplemented

    with open(f'./config_files/{args.model}_neigh_mean_{args.top_k}_{args.dataset}.yml', 'w') as f:
        f.write(config.format(args.dataset).replace('dataset_name', args.dataset))

    visual_folder_imputed_indexed = f'./data/{args.dataset}/visual_embeddings_neigh_mean_{args.top_k}_indexed'
    textual_folder_imputed_indexed = f'./data/{args.dataset}/textual_embeddings_neigh_mean_{args.top_k}_indexed'
    visual_folder_complete = f'./data/{args.dataset}/visual_embeddings_neigh_mean_{args.top_k}_complete_indexed'
    textual_folder_complete = f'./data/{args.dataset}/textual_embeddings_neigh_mean_{args.top_k}_complete_indexed'

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

    run_experiment(f"config_files/{args.model}_neigh_mean_{args.top_k}_{args.dataset}.yml")

    shutil.rmtree(visual_folder_complete)
    shutil.rmtree(textual_folder_complete)

    os.remove(f"config_files/{args.model}_neigh_mean_{args.top_k}_{args.dataset}.yml")
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
