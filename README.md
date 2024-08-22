# Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation? 

![GitHub Repo stars](https://img.shields.io/github/stars/sisinflab/Graph-Missing-Modalities)
 [![arXiv](https://img.shields.io/badge/arXiv-2408.11767-b31b1b.svg)](https://arxiv.org/abs/2408.11767)
 
This is the official implementation of the paper "_Do We Really Need to Drop Items with Missing Modalities in
Multimodal Recommendation?_", accepted at CIKM 2024 as a short paper.

## Requirements

Install the useful packages:

```sh
pip install -r requirements.txt
pip install -r requirements_torch_geometric.txt
```

## Datasets

### Datasets download

Download the Office, Music, and Beauty datasets from the original repository:

- Office:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz
- Music:
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Digital_Music.json.gz
- Beauty: 
  - Reviews: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
  - Metadata: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz

And place each of them in the corresponding dataset folder accessible at ./data/<dataset-name>/. Then, run the following script:

```sh
python prepare_datasets.py --data <dataset_name>
```

this will create files for the items metadata and user-item reviews, and save the product images (check in the corresponding dataset folder). Moreover, statistics about the considered datasets will be displayed (e.g., missing modalities).


### Multimodal features extraction

After that, we need to extract the visual and textual features from items metadata and images. To do so, we use the framework [Ducho](https://github.com/sisinflab/Ducho), running the following configuration file for each dataset:

```yaml
dataset_path: ./data/<dataset_name>
gpu list: 0

visual:
    items:
        input_path: images
        output_path: visual_embeddings
        model: [
               { model_name: ResNet50,  output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch},
        ]

textual:
    items:
        input_path: final_meta.tsv
        item_column: asin
        text_column: description
        output_path: textual_embeddings
        model: [
            { model_name: sentence-transformers/all-mpnet-base-v2,  output_layers: 1, clear_text: False, backend: sentence_transformers},
          ]
```

where <dataset_name> should be substituted accordingly. This will extract visual and textual features for all datasets, accessible under each dataset folder.

### Missing features imputations

First, we perform imputation through traditional machine learning methods (zeros, random, mean). To do so, run the following script:

```sh
python run_split.py --data <dataset_name>
python impute.py --data <dataset_name> --gpu <gpu_id> --method <zeros_random_mean>
```

this will create, under the specific dataset folder, an additional folder with **only** the imputed features, both visual and textual.

Before running the imputation through the graph-aware methods (neigh_mean, feat_prop, pers_page_rank), we need to split intro train/val/test and map all data processed so far to numeric ids. To do so, run the following script:

```sh

python to_id.py --data <dataset_name> --method <zeros_random_mean>
```

this will create, for each dataset/modality/imputation folder, a new folder with the mapped (indexed) data. 

Now we can run the imputation with graph-aware methods. 

#### NeighMean

```sh
python impute_all_neigh_mean.py --data <dataset_name> --gpu <gpu_id>
chmod +777 impute_all_neigh_mean_<dataset_name>.sh
./impute_all_neigh_mean_<dataset_name>.sh
```

#### MultiHop

```sh
python impute_all_feat_prop_pers_page_rank.py --data <dataset_name> --gpu <gpu_id> --method feat_prop
chmod +777 impute_all_feat_prop_<dataset_name>.sh
./impute_all_feat_prop_<dataset_name>.sh
```

#### PersPageRank

```sh
python impute_all_feat_prop_pers_page_rank.py --data <dataset_name> --gpu <gpu_id> --method pers_page_rank
chmod +777 impute_all_pers_page_rank_<dataset_name>.sh
./impute_all_pers_page_rank_<dataset_name>.sh
```

Now we are all set to run the experiments. We use [Elliot](https://github.com/sisinflab/Formal-MultiMod-Rec) to train/evaluate the multimodal recommender systems.

### Results

#### Dropped setting

To obtain the results in the **dropped** setting, run the following scripts:
```sh
python run_split.py --data <dataset_name> --dropped yes
python data/<dataset_name>/to_id_final.py
python run_dropped.py --data <dataset_name>
```

#### Imputed setting

Then, we can compute the performance for the **imputed** setting. In the case of traditional machine learning imputation, we have:

```sh
chmod +777 ./do_all_zeros_random_mean.sh
./do_all_zeros_random_mean.sh <dataset_name> <zeros_random_mean>
python run_lightgcn_sgl.py --data <dataset_name>
```

For the graph-aware imputation methods, we run:

```sh
# For NeighMean
python neigh_mean.py --data <dataset_name> --gpu <gpu_id> --model <multimodal_recommender>
chmod +777 run_multimodal_all_neigh_mean_<dataset_name>_<multimodal_recommender>.sh
./run_multimodal_all_neigh_mean_<dataset_name>_<multimodal_recommender>.sh
```

```sh
# For MultiHop
python feat_prop.py --data <dataset_name> --gpu <gpu_id> --model <multimodal_recommender>
chmod +777 run_multimodal_all_feat_prop_<dataset_name>_<multimodal_recommender>.sh
./run_multimodal_all_feat_prop_<dataset_name>_<multimodal_recommender>.sh
```

```sh
# For PersPageRank
python pers_page_rank.py --data <dataset_name> --gpu <gpu_id> --model <multimodal_recommender>
chmod +777 run_multimodal_all_pers_page_rank_<dataset_name>_<multimodal_recommender>.sh
./run_multimodal_all_pers_page_rank_<dataset_name>_<multimodal_recommender>.sh
```

#### Collect results

To collect all results for MultiHop and PersPageRank (the settings with the highest number of configurations), run the following:
```sh
chmod +777 collect_results.sh
./collect_results.sh <dataset_name> <method> <model>
python collect_results.py --data <dataset_name> --model <multimodal_recommender> --method <method> --metric <metric_name>
```
