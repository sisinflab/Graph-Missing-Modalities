# Graph-Missing-Modalities

To begin with, download the original datasets from the following URLs:

* [MMSSL](https://github.com/HKUDS/MMSSL)
  * Amazon Baby: https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0
  * Amazon Toys: https://file.io/eXpmDrS11JZt
  * Amazon Sports: https://drive.google.com/drive/folders/1AB1RsnU-ETmubJgWLpJrXd8TjaK_eTp0
* FREEDOM:
  * Amazon Baby: https://file.io/OCDmI7Jcbxst
  * Amazon Toys: https://file.io/8HYBcl8eTRiM
  * Amazon Sports: https://file.io/3GiTMivhIEzT

and place them to:
```
# MMSSL

├── MMSSL/
│   ├── data
│       ├── baby/
│       ├── toys/
│       ├── sports/

# FREEDOM

├── data
│   ├── baby/
│   ├── toys/
│   ├── sports/
``` 

Second, we install the useful requirements:
```sh
pip install -r requirements.txt && \
pip install -r requirements_dgl.txt && \
pip install -r requirements_torch_geometric.txt
``` 

Then, we generate the multimodal recommendation datasets with missing modalities:
```sh
python3 mask_items_MMSSL.py --data {dataset} && \
python3 mask_items_FREEDOM.py --data {dataset}
``` 

The output files will be stored at the following paths:
```
# MMSSL

├── MMSSL/
│   ├── data/
│       ├── baby/
│           ├── sampled_10_1.txt
│           ├── sampled_10_2.txt
│           ├── ...
│       ├── ...

# FREEDOM

├── data/
│   ├── baby/
│       ├── sampled_10_1.txt
│       ├── sampled_10_2.txt
│       ├── ...
│   ├── ...
```

After this necessary pre-processing step, we can run the training/test of the recommendation models with all the baselines for missing modalities:
```sh
# MMSSL (this covers all possible settings)
PYTHONPATH=. ./MMSSL/run_all.sh {dataset} 
```
```sh
# FREEDOM (this should be run for all propagation layers)
python3 start_experiments.py --data {dataset} --gpu {gpu} --layers {layers}
``` 
