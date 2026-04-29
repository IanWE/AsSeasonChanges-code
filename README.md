# AsSeasonChanges-code
This is the implementation for paper **As Seasons Change: Temporal Feature Powers Malware Classification**.

## Directly-related Code
If you only care about the core algorithm of this paper, see `core/moe/moe.py` and `core/nn.py` for implementation of G-MoE.py

## Environments
The code repository is build on a Ubuntu 24.04 with python 3.13.5.  
All required packages are listed in `requirements.txt`; you can use `pip install -r requirements` to install them. For ember, you can install use `pip install git+https://github.com/elastic/ember.git`. After that, use `bash init.sh` to create the neccesary directories.

## Datasets
Before starting the reproduction, please download the necessary datasets: [ANDROID](https://androzoo.uni.lu/),[EMBER](https://github.com/elastic/ember/), [Anoshif](https://github.com/bit-ml/AnoShift/tree/main) and [WildTime](https://github.com/huaxiuyao/Wild-Time). The meta information for the whole dataset `dataset/2024-GP1-meta.json` and hypercube dataset `dataset/hypercube_metadata.json` are shared while `dataset/proposed.json` include the family information prossed by euphony. You can directly use them. Note that when extracting DREBIN features, make sure the [baksmali](https://github.com/baksmali/smali/releases) is the newest version (we are using baksmali-2.5.2), otherwise, your extracted feature will loss large portion of features. We also shared the feature extraction code that we used in `feature-extraction/`

Once you downloaded the dataset and extracted features, refer to `process_feature.ipynb` to process and save them, then you can use use `data_utils.load_gp_data(NAME)` or `data_utils.load_hypercube(NAME)` to load them. 

### Quick start
You can also use datasets provided by other existing works (e.g., [Transcendent](https://github.com/s2labres/transcendent-release)) for evaluation. 

## Description of files
There are four scripts you can directly use them to replicate the experimental results.

`verify_negative_effect.py`: This is the script for verifying negative of old samples, it first train the model on the whole set and then gradually removing old samples year by year to verify the distribution conflict effect, you can select different datasets, models and feature types under the main function.

`next_year_prediction.py`: This is the script for the experiment of next year prediction, it first train the model on all sample before the testing year and then use the model predict the testing year.

`sustainability_verification.py`: This is the script for verifying the sustainability of different models, it will train the model on 2014 and use the model to predict samples in the next nine years. You can also modify it for different datasets, models and feature types.

`active_learning.py`: This is for verifying the effect on active learning. You can set up for different features/datasets in this file.

`process_apigraph.py`: Code for processing DREBIN into APIGraph

`process_data_bundle.py` and `process_data_bundle_original.py`: These are the files to process dataset using subspace compression with binarization (SCB).`process_data_bundle_original.py` allows you to process train and test set separately.

`core/data_utils.py`: Code for loading datasets

`core/model_utils.py`: Code for train/save/load different models

`code/nn.py`: Model settings for training NN/G-MoE/T-MoE/MoE

`code/moe/*.py`: Code for G-MoE (`moe.py`), T-MoE (`moe_wg.py`) and MoE(`moe_o.py`)

`utils.py`: Other helper functions

### Testing on other datasets
Before testing, please install the environments and download datasets for them first.
`ember_verification.py`: Code for evaluation on EMBER dataset

`Anoshift/test_on_anoshift.py`: Code for testing G-MoE on AnoShift dataset

`Anoshift/kyoto.py`: G-MoE combined with deepSVDD.py

`wildtime/wildtime.ipynb`: Testing G-MoE on wildtime datasets








