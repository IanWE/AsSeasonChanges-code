import pickle
import os
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
#from termcolor import cprint
from tqdm import tqdm
import time
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
from core import model_utils, data_utils
from core import utils
import json
import numpy as np
from datetime import datetime
from core import temporal
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
from scipy.sparse import vstack
from scipy.sparse import vstack,csr_matrix, hstack
import scipy
import pandas as pd
from core import utils
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher


def transfer_apigraph_feature(drebin_feture, api_cluster_dict):
    apicall_name = drebin_feture.split("::")[1]
    if ";->" in apicall_name:
        method_name = apicall_name.split(";->")[1]
        class_name = apicall_name.split(";->")[0].replace("/", ".")
        drebin_api_name = class_name + "." + method_name
        if api_cluster_dict.get(drebin_api_name) is not None:
            return "apigraph::cluster-{}".format(api_cluster_dict[drebin_api_name])
        else:
            return None
    else:
        drebin_api_name = apicall_name.replace("/", ".") + "."
        for key in api_cluster_dict:
            if key.startswith(drebin_api_name):
                return "apigraph::cluster-{}".format(api_cluster_dict[key])

apigraph_clustering_feature_fn = "datasets/method_cluster_mapping_2000.pkl"
with open(apigraph_clustering_feature_fn, "rb") as f:
    apigraph_clustering_feature = pickle.load(f)

dataset = "2024-GP"
with open(f"./datasets/{dataset}-X.json", 'rb') as f:
    X = json.load(f)

apigraph_features = []
for drebin_feature in tqdm(X):
    apigraph_feature = dict()
    for key in drebin_feature:
        if key.startswith("api_calls"):
            apigraph_feature_value = transfer_apigraph_feature(key, apigraph_clustering_feature)
            if apigraph_feature_value is not None:
                apigraph_feature[apigraph_feature_value] = 1
        else:
            apigraph_feature[key] = 1 
    apigraph_features.append(apigraph_feature)

json.dump(apigraph_features,open("datasets/2024-apigraph-X.json","w"))

