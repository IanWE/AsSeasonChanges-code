import os

import ember
import joblib
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from core import ember_feature_utils, constants
from core import utils
from logger import logger
from multiprocessing import Pool

from tqdm import tqdm
import json
import random

def load_drebin_features():
    """ Return the list of Drebin features.

    :return:
    """
    prefixes = {
        '':'manifest',
        'activities': 'manifest',
        's_and_r': 'manifest',
        'providers': 'manifest',
        'intents':'manifest',
        'app_permissions':'manifest',
        'api_permissions':'code',
        'interesting_calls':'code',#dangerous calls
        'api_calls':'code',
        'urls': 'code'
    }
    return prefixes

# The whole dataset from 2014-2023
def load_gp_dataset(dataset):
    if dataset=="2024-GP" or dataset=="2024-apigraph":
        with open(f"./datasets/{dataset}-X.json", 'rb') as f:
            X = json.load(f)
            vec = DictVectorizer()
            X = vec.fit_transform(X)
            dataset = "2024-GP"
        with open(f"./datasets/{dataset}-Y.json", 'r') as f:
            y_o = json.load(f)
        with open(f"./datasets/{dataset}-meta.json", 'r') as f:
            T = json.load(f)
        if dataset=="2024-GP":
            t_disc = np.array([datetime.strptime(o['discovery'][:10], '%Y-%m-%d') for o in T])
        else:
            T_ = [o['dex_date'] for o in T]
            t = np.array(
                [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o else datetime.strptime(o, '%Y-%m-%d %H:%M:%S')
                    for o in T_])
            T_ = [o['vt_scan_date'] for o in T]
            t_scan = np.array(
                [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') if "T" in o else datetime.strptime(o, '%Y-%m-%d %H:%M:%S')
                    for o in T_])
            T_ = [o['added'] for o in T]
            t_add = np.array(
                [datetime.strptime(o, '%Y-%m-%d %H:%M:%S') if "." not in o else datetime.strptime(o, '%Y-%m-%d %H:%M:%S.%f')
                    for o in T_])
            t_disc = [i if i < j else j
                      for i, j in zip(t_scan, t_add)]
        y = np.asarray(y_o)
        shalist = [o['sha256'] for o in T]
        return X,y,t_disc,(shalist,vec.get_feature_names_out())
    elif 'malscan' in dataset or "combined" in dataset or "downsampled" in dataset:
        return joblib.load(f"datasets/{dataset}_gp.pkl")
    else:
        raise Exception(f"Not found the dataset {dataset}")

#Load only hypercube dataset (2021-2023)
def load_hypercube(feature):
    feature = feature.lower()
    # Load feature file, the originally shared hypercube dataset seems lacking part of features, so we use our re-extracted datasets
    if feature == 'drebin':
        with open('datasets/hypercube/hypercube_drebin.json', 'r') as f:
            all_features = json.load(f)
    elif feature in {'malscan', 'ramda'}:
        feature_path = f'datasets/hypercube/hypercube_{feature}.pickle'
        with open(feature_path, 'rb') as f:
            all_features = pickle.load(f)
    elif feature == 'apigraph':
        X, y, t_disc, filtered_shas = joblib.load(f'datasets/hypercube/hypercube_{feature}.pkl')
        return X, y, t_disc, filtered_shas
    elif feature == 'concatenation':
        X, y, t_disc, filtered_shas = joblib.load(f'datasets/hypercube/concatenation.pkl')
        return X, y, t_disc, filtered_shas
    else:
        raise ValueError(f"Feature {feature} is not included")
    # Load metadata
    with open('datasets/hypercube/hypercube_metadata.json', 'r') as f:
        metadata = json.load(f)
    # Filter to samples that exist in both metadata and features
    filtered_shas = []
    filtered_features = []
    y = []
    time = []
    for entry in metadata:
        sha = entry['sha256']
        if sha in all_features:
            filtered_shas.append(sha)
            filtered_features.append(all_features[sha])
            y.append(1 if entry['vt_detection'] >= 4 else 0)
            time.append(entry['gp_date'])
    # Convert to feature matrix
    if feature == 'drebin':
        X = filtered_features
        from sklearn.feature_extraction import DictVectorizer
        vec = DictVectorizer()
        X = vec.fit_transform(filtered_features)  # Returns a sparse matrix
    elif feature == "malscan":
        X = utils.convert_to_sparse(filtered_features)
    else:
        #X = filtered_features #np.stack([x for x,_,_ in filtered_features])
        X = np.stack(filtered_features)  # Assumes all arrays have the same shape
    return X, y, np.array([datetime.strptime(o, '%Y-%m-%d') for o in time]), filtered_shas

# DATA SETS
def load_ember(dataset='ember'):
    """Load a ember dataset based on the dataset name.
    @param dataset: (str) The name of the dataset to load, default is 'ember'.
    @param selected: (bool) A flag for dataset selection, default is True.
    @param processor: (object) A processor used to process data, default is None.
    @param month: (int) The number of month used for training, only functional for drebin

    return: (numpy.ndarray) x_train, (numpy.ndarray) y_train, (numpy.ndarray) x_test, (numpy.ndarray) y_test
    """
    if dataset == 'ember':
        x_train, y_train, x_test, y_test = load_ember_dataset()
    elif dataset == "ember2018":
        x_train, y_train, x_test, y_test = load_ember_2018()
    elif dataset == "emberall":
        x_train, y_train, x_test, y_test = load_ember_dataset()
        x_train_2018, y_train_2018, x_test_2018, y_test_2018 = load_ember_2018()
        x_train = np.concatenate([x_train,x_test,x_train_2018,x_test_2018],axis=0)
        y_train = np.concatenate([y_train,y_test,y_train_2018,y_test_2018],axis=0)
        emberdf = ember.read_metadata("datasets/ember_2017_2/")
        emberdf2018 = ember.read_metadata("datasets/ember2018/")
        emberdf = pd.concat([emberdf,emberdf2018])
        emberdf = emberdf[emberdf['label']!=-1] #1600000
        #x_test = x_test_2018
        #y_test = y_test_2018
        return x_train, y_train, emberdf
    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    return x_train, y_train, x_test, y_test

class Processor(object):
    def __init__(self,up,lp,valueset_list,rules,c_valueset_list,binarization=False,bundle_rule=[]):
        self.up = up
        self.lp = lp
        self.valueset_list = valueset_list
        self.rules = rules
        self.c_valueset_list = c_valueset_list
        self.binarization = binarization
        self.bundle_rule = bundle_rule

    def process(self,x):
        for i in range(x.shape[1]):
            x_i = x[:,i]
            if self.lp[i]==self.up[i] and self.binarization:
                indices = self.lp[i]==x_i
                x_i[indices] = 0
                x_i[~indices] = 1
            else:
                x_i[x_i<self.lp[i]] = self.lp[i]
                x_i[x_i>self.up[i]] = self.up[i]
        if not self.rules:
            return x
        x_copy = x.copy()
        for i in range(0,x.shape[1]):
            #evenly cut it into 100 sections
            x_i = x[:,i]
            x_clip = x_copy[:,i]
            valueset = self.valueset_list[i]
            for vi in range(len(valueset)-1): #redundant features are eliminated here
                x_i[(x_clip>=valueset[vi])&(x_clip<valueset[vi+1])]=valueset[vi]
            x_i[x_clip<valueset[0]] = valueset[0]
            c_valueset = self.c_valueset_list[i]
            rule = self.rules[i]
            if rule is not None:
                x_i[x_i<c_valueset[0]] = c_valueset[0]
                x_i[x_i>c_valueset[-1]] = c_valueset[-1]
                for v in rule:
                    for r in rule[v]:
                        x_i[x_i==r] = v
                #move outside 
                gap = 1/len(c_valueset)
                x_orig = x_i.copy()
                for idx,j in enumerate(c_valueset):
                    x_i[x_orig==j] = idx*gap
        if not self.bundle_rule:
            return x
        for i in self.bundle_rule:
            #traveling all rules
            for rule in self.bundle_rule[i]:
                indicies = x[:,i] == rule[0]
                x[indicies,rule[1]] = rule[2]
                #if it is a different feature, remove the origin feature
                if i != rule[1]:
                    x[:,i] = 0
        return x

#please run process.py first boxoutdata_2017_100_new.pkl
def load_compressed_dataset(tag, ratio=8, binarization=False):
    if binarization == 'bundle':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_js.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,bundle_rule)
    else:
        print("Wrong compression method!")
        return 
    return x_train, y_train, x_test, y_test, processor

def load_processor(tag,ratio,binarization):
    """Load the processor
    :param tag: (str) tag of the processor
    :param ratio: (float) compression rate
    :param binarization: (bool) whether to use the binarization mechanism
    """
    if binarization == 'bundle':
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,bundle_rule)
    else:
        print("Wrong compression method!")
    return processor


def load_ember_2018():
    """ Return train and test data from EMBER.

    :return: (array, array, array, array)
    """

    # Perform feature vectorization only if necessary.
    try:
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
             constants.EMBER_DATA_DIR_2018,
             feature_version=2
        )
    except:
        ember.create_vectorized_features(
            constants.EMBER_DATA_DIR_2018,
            feature_version=2
        )
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR_2018,
            feature_version=2
        )

    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Get rid of unknown labels
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return x_train, y_train, x_test, y_test
    


# noinspection PyBroadException
def load_ember_dataset():
    """ Return train and test data from EMBER.

    :return: (array, array, array, array)
    """

    # Perform feature vectorization only if necessary.
    try:
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )

    except:
        ember.create_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )

    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Get rid of unknown labels
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return x_train, y_train, x_test, y_test

