import pickle
import logging
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
from sklearn.feature_selection import VarianceThreshold

#Load dataset
dataset = "drebin"# apigraph
feat_type = "DENSE"# for malscanscb, just set as FULL
metric = "uncertanty"

X, y, t_disc, filtered_shas = data_utils.load_hypercube(dataset)#select the dataset here
t_disc = np.array(t_disc)

t_appear = np.array(t_disc)[np.where((np.array(t_disc)>=datetime(2021, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(2022, 1, 1, 0, 0, 0)))[0]]
t_appear = [i.timestamp() for i in t_appear]
t_all = [i.timestamp() for i in t_disc]
scaler = MinMaxScaler()
scaler.fit(np.array(t_appear).reshape(-1,1))
t_normalized = scaler.transform(np.array(t_all).reshape(-1,1))

if scipy.sparse.issparse(X):
    X = hstack([X,t_normalized]).tocsr()
else:
    X = np.hstack([X,t_normalized])
idx = np.where((np.array(t_disc)>=datetime(2021, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(2024, 1, 1, 0, 0, 0)))[0]
splits = temporal.time_aware_train_test_split(X[idx,:], np.asarray(y)[idx], np.array(t_disc)[idx], train_size=12, test_size=1, granularity='month')
X_train_, X_test_, y_train_, y_test_, t_train, t_test = splits
print("X_test: ",len(X_test_))

if feat_type == "DENSE":
    mask = np.where(abs(X_train_.sum(axis=0))>=X_train_.shape[0]//100)[1]
    X_train_ = np.array(X_train_[:,mask].todense())
    X_test_ = [np.array(x_test[:,mask].todense()) for x_test in X_test_]
    y_train = y_train_
    y_test = y_test_
elif feat_type == 'FULL':    
    y_train = y_train_
    y_test = y_test_
elif feat_type == 'APIGraph':
    feat_heads = [i.split("::")[0] for i in features]
    AGIndex = np.where(np.array(feat_heads)=="apigraph")[0].tolist()
    feat_type = "APIGraph"
    mask = np.where(X_train_.sum(axis=0)>=X_train_.shape[0]//100)[1].tolist()+AGIndex
    mask = sorted(list(set(mask)))
    X_train_ = np.array(X_train_[:,mask].todense())
    X_test_ = [np.array(x_test[:,mask].todense()) for x_test in X_test_]
    y_train = y_train_
    y_test = y_test_
elif feat_type == 'VR':
    selector = VarianceThreshold(threshold=0.003)
    selector.fit(X_train_)
    mask = selector.get_support()
    print(f"Original features: {X_train.shape[1]}")
    X_train_ = X_train_[:,mask].toarray()
    X_test_ = [x_test[:,mask].toarray() for x_test in X_test_]
    print(f"Selected features (variance > 0.003): {X_train_.shape[1]}")

#Start active learning
data_id = "drebin"
f1_dict = dict()
fpr_dict = dict()
fnr_dict = dict()
month_strings = []
for year in [2022, 2023]:
    for month in range(1, 13):
        month_str = f"{year}-{month:02d}"
        month_strings.append(month_str)
downsample = False
for model_type in ['G-MOE']:#nn_notime,"warm","G-MOE","nn_time","moe_wg","moe_o",linearsvm,rf
    for COUNT in [50,100,200,400]:
        random.seed(0)
        X_train = X_train_.copy()
        X_test = [x_test.copy() for x_test in X_test_]
        y_train = y_train_[:X_train_.shape[0]]
        y_test = y_test_
        print(X_train[:,:-1].sum(),y_train.sum())
        t_tr = t_train.copy()
        t_te = t_test.copy()

        if downsample:
            print("Before downsample: ",X_train.shape)
            month_indices = defaultdict(list)
            for idx, dt in enumerate(t_tr):
                # 生成 key：如 "2021-01"
                month_key = f"{dt.year}-{dt.month:02d}"
                month_indices[month_key].append(idx)
            np.random.seed(42)  # 固定随机，可复现
            selected_idx = []
            for month, idxs in month_indices.items():
                if len(idxs) >= 400:
                    sampled = np.random.choice(idxs, size=400, replace=False)
                else:
                    sampled = idxs
                selected_idx.extend(sampled)
            # 转成 numpy 数组
            selected_idx = np.array(selected_idx)
            X_train = X_train[selected_idx]
            y_train = y_train[selected_idx]
            print("After downsample:",X_train.shape)
        f1_list = []
        fpr_list = []
        fnr_list = []
        f1_list_filtered = []
        fpr_list_filtered = []
        fnr_list_filtered = []
        update_set = []
        accumulated_samples = 0
        for i in range(len(X_test)):
            t_tr_temp = np.array([i.timestamp() for i in t_tr]).reshape(-1,1)
            t_te_temp = np.array([i.timestamp() for i in t_te[i]]).reshape(-1,1)
            if model_type not in ["linearsvm","rf"]:
                #Renormalized for every month
                scaler_temp = MinMaxScaler()
                X_train[:,-1] = scaler_temp.fit_transform(X_train[:,-1:]).reshape(-1)
                X_test[i][:,-1] = scaler_temp.transform(X_test[i][:,-1:]).reshape(-1)
                print(X_train[:,-1],X_test[i][:,-1])
            model_name = f"al_{model_type}_{i}_{COUNT}{metric}{feat_type}"
            if model_type == "G-MOE":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model('moe',data_id,X_train, y_train, X_train, y_train, epoch=100)

                    model_utils.save_model("moe", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model("moe", data_id, "models/", model_name, X_train.shape[1])
                x_t = X_test[i].copy()
                x_t[:,-1] = 1
                y_pred = nn.predict(x_t).numpy()
            elif model_type == "nn_time":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model('nn',data_id,X_train, y_train, X_train, y_train, epoch=100)
                    model_utils.save_model("nn", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model("nn", data_id, "models/", model_name, X_train.shape[1])
                x_t = X_test[i].copy()
                x_t[:,-1] = 1
                y_pred = nn.predict(x_t).numpy()
            elif model_type == "moe_wg":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model('moe_wg',data_id,X_train, y_train, X_train, y_train, epoch=100)
                    model_utils.save_model("moe", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model('moe_wg', data_id, "models/", model_name, X_train.shape[1])
                x_t = X_test[i].copy()
                x_t[:,-1] = 1
                y_pred = nn.predict(x_t).numpy()
            elif model_type == "moe_o":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model('moe_o',data_id,X_train[:,:-1], y_train, X_train[:,:-1], y_train, epoch=100)
                    model_utils.save_model("moe_o", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model('moe_o', data_id, "models/", model_name, X_train.shape[1]-1)
                y_pred = nn.predict(X_test[i][:,:-1])#.numpy()
            elif model_type == "linearsvm" or model_type == "rf":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model(model_type,data_id,X_train[:,:-1], y_train, X_train[:,:-1], y_train, epoch=100)
                    model_utils.save_model(model_type, nn, "models/", model_name)
                else:
                    nn = model_utils.load_model(model_type, data_id, "models/", model_name, X_train.shape[1]-1)
                y_pred = nn.predict(X_test[i][:,:-1])#.numpy()
            elif model_type == "nn_notime" or i==0:
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = model_utils.train_model('nn',data_id,X_train[:,:-1], y_train, X_train[:,:-1], y_train, epoch=100)
                    model_utils.save_model("nn", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model("nn", data_id, "models/", model_name, X_train.shape[1]-1)
                y_pred = nn.predict(X_test[i][:,:-1]).numpy()
            elif model_type == "warm":
                if True:#not os.path.exists(f"models/{model_name}.pkl"):
                    nn = NN(X_train_.shape[1]-1,data_id,512,False)
                    pre_model = f"al_{model_type}_{i-1}_{COUNT}{metric}{feat_type}"
                    nn.load("models/",pre_model)
                    #nn.opt = torch.optim.Adam(nn.net.parameters(), lr=1e-3)
                    nn.fit(X_train[:,:-1], y_train, X_train[:,:-1], y_train, 100, "")
                    model_name = f"al_{model_type}_{i}_{COUNT}{metric}{feat_type}"
                    model_utils.save_model("nn", nn, "models/", model_name)
                else:
                    nn = model_utils.load_model("nn", data_id, "models/", model_name, X_train.shape[1]-1)
                y_pred = nn.predict(X_test[i][:,:-1]).numpy()
            f1 = f1_score(y_pred>0.5,y_test[i])
            f1_list.append(f1)
            tn, fp, fn, tp = confusion_matrix(y_test[i], y_pred>0.5).ravel()
            fpr = fp / (fp + tn + 1e-10)  # 1e-10防止分母为0
            fpr_list.append(fpr)
            fnr = fn / (fn + tp + 1e-10)
            fnr_list.append(fnr)
            print(f"{metric},{month_strings[i]},{model_type},{COUNT},{fpr},{fnr},{f1}\n")
            #ff.write(f"{month_strings[i]},{model_type},{COUNT},{fpr},{fnr},{f1}\n")
            #Assume the best active learning strategies
            if metric == 'uncertainty':
                selected = abs(y_pred-0.5).argsort()
                if scipy.sparse.issparse(X_test[i]):
                    #unique is not available for sparse matrix
                    feat_idx = np.where(X_test[i][:,:-1].sum(axis=0)!=0)[1]
                    X_temp = X_test[i][:,feat_idx]
                    unique_vectors, indices = np.unique(X_temp[selected].toarray(), axis=0, return_index=True)
                    indices.sort()
                    selected = selected[indices[:COUNT]]
                else:
                    unique_vectors, indices = np.unique(X_test[i][selected,:-1], axis=0, return_index=True)
                    indices.sort()
                    selected = selected[indices[:COUNT]]
                    #selected = selected[:COUNT]
            else:
                raise Exception
            accumulated_samples += COUNT
            if scipy.sparse.issparse(X_test[i]):
                X_train = vstack([X_train,X_test[i][selected]])
            else:
                X_train = np.vstack([X_train,X_test[i][selected]])
            y_train = np.hstack([y_train,y_test[i][selected]])
            t_tr = np.hstack([t_tr,t_te[i][selected]])

            update_set.append(abs(y_pred[selected]-y_test_[i][selected]))
            #filter
            mask1 = np.ones(len(y_pred), dtype=bool)
            mask1[selected] = False
            f1 = f1_score(y_pred[mask1]>0.5,y_test[i][mask1])
            f1_list_filtered.append(f1)
            tn, fp, fn, tp = confusion_matrix(y_test[i][mask1], y_pred[mask1]>0.5).ravel()
            fpr = fp / (fp + tn + 1e-10)  # 1e-10防止分母为0
            fpr_list_filtered.append(fpr)
            fnr = fn / (fn + tp + 1e-10)
            fnr_list_filtered.append(fnr)
            print(f"Filtered: {metric},{month_strings[i]},{model_type},{COUNT},{fpr},{fnr},{f1}\n")
        f1_dict[model_type+str(COUNT)+"base"] = f1_list
        f1_dict[model_type+str(COUNT)+"filter"] = f1_list_filtered
        fpr_dict[model_type+str(COUNT)+"base"] = fpr_list
        fpr_dict[model_type+str(COUNT)+"filter"] = fpr_list_filtered
        fnr_dict[model_type+str(COUNT)+"base"] = fnr_list
        fnr_dict[model_type+str(COUNT)+"filter"] = fnr_list_filtered
        print(f"* Result: {metric},{month_strings[i]},{model_type},{COUNT}, FPR: {sum(fpr_list)/24}, FNR: {sum(fnr_list)/24},F1: {sum(f1_list)/24}\n")
        print(f"* Result: {metric},{month_strings[i]},{model_type},{COUNT}, FPR: {sum(fpr_list_filtered)/24}, FNR: {sum(fnr_list_filtered)/24},F1: {sum(f1_list_filtered)/24} - Filtered")
        ff.write(f"{feat_type},{metric},{month_strings[i]},{model_type},{COUNT},{sum(fpr_list)/24},{sum(fnr_list)/24},{sum(f1_list)/24}\n")
        ff.write(f"{feat_type+"_filtered"},{metric},{month_strings[i]},{model_type},{COUNT},{sum(fpr_list_filtered)/24},{sum(fnr_list_filtered)/24},{sum(f1_list_filtered)/24}\n")
        ff.flush()
