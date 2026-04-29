import pickle
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
import time
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
from core import model_utils
from core import utils
import json
import numpy as np
from datetime import datetime
from core import temporal
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.sparse import vstack
from scipy.sparse import vstack,csr_matrix, hstack
import scipy
import pandas as pd
from core import utils, data_utils
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from process_data_bundle import preprocess,combine
ff = open("results/testing_for_invariance_learning.txt","a")
ff.write("===================================================\n")
ff.flush()
import numpy as np
from collections import defaultdict


def evaluate(net,X_test,y_test):
    f1_list = []
    for x_test,y_test in zip(X_test,y_test):
        r = net.predict(x_test)>0.5
        f1 = metrics.f1_score(r,y_test)
        f1_list.append(f1)
    return f1_list

year = 2014
end = 2024
fold = 0
feat_type = "malscanscb"#"DENSE"
for dataset in ["malscanscb"]:#'2024-GP1']:#,'2024-apigraph']:
    print(dataset)
    data_id = dataset
    if feat_type in ['DENSE','APIGraph',"malscan","malscanscb"]:
        X,y_o,t_disc,(shalist,features) = data_utils.load_gp_dataset(dataset)
    elif feat_type == "HCC":
        X,y_o, t_disc, (shalist,features) = joblib.load(f"datasets/HCC/2014_{year+1}.pkl")
    else:
        raise Exception(f"No dataset for feature type {feat_type}")
    y = np.array(y_o)
    t_appear = np.array(t_disc)[np.where((np.array(t_disc)>=datetime(year, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(year+1, 1, 1, 0, 0, 0)))[0]]
    t_appear = [i.timestamp() for i in t_appear]
    t_all = [i.timestamp() for i in t_disc]
    scaler = MinMaxScaler()
    scaler.fit(np.array(t_appear).reshape(-1,1))
    t_normalized = scaler.transform(np.array(t_all).reshape(-1,1))
    t_normalized[t_normalized>6] = 6
    print(t_normalized.min(),t_normalized.max())
    if scipy.sparse.issparse(X):
        X = hstack([X,t_normalized]).tocsr()
    else:
        X  = np.hstack([X,t_normalized])
    #Get training set and testing test
    idx = np.where((np.array(t_disc)>=datetime(year, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(end, 1, 1, 0, 0, 0)))[0]
    t_size = 1
    splits = temporal.time_aware_train_test_split(X[idx,:], np.asarray(y_o)[idx], np.array(t_disc)[idx], train_size=t_size, test_size=1, granularity='year')
    X_train_, X_test_, y_train_, y_test_, t_train, t_test = splits
    print(f"{year}: Train {t_train[:]} {t_test[0]} {len(X_test_)}")
    if feat_type == "DENSE":
        #get samples with 1% density
        mask = np.where(abs(X_train_.sum(axis=0))>=X_train_.shape[0]//100)[1]
        #mask = sorted(np.array(X_train_.sum(axis=0))[0].argsort()[-2000:])
        X_train = np.array(X_train_[:,mask].toarray())
        X_test = [np.array(x_test[:,mask].toarray()) for x_test in X_test_]
        y_test = y_test_
        y_train = y_train_
    elif feat_type == "APIGRAPH":
        feat_heads = [i.split("::")[0] for i in features]
        AGIndex = np.where(np.array(feat_heads)=="apigraph")[0].tolist()
        mask = np.where(X_train.sum(axis=0)>=X_train.shape[0]//100)[1].tolist()+AGIndex
        #mask = np.array(X_train_.sum(axis=0))[0].argsort()[-2000:].tolist()+AGIndex
        mask = sorted(list(set(mask)))
        print(len(mask))
        X_train = X_train[:,mask].toarray()
        X_test = [x_test[:,mask].toarray() for x_test in X_test_]
        y_test = y_test_
        y_train = y_train_
    elif feat_type == "HCC" or feat_type == "scb":
        X_train = X_train_
        X_test = X_test_
        y_test = y_test_
        y_train = y_train_
    elif feat_type == "malscan":
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.003)
        selector.fit(X_train_)
        mask = selector.get_support()
        print(f"Original features: {X_train_.shape[1]}")
        X_train = X_train_[:,mask].toarray()
        X_test = [x_test[:,mask].toarray() for x_test in X_test_]
        y_test = y_test_
        y_train = y_train_
        print(f"Selected features (variance > 0.003): {X_train_.shape[1]}")
    else:
        raise ValueError("Feature type {feat_type} are not included.")

    #malscan and malscanscb are with large value space and easy to overfit, so it's better to split into train and val
    if "malscan" in feat_type:
        indicies = np.arange(X_train.shape[0])
        from sklearn.model_selection import train_test_split
        X_train, x_val, y_train, y_val, train_indicies, val_indicies = train_test_split(
            X_train,          
            y_train_,        
            indicies,
            test_size=0.2,   
            random_state=0,  
            shuffle=True    
        )

    print("Train shape:",X_train.shape)
    epoch = 100
    #consider training method crop0 or timedensity0 can benefit the performance
    method = ""#"crop0" if "scb" in dataset else ""
    if feat_type == "DENSE":
        print("NN-time")
        model_id = "nn-time"
        if True:#not os.path.exists(f"models/nn_time_{year}_{fold}_{data_id}.pkl"):
            #nn = model_utils.train_model('nn',data_id,x_train_new, y_train_new, x_val, y_val, epoch=epoch, method=method)
            nn = model_utils.train_model('nn',data_id,X_train, y_train_, X_train, y_train_, epoch=epoch, method=method)
            model_utils.save_model("nn", nn, "models/", f"nn_time_{year}_{fold}_{data_id}")
        else:
            nn = model_utils.load_model("nn", data_id, "models/", f"nn_time_{year}_{fold}_{data_id}",x_train_new.shape[1])
        final_f1 = evaluate(nn,X_test,y_test)
        ff.write(f"Testing on {year} - {model_id} - {dataset}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
        ff.flush()

        print("MOE original")
        model_id = "moe_o"
        if not os.path.exists(f"models/moe_o_{year}_{fold}_{data_id}.pkl"):
            #nn = model_utils.train_model('moe_o',data_id,x_train_new[:,:-1], y_train_new, x_val[:,:-1], y_val, epoch=epoch, method=method)#IID
            nn = model_utils.train_model('moe_o',data_id,X_train[:,:-1], y_train_, X_train[:,:-1],y_train_, epoch=epoch, method=method)#IID
            model_utils.save_model("moe", nn, "models/", f"moe_o_{year}_{fold}_{data_id}")
        else:
            nn = model_utils.load_model("moe_o", data_id, "models/",  f"moe_o_{year}_{fold}_{data_id}", X_train[:,:-1].shape[1])
        X_test_temp = [x_test[:,:-1] for x_test in X_test]
        final_f1 = evaluate(nn,X_test_temp,y_test)
        ff.write(f"Testing on {year} - {model_id} - {dataset}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
        ff.flush()
        
        print("T-MOE")
        model_id = "moe_wg"
        if True:#not os.path.exists(f"models/moe_wg_{year}_{fold}_{data_id}.pkl"):
            nn = model_utils.train_model('moe_wg',data_id,X_train, y_train_, X_train, y_train_, epoch=epoch, method=method)#IID
            model_utils.save_model("moe", nn, "models/", f"moe_wg_{year}_{fold}_{data_id}")
        else:
            nn = model_utils.load_model("moe_wg", data_id, "models/",  f"moe_wg_{year}_{fold}_{data_id}", X_train.shape[1])
        final_f1 = evaluate(nn,X_test,y_test_)
        ff.write(f"Testing on {year} - {model_id} - {dataset}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
        ff.write(f"========================================================================= {fold}_{data_id}\n") 
        ff.flush()
    
    model_id = "nn"
    print(model_id)
    if True:#not os.path.exists(f"models/{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance.pkl"):
        nn = model_utils.train_model(model_id,data_id,X_train[:,:-1], y_train, X_train[:,:-1], y_train, epoch=epoch,method=method)#IID
        #nn = model_utils.train_model(model_id,data_id,X_train[:,:-1], y_train, x_val[:,:-1], y_val, epoch=epoch,method="")#IID
        model_utils.save_model(model_id, nn, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance")
    else:
        nn = model_utils.load_model(model_id, data_id, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance", X_train.shape[1]-1)
    X_test_temp = [x_test[:,:-1] for x_test in X_test]
    final_f1 = evaluate(nn,X_test_temp,y_test)
    ff.write(f"Testing on {year} - {model_id} - {dataset}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
    ff.flush()

    model_id = "moe"
    print(model_id)
    if True:#not os.path.exists(f"models/{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance.pkl"):
        #nn = model_utils.train_model(model_id,data_id,X_train, y_train, X_train, y_train, epoch=epoch,method=method)#for dense and apigraph, use this one
        nn = model_utils.train_model(model_id,data_id,X_train, y_train, x_val, y_val, epoch=epoch,method=method)#for malscan, trained with validation, as it is easily overfitted
        model_utils.save_model(model_id, nn, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance")
    else:
        nn = model_utils.load_model(model_id, data_id, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance", X_train.shape[1])
    final_f1 = evaluate(nn,X_test,y_test)
    ff.write(f"Testing on {year} - {model_id} - {dataset}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
    ff.flush()
    if dataset == "malscan":
        print("Random Forest")
        model_id = "rf"
        if not os.path.exists(f"models/{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance.pkl"):
            nn = model_utils.train_model(model_id,data_id,X_train_[:,:-1], y_train_, X_train_[:,:-1], y_train_, epoch=epoch)#IID
            model_utils.save_model(model_id, nn, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance")
        else:
            nn = model_utils.load_model(model_id, data_id, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance", X_train_.shape[1]-1)
        X_test_temp = [x_test[:,:-1] for x_test in X_test_]
        final_f1 = evaluate(nn,X_test_temp,y_test_)
        ff.write(f"Testing on {year} - {model_id} - {dataset} - {feat_type}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
        ff.flush()
    
    #model_id = "linearsvm"
    if dataset == "2024-GP1" and model_id == "linearsvm":
        print("Linear SVM")
        if not os.path.exists(f"models/{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance.pkl"):
            nn = model_utils.train_model(model_id,data_id,X_train_[:,:-1], y_train_, X_train_[:,:-1], y_train_, epoch=epoch)#IID
            model_utils.save_model(model_id, nn, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance")
        else:
            nn = model_utils.load_model(model_id, data_id, "models/", f"{model_id}_{year}_{fold}_{data_id}_{dataset}_{feat_type}_invariance", X_train_.shape[1]-1)
        X_test_temp = [x_test[:,:-1] for x_test in X_test_]
        final_f1 = evaluate(nn,X_test_temp,y_test_)
        ff.write(f"Testing on {year} - {model_id} - {dataset} - {feat_type}: {final_f1}, avg: {sum(final_f1)/len(final_f1)} \n") 
        ff.flush()

