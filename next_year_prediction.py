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
from core import model_utils, data_utils
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
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from process_data_bundle import preprocess,combine
from sklearn.feature_selection import VarianceThreshold
ff = open("results/next_year_prediction_gp.txt","a")
ff.write("===================================================\n")
ff.flush()

import numpy as np
from collections import defaultdict


def evaluate(net,X_test,y_test_,x_val,y_val):
    r = net.predict(x_val)>0.5
    iid_f1 = metrics.f1_score(r,y_val)
    r = net.predict(X_test)>0.5
    final_f1 = metrics.f1_score(r,y_test_)
    print("IID:",iid_f1)
    print("F1:", final_f1)
    return iid_f1, final_f1

dataset = "2024-GP1"#"2024-apigraph"#"2024-GP1"#'2024-mamadroid'
for dataset in ['2024-GP1']:#combined
    feat_type = "DENSE"
    data_id = dataset#"drebin"#"mamadroid"
    if feat_type not in ["HCC","TIF"]:
        X,y_o,t_disc,(shalist, features) = data_utils.load_gp_dataset(dataset)
        y = np.asarray(y_o)
        dimension = X.shape[1]
    start = 2023
    end = 2014
    #for fold in range(0,3):
    fold = 0
    alpha = 0
    for fold in range(3):
        #for fold in [0,1,2]:
        for year in range(start,end,-1):
            if feat_type == "HCC":
                X,y_o, t_disc, (shalist,features) = joblib.load(f"datasets/HCC/2014_{year}.pkl")
                dimension = X.shape[1]
                y = np.asarray(y_o)
            #use training set for sclaing the time
            t_appear = np.array(t_disc)[np.where((np.array(t_disc)>=datetime(end, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(year, 1, 1, 0, 0, 0)))[0]]
            t_appear = [i.timestamp() for i in t_appear]
            t_all = [i.timestamp() for i in t_disc]
            scaler = MinMaxScaler()
            scaler.fit(np.array(t_appear).reshape(-1,1))
            t_normalized = scaler.transform(np.array(t_all).reshape(-1,1))
            if dimension+1 <= X.shape[1]:
                X[:,-1] = t_normalized.reshape(-1)
            else:
                if scipy.sparse.issparse(X):
                    X = hstack([X,t_normalized]).tocsr()
                else:
                    X  = np.hstack([X,t_normalized])
            #Get training set and testing test
            idx = np.where((np.array(t_disc)>=datetime(end, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(year+1, 1, 1, 0, 0, 0)))[0]
            t_size = (year - end)*12
            splits = temporal.time_aware_train_test_split(X[idx,:], np.asarray(y_o)[idx], np.array(t_disc)[idx], train_size=t_size, test_size=1, granularity='month')
            X_train_, X_test_, y_train_, y_test_, t_train, t_test = splits
            print(f"{year}: Train {t_train[:]} {t_test[0]}")
            if feat_type == "DENSE":
                #get samples with 1% density
                mask = np.where(abs(X_train_.sum(axis=0))>=X_train_.shape[0]//100)[1]
                X_train = np.array(X_train_[:,mask].toarray())
                X_test = [np.array(x_test[:,mask].toarray()) for x_test in X_test_]
                X_test = np.concatenate(X_test,axis=0)
                y_test_ = np.concatenate(y_test_)
            elif feat_type == "APIGRAPH":
                feat_heads = [i.split("::")[0] for i in features]
                AGIndex = np.where(np.array(feat_heads)=="apigraph")[0].tolist()
                mask = np.where(X_train_.sum(axis=0)>=X_train_.shape[0]//100)[1].tolist()+AGIndex
                mask = sorted(list(set(mask)))
                print(len(mask))
                X_train = X_train_[:,mask].toarray()
                X_test = [x_test[:,mask].toarray() for x_test in X_test_]
                X_test = np.concatenate(X_test,axis=0)
                y_test_ = np.concatenate(y_test_)
            elif feat_type in ["HCC","mamadroid","malscanscb","mamadroidscb"] or "scb" in feat_type:
                #get samples with 1% density
                X_train = X_train_
                X_test = np.concatenate(X_test_,axis=0)
                y_test_ = np.concatenate(y_test_)
            elif feat_type == 'malscan':
                selector = VarianceThreshold(threshold=0.003)
                selector.fit(X_train_)
                mask = selector.get_support()
                # Optional: Print results
                print(f"Original features: {X_train_.shape[1]}")
                X_train = X_train_[:,mask].toarray()
                X_test = np.concatenate([x_test[:,mask].toarray() for x_test in X_test_],axis=0)
                y_test_ = np.concatenate(y_test_)
                print(f"Selected features (variance > 0.003): {X_train.shape[1]}")
                
    
            indicies = np.arange(X_train.shape[0])
            from sklearn.model_selection import train_test_split
            x_train_new, x_val, y_train_new, y_val, train_indicies, val_indicies = train_test_split(
                X_train,          
                y_train_,        
                indicies,
                test_size=0.1,  
                random_state=0, 
                shuffle=True   
            )
            print("Train shape:",x_train_new.shape)
            epoch = 150
            method = ""# if "scb" not in data_id else "timedensity0"
            if feat_type == "DENSE":
                print("NN-time")
                if not os.path.exists(f"models/nn_time_{year}_{fold}_{data_id}_{feat_type}.pkl"):
                    #nn = model_utils.train_model('nn',data_id,x_train_new, y_train_new, x_val, y_val, epoch=epoch, method=method)
                    nn = model_utils.train_model('nn',data_id,X_train, y_train_, X_train, y_train_, epoch=epoch, method=method)
                    model_utils.save_model("nn", nn, "models/", f"nn_time_{year}_{fold}_{data_id}_{feat_type}")
                else:
                    nn = model_utils.load_model("nn", data_id, "models/", f"nn_time_{year}_{fold}_{data_id}_{feat_type}",x_train_new.shape[1])
                iid_f1, final_f1 = evaluate(nn,X_test,y_test_,x_val,y_val)
                ff.write(f"Testing on {year} - NN-time: IID F1 {iid_f1}, OOD F1 {final_f1} \n") 
                ff.flush()

                if not os.path.exists(f"models/moe_o_{year}_{fold}_{data_id}_{feat_type}.pkl"):
                    nn = model_utils.train_model('moe_o',data_id,X_train[:,:-1], y_train_, X_train[:,:-1],y_train_, epoch=epoch, method=method)#IID
                    model_utils.save_model("moe", nn, "models/", f"moe_o_{year}_{fold}_{data_id}_{feat_type}")
                else:
                    nn = model_utils.load_model("moe_o", data_id, "models/",  f"moe_o_{year}_{fold}_{data_id}_{feat_type}", x_train_new[:,:-1].shape[1])
                iid_f1, final_f1 = evaluate(nn,X_test[:,:-1],y_test_,x_val[:,:-1],y_val)
                ff.write(f"Testing on {year} - MOE original time: IID F1 {iid_f1}, OOD F1 {final_f1} \n") 
                ff.flush()
                print("MOE original")
                #
                print("T-MOE")
                if not os.path.exists(f"models/moe_wg_{year}_{fold}_{data_id}_{feat_type}.pkl"):
                    nn = model_utils.train_model('moe_wg',data_id,X_train, y_train_, X_train, y_train_, epoch=epoch, method=method)#IID
                    model_utils.save_model("moe", nn, "models/", f"moe_wg_{year}_{fold}_{data_id}_{feat_type}")
                else:
                    nn = model_utils.load_model("moe_wg", data_id, "models/",  f"moe_wg_{year}_{fold}_{data_id}_{feat_type}", x_train_new.shape[1])
                iid_f1, final_f1 = evaluate(nn,X_test,y_test_,x_val,y_val)
                ff.write(f"Testing on {year} - T-MOE: IID F1 {iid_f1}, OOD F1 {final_f1} \n") 
                ff.write(f"========================================================================= {fold}_{data_id}_{feat_type}\n") 
                ff.flush()

            print("NN")
            if not os.path.exists(f"models/nn_{year}_{fold}_{data_id}_{feat_type}.pkl"):
                nn = model_utils.train_model('nn',data_id,X_train[:,:-1], y_train_, X_train[:,:-1],y_train_, epoch=epoch, method=method)#IID
                model_utils.save_model("nn", nn, "models/", f"nn_{year}_{fold}_{data_id}_{feat_type}")
            else:
                nn = model_utils.load_model("nn", data_id, "models/",  f"nn_{year}_{fold}_{data_id}_{feat_type}", x_train[:,:-1].shape[1])
            iid_f1, final_f1 = evaluate(nn,X_test[:,:-1],y_test_,x_val[:,:-1],y_val)
            ff.write(f"Testing on {year} - NN: IID F1 {iid_f1}, OOD F1 {final_f1} \n") 
            ff.flush()
            
            print("G-MOE")
            if True:#not os.path.exists(f"models/moe_{year}_{fold}_{data_id}_{feat_type}.pkl"):
                nn = model_utils.train_model('moe',data_id,X_train, y_train_, X_train, y_train_, epoch=epoch, method=method)#IID
                model_utils.save_model("moe", nn, "models/", f"moe_{year}_{fold}_{data_id}_{feat_type}")
            else:
                nn = model_utils.load_model("moe", data_id, "models/", f"moe_{year}_{fold}_{data_id}_{feat_type}", x_train_new.shape[1])
            iid_f1, final_f1 = evaluate(nn,X_test,y_test_,x_val,y_val)
            ff.write(f"Testing on {year} {alpha} - G-MOE: IID F1 {iid_f1}, OOD F1 {final_f1} \n") 
            ff.flush()
    
            #del X
            del X_train
            del X_test
            del x_train_new
            import gc
            gc.collect()
