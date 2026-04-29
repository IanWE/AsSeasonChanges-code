import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
import json
import numpy as np
import pandas as pd
from datetime import datetime
from core import temporal
from core import utils as cutils
from core import model_utils, data_utils
from logger import logger
import joblib
#import transcend.data as data
from scipy.sparse import vstack
import argparse
from process_data_bundle import combine, preprocess
from sklearn.feature_selection import VarianceThreshold

def train_model(splits, year, month, model_id, timefeat='', data_id="drebin", year_of_test=2023, feat_type="DENSE", features="", seed=0):
    #print(X_train_.shape, t_test)
    X_train_, X_test_, y_train_, y_test_, t_train, t_test = splits
    print(year,month)
    mask = np.array([(dt.year > year)|((dt.year == year)&(dt.month>month)) for dt in t_train])
    t_train = t_train[mask]
    X_train = X_train_[mask]
    y_train = y_train_[mask]
    if model_id == 'nn1': 
        #if model_id is nn, separating testing set from training set
        indicies = np.arange(X_train.shape[0])
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, train_indicies, val_indicies  = train_test_split(
            X_train,          
            y_train,          
            indicies,
            test_size=0.1,    
            random_state=0,  
        )
        t_train = t_train[train_indicies]
        print(f"New training data size: {t_train.shape[0]}")
        print(f"New testing data size: {X_val.shape[0]}")
    else:
        X_val, y_val = X_train, y_train
    #mask = np.array([dt.year >= year for dt in t_train])
    epoch = 150
    if data_id == 'drebin' or data_id=='downsampled':
        if model_id == 'linearsvm' or feat_type in ['TIF','HCC']:
            X_test = X_test_
        elif feat_type == "DENSE":
            mask = np.where(X_train_.sum(axis=0)>=X_train_.shape[0]//100)[1]
            #selector = VarianceThreshold(threshold=0.003)
            #selector.fit(X_train_)
            #mask = selector.get_support()
            X_train = X_train[:,mask].toarray()
            X_val = X_val[:,mask].toarray()
            X_test = [x_test[:,mask].toarray() for x_test in X_test_]
        elif feat_type == "APIGRAPH":
            feat_heads = [i.split("::")[0] for i in features]
            AGIndex = np.where(np.array(feat_heads)=="apigraph")[0].tolist()
            mask = np.where(X_train.sum(axis=0)>=X_train_.shape[0]//100)[1].tolist()+AGIndex
            mask = sorted(list(set(mask)))
            print(len(mask))
            X_train = X_train[:,mask].toarray()
            X_val = X_val[:,mask].toarray()
            X_test = [x_test[:,mask].toarray() for x_test in X_test_]
    elif data_id == 'malscan':
        selector = VarianceThreshold(threshold=0.001)
        selector.fit(X_train_)
        mask = selector.get_support()
        # Optional: Print results
        print(f"Original features: {X_train.shape[1]}")
        X_train = X_train[:,mask].toarray()
        X_val = X_val[:,mask].toarray()
        X_test = [x_test[:,mask].toarray() for x_test in X_test_]
        print(f"Selected features (variance > 0.003): {X_train.shape[1]}")
    elif data_id == 'malscan':
        X_test = X_test_
    elif "malscanscb" in data_id:
        X_test = X_test_

    print(X_train.shape,X_test[0].shape)
    method = ""#"density0" if "scb" in data_id else ""
    modelname = f"{model_id}_{data_id}_{year}_{month}_neg_{timefeat}_{year_of_test}_{feat_type}"
    print("Model:"+ modelname)
    if os.path.exists(f"models/{modelname}.pkl"):
        model = model_utils.load_model(model_id,data_id,'models/',modelname, X_train.shape[1])
    else:
        model = model_utils.train_model(model_id, data_id, X_train, y_train, X_val, y_val, epoch=epoch, method=method)
        model_utils.save_model(model_id, model,'models/',modelname)
    if model_id == 'nn': 
        r = model.predict(X_val)>0.5
        id_f1 = metrics.f1_score(r,y_val)
        print(f"ID Performance: {id_f1}")
    f1_list = []
    y_pred = []
    print(len(X_test))
    for x_test,y in zip(X_test,y_test_):
        #x_test[:,-1] = X_train[:,-1].min()
        r = model.predict(x_test)>0.5
        y_pred.append(r)
        f1_list.append(metrics.f1_score(r,y))
    aut = cutils.aut_score(f1_list)
    logger.info(f'F1 list: {f1_list}')
    r = metrics.f1_score(np.hstack(y_pred),np.hstack(y_test_))
    logger.info(f'Experiments {year} - {month} - AUT:{aut} - F1: {r} - {timefeat}')
    if model_id == 'nn':
        return round(id_f1,5), round(r,5)
    return round(r,5)

if __name__ == '__main__':
    #dataset = 'extended-features'
    ff = open("results/verification_of_negative_effect_gp.txt","a")
    for data_id in ["drebin"]:#,"malscanscb"]:
        for _ in range(1):
            for feat_type in ['DENSE']:#'DENSE',"APIGRAPH"]:#
                if data_id == 'drebin' and feat_type == 'DENSE':
                    dataset = '2024-GP1'
                elif data_id == 'drebin' and feat_type == 'APIGRAPH':
                    dataset = '2024-apigraph'
                else:
                    dataset = data_id
                X,y,t_disc,(shalist,features) = data_utils.load_gp_dataset(dataset)
                y = np.asarray(y)
                timefeat = "" #'appear'
                start_year = 2014
                end_year = 2023
                for model_id in ["nn"]:#'linearsvm'
                    if feat_type == 'APIGRAPH' and model_id == 'linearsvm':
                        continue
                    for year_of_test in range(end_year,start_year,-1):
                        idx = np.where(np.array(t_disc)<datetime(year_of_test+1, 1, 1, 0, 0, 0))[0]
                        t_size = (year_of_test - start_year)*12
                        splits = temporal.time_aware_train_test_split(X[idx], y[idx], np.array(t_disc)[idx], train_size=t_size, test_size=1, granularity='month')
                        aut_list = []
                        val_list = []
                        for year in range(start_year,year_of_test):
                            for month in [0]:
                                print(year,year_of_test)
                                r = train_model(splits, year, month, model_id, timefeat, data_id, year_of_test, feat_type, features,_)
                                if model_id == 'nn':
                                    val_list.append(r[0])
                                    aut_list.append(r[1])
                                else:
                                    aut_list.append(r)
                        print("===================================",year)
                        ff.write(f"{model_id},{feat_type},{year_of_test}")
                        for aut in aut_list:
                            ff.write(","+str(round(aut,5)))
                        ff.write("\n")
                        ff.flush()
                        print(aut_list)
                        print("===================================",year)
