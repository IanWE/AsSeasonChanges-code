import os
import torch
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

from core import constants, utils, data_utils
from core.nn import NN
from core.largenn import LargeNN
from core.moe.moe import MOENet
from core.moe.moe_wg import MOENet_wg
from core.moe.moe_o import MOENet_o


from logger import logger
# FRONT-END

def load_model(model_id, data_id, save_path, file_name, dimension = None):
    """ Load a trained model

    :param model_id: (str) model type
    :param data_id: (str) dataset id
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :param selected: (bool) feature type of drebin
    :return: trained model
    """
    if model_id == 'lightgbm':
        return load_lightgbm(
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'nn' or "moe" in model_id:
        return load_nn(
            data_id=data_id,
            save_path=save_path,
            file_name=file_name,
            dimension=dimension,
            moe=model_id
        )
    elif model_id == 'largenn':
        return load_largenn(
            data_id=data_id,
            save_path=save_path,
            file_name=file_name,
            dimension=dimension
        )
    elif model_id == 'rf':
        return load_rf(
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'linearsvm':
        return load_linearsvm(
            save_path=save_path,
            file_name=file_name
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))

def train_model(model_id, data_id, x_train, y_train, x_test=None, y_test=None, epoch=20, method='', normal=False):
    """ Train a classifier

    :param model_id: (str) model type
    :param data_id: (str) dataset type - ember/pdf/drebin
    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :param epoch: (int) epoch of training
    :param method: (str) type of robust training
    :return: trained classifier
    """
    if model_id == 'nn' or 'moe' in model_id:
        return train_nn(
            x_train=x_train,
            y_train=y_train,
            x_test = x_test,
            y_test = y_test,
            data_id = data_id,
            epoch = epoch,
            method = method,
            moe = model_id
        )

    elif model_id == 'largenn':
        return train_ltnn(
            x_train=x_train,
            y_train=y_train,
            x_test = x_test,
            y_test = y_test,
            data_id = data_id,
            epoch = epoch,
        )
    elif model_id == 'lightgbm' and data_id == 'ember':
        return train_lightgbm(
            x_train=x_train,
            y_train=y_train,
            epoch=epoch
        )

    elif model_id == 'rf':# and data_id in ['pdf', 'mamadroid', 'malscan']:
        return train_rf(
            x_train=x_train,
            y_train=y_train,
            data_id=data_id
        )
    elif model_id == 'linearsvm' and data_id == 'drebin':
        return train_linearsvm(
            x_train=x_train,
            y_train=y_train
        )
    else:
        raise NotImplementedError('Model {} with Dataset {} not supported'.format(model_id, data_id))


def save_model(model_id, model, save_path, file_name):
    """ Save trained model

    :param model_id: (str) model type
    :param model: (object) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    if model_id == 'lightgbm':
        return save_lightgbm(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'nn' or 'moe' in model_id or model_id == 'largenn':#including lt
        return save_nn(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'rf':
        return save_rf(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    elif model_id == 'linearsvm':
        return save_linearsvm(
            model=model,
            save_path=save_path,
            file_name=file_name
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))

def predict_compressed_data(model, x_test, processor):
    """ Predicate with the compression

    :param model: (object) binary classifier
    :param x_test: (ndarray) data to test
    :param processor: (object) compression processor

    :return: f1 measure
    """
    if processor is not None:
        x_t = x_test.copy()
        processor.process(x_t)
    return model.predict(x_t)

def evaluate_model(model, x_test, y_test, target=None):
    """ Print evaluation information of binary classifier

    :param model: (object) binary classifier
    :param x_test: (ndarray) data to test
    :param y_test: (ndarray) labels of the test set
    :param target: (int) setting of fixed false positive rate
    :return:
    """
    if target:
        pred = model.predict(x_test)# > 0.5
        auc, f1, tpr, thr = utils.evaluate(y_test,pred,target)
        pred = pred>thr
    else:
        pred = model.predict(x_test) 
        auc = roc_auc_score(y_test,pred)
        pred = pred>0.5
        f1 = f1_score(y_test,pred)
    logger.debug(pred)
    print(classification_report(y_test, pred, digits=7))
    conf_matrix = confusion_matrix(y_test, pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    logger.info(f"f1:{f1} , false positive rate:{FPR}, false negative rage:{FNR}")
    return f1

# LIGHTGBM
def load_lightgbm(save_path, file_name):
    """ Load pre-trained LightGBm model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained LightGBM model
    """

    model_path = os.path.join(save_path, file_name+".pkl")
    trained_model = lgb.Booster(model_file=model_path)
    return trained_model


def train_lightgbm(x_train, y_train, epoch):
    """ Train a LightGBM classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained LightGBM classifier
    """
    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "num_iterations": 1000,
        "learning_rate": 0.05,
        "num_leaves": 2048,
        "max_depth": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.5
    }
    lgbm_dataset = lgb.Dataset(x_train, y_train)
    if epoch <= 20:
        #training with default parameters
        lgbm_model = lgb.train({"application": "binary"}, lgbm_dataset)
    else:
        #training with finetuned parameters
        lgbm_model = lgb.train(params, lgbm_dataset)

    return lgbm_model


def save_lightgbm(model, save_path, file_name):
    """ Save trained LightGBM model

    :param model: (LightGBM) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    save_f = os.path.join(save_path, file_name + ".pkl")
    model.save_model(save_f)

# EMBERNN
def load_nn(data_id, save_path, file_name, dimension = None, moe="nn"):
    """ Load pre-trained NN model

    :param data_id: (str) dataset id
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained EmberNN model
    """

    if dimension == None:
        nfeat = constants.num_features[data_id]
    else:
        nfeat = dimension
    if data_id == "ember":
        hidden = 1024
    else:
        hidden = 512
    trained_model = NN(nfeat, data_id, hidden=hidden, moe=moe)
    trained_model.load(save_path, file_name)

    return trained_model

def load_largenn(data_id, save_path, file_name, dimension = None):
    """ Load pre-trained NN model

    :param data_id: (str) dataset id
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained EmberNN model
    """

    if dimension == None:
        nfeat = constants.num_features[data_id]
    else:
        nfeat = dimension
    trained_model = LargeNN(nfeat, data_id)
    trained_model.load(save_path, file_name)
    return trained_model


def train_largenn(x_train, y_train, x_test, y_test, data_id, epoch=20):
    """ Train an LTNN classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :param data_it: (str) dataset id

    :return: trained NN classifier
    """
    trained_model = LargeNN(x_train.shape[1],data_id)
    trained_model.fit(x_train, y_train, x_test, y_test, epoch)
    return trained_model

def train_nn(x_train, y_train, x_test, y_test, data_id, epoch, method, moe):
    """ Train an NN classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :param data_it: (str) dataset id
    :param method: (str) type of adversarial training

    :return: trained NN classifier
    """
    if data_id == "ember":
        hidden = 1024
    else:
        hidden = 512
    trained_model = NN(x_train.shape[1],data_id,hidden=hidden,moe=moe)
    trained_model.fit(x_train, y_train, x_test, y_test, epoch, method)
    return trained_model


def save_nn(model, save_path, file_name):
    """ Save trained NN model

    :param model: model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    model.save(save_path=save_path, file_name=file_name)


# PDFRate RANDOM FOREST
def train_rf(x_train, y_train, data_id='pdf'):
    """ Train a Random Forest classifier based on PDFRate

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :param data_id: (str) data id
    :return: trained Random Forest classifier
    """
    if data_id == 'pdf':
        # The parameters are taken from
        # https://github.com/srndic/mimicus/blob/master/mimicus/classifiers/RandomForest.py
        model = RandomForestClassifier(
            n_estimators=1000,  # Used by PDFrate
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=43,  # Used by PDFrate
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,  # Run in parallel
            random_state=43,
            verbose=0
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,  #
            random_state=43,
        )
    model.fit(x_train, y_train)
    return model


def save_rf(model, save_path, file_name):
    """ Save trained Random Forest model

    :param model: (RandomForestClassifier) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    joblib.dump(model, file_path)


def load_rf(save_path, file_name):
    """ Load pre trained Random Forest model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained Random Forest model
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    model = joblib.load(file_path)
    return model

# Drebin SVM classifier
def train_linearsvm(x_train, y_train):
    """ Train a Support Vector Machine classifier based on the Drebin paper

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: (LinearSVC) trained SVM classifier
    """
    model = LinearSVC(verbose=True)#, max_iter=10000)
    model.fit(x_train, y_train)

    return model


def save_linearsvm(model, save_path, file_name):
    """ Save trained Support Vector Machine model

    :param model: (LinearSVC) model object
    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return:
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    joblib.dump(model, file_path)


def load_linearsvm(save_path, file_name):
    """ Load pre trained Support Vector Machine model

    :param save_path: (str) path of save file
    :param file_name: (str) name of save file
    :return: trained SVM model
    """

    file_path = os.path.join(save_path, file_name + '.pkl')
    model = joblib.load(file_path)
    return model
