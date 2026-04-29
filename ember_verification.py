import torch
import os
import numpy as np
import ember
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from core import data_utils
from core import model_utils
from core import utils
from core import constants
import joblib
import lightgbm as lgb
from tqdm import tqdm
import time
from datetime import datetime
import joblib
import torch
import torch.nn.functional as F
from core import utils
from sklearn.preprocessing import StandardScaler
from logger import logger
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder

## multi-classes
import tempfile
def train(X_train, y_train, x_test, y_test, batch_size, net, loss, optimizer, device, num_epochs, method='', *args):
    net = net.to(device)
    logger.info("training on "+device)
    temp_filename = tempfile.mktemp()
    best_acc = 0
    best_f1 = 0
    best_val_loss = 1e35
    best_train_loss = 1e35
    if "density" in method or 'crop' in method:
        if 'density' in method:
            fn = float(method.split('density')[-1])
            logger.info(f"Density-based robust training: {fn}")
        coredict = utils.get_coredict(X_train)
        utils.transform_as_prob(coredict)#transform values and densities into sparse distribution

    for epoch in tqdm(range(num_epochs)):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        batch_count = 0
        train_iter = utils.data_iter(batch_size, X_train, y_train)
        for X, y in train_iter:
            if 'density' in method:
                x_list = [X]
                y_list = [y]
                if fn >= 0:
                    for ii in range(101):
                        X_filled = utils.fill_density(X[y==ii].clone(), coredict, fn)
                        x_list.append(X_filled)
                        y_list.append(y[y==ii])
                X = torch.cat(x_list)
                y = torch.cat(y_list)
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        net.eval()
        if not isinstance(x_test, np.ndarray):
            logger.info('epoch %d, loss %.4f,  train acc %.5f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
            torch.save(net.state_dict(), os.path.join('tmp/', temp_filename + '.pkl'))
            continue
        #calculate val loss
        y_pred = utils.predict(net, x_test, device=None, batch=512, args=[])
        f1 = f1_score(y_test,y_pred.argmax(axis=1),average='macro')
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(net.state_dict(), os.path.join('tmp/', temp_filename + '.pkl'))
        logger.info('epoch %d, loss %.4f, train acc %.5f, test f1 %.5f, best f1 %.5f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, f1, best_f1, time.time() - start))
    #Loading the model performing best on the validation set.
    if isinstance(x_test, np.ndarray):
        net.load_state_dict(torch.load(os.path.join('tmp/', temp_filename + '.pkl'),weights_only=True))
        os.remove(os.path.join('tmp/', temp_filename + '.pkl'))
    return net

# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack

from sklearn import metrics
#import transcend.utils as utils
import json
import numpy as np
from datetime import datetime
from core import temporal
from core import utils as cutils
from core import model_utils
from logger import logger
import joblib
dataset = 'extended-features'
#import transcend.data as data
from scipy.sparse import vstack
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
class MOENet(nn.Module):
    def __init__(self, input_dim, hidden, num_experts, output_dim = 2):
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.centers = nn.Parameter(torch.linspace(0,1,num_experts))  #use a fixed start point, not good
        #self.centers = nn.Parameter(torch.randn(num_experts))
        init = torch.full((num_experts,), 1.0)#1/2*num_experts)
        #self.temprature = nn.Parameter(torch.tensor(0.1))
        #self.register_buffer('log_sigma', torch.log(init))
        self.log_sigma = nn.Parameter(torch.log(init))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim-1, hidden),#removing time feature from expert is better
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                torch.nn.Dropout(0.5),
                nn.Linear(hidden, output_dim),
            ) for _ in range(num_experts)
        ])
        # 门控网络（使用最后一列特征作为输入） #考虑结合
        self.gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts),  #
        )
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim-1, hidden),#removing time feature from expert is better
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.5),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.5),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        timestamp = x[:, -1].unsqueeze(1) # (B,1)
        #size = x[:, -2:]#.unsqueeze(1) # (B,1)
        #market = x[:, -2:] # (B,1)
        centers = self.centers.unsqueeze(0)  # (1,N)
        B = x.size(0)
        # 4) 计算 Gaussian 门控权重
        #    w_ij = exp(- (t_j - c_i)^2 / (2σ^2) )
        #route1 = F.softmax(self.gate(timestamp),dim=1)

        sigma = torch.exp(self.log_sigma).unsqueeze(0) #(1,N)
        numerator = timestamp - centers
        route = torch.exp(- numerator.pow(2) / (2 * sigma.pow(2))) #/(sigma*torch.sqrt(2*torch.pi))  # (B, N)
        expert_outputs = torch.stack([expert(x[:,:-1]) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', route, expert_outputs)
        return output


class NN(object):
    def __init__(self, n_features, data_id, hidden=1024, moe=False):
        self.n_features = n_features
        self.normal = StandardScaler()
        self.hidden = hidden
        self.data_id = data_id
        if moe:
            self.net = self.build_moe(moe)
        else:
            self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.exp = None
        self.lr = 3e-4
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999),weight_decay=1e-3)

    def fit(self, X, y, x_test, y_test, epoch, method):
        self.net.train()
        batch_size = 1024
        logger.debug("batch size:",batch_size)
        train(X, y, x_test, y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        self.net.eval()

    def predict(self, X):
        if self.data_id in ['ember','pdf']:
            X[:,:-1] = self.normal.transform(X[:,:-1])
        return utils.predict(self.net, X)

    def build_model(self):
        hidden = self.hidden
        layer_sizes = None
        layers = []
        p = 0.5
        if layer_sizes is None:layer_sizes=[512,512,256]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(torch.nn.Linear(self.n_features,ls))
            else:
                layers.append(torch.nn.Linear(layer_sizes[i-1],ls))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(ls))
            #layers.append(torch.nn.LayerNorm(ls))
            layers.append(torch.nn.Dropout(p))
        layers.append(torch.nn.Linear(ls,11))
        net = torch.nn.Sequential(*tuple(layers))
        return net

    def build_moe(self,moe):
        nn = MOENet(input_dim=self.n_features, hidden=self.hidden, num_experts=moe, output_dim=11)
        return nn

    def save(self, save_path, file_name='nn'):
        joblib.dump(self.normal, os.path.join(save_path, file_name + '_scaler.pkl'))
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        self.normal = joblib.load(os.path.join(save_path, file_name + '_scaler.pkl'))
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl')))

if __name__ == "__main__":
    model_id = 'lightgbm'
    task = "binary"#family
    method = "density0" if model_id!='lightgbm' else ''
    if model_id == "lightgbm" or model_id == 'nn':
        x_train, y_train, emberdf = data_utils.load_ember(dataset='emberall')
    elif model_id == "moe":
        x_train,y_train,emberdf = joblib.load("materials/processed_ember.pkl")
        #Processed dataset using SCB
        t_disc = np.array(emberdf.appeared.tolist())
        t_disc[t_disc<"2017-01"] = "2017-01"#only few older samples, merge them into 2017
        t_disc = np.array([datetime.strptime(s, "%Y-%m") for s in t_disc])
    
        if task == 'binary':
            t_appear = np.array(t_disc)[np.where((np.array(t_disc)>=datetime(2017, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(2018, 11, 1, 0, 0, 0)))[0]]
        else:
            t_appear = np.array(t_disc)[np.where((np.array(t_disc)>=datetime(2018, 1, 1, 0, 0, 0))&(np.array(t_disc)<datetime(2018, 11, 1, 0, 0, 0)))[0]]
        t_appear = [i.timestamp() for i in t_appear]
        t_all = [i.timestamp() for i in t_disc]
        scaler = MinMaxScaler()
        scaler.fit(np.array(t_appear).reshape(-1,1))
        t_normalized = scaler.transform(np.array(t_all).reshape(-1,1))
        x_train[:,-1] = t_normalized.reshape(-1)    
    x_train, x_test, y_train, y_test = x_train[:-200000],x_train[-200000:],y_train[:-200000],y_train[-200000:]
    if task == "binary":
        name = f"{model_id}_2017_2018_ember"
        saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
        if not os.path.exists(os.path.join(saved_dir,name+".pkl")):
            base_lgb = model_utils.train_model(model_id, "ember", x_train, y_train, epoch=200,method=method)#density0
            model_utils.evaluate_model(base_lgb,x_test,y_test)
            model_utils.save_model(model_id,base_lgb,saved_dir,name)
        else:
            base_lgb = model_utils.load_model(model_id,'ember',saved_dir,name)
            model_utils.evaluate_model(base_lgb,x_test,y_test)
    else:
        emberdf2018tr = emberdf[:-200000].copy()
        emberdf2018te = emberdf[-200000:].copy()    
        x_train = x_train[~emberdf2018tr['avclass'].isna()]
        x_test = x_test[~emberdf2018te['avclass'].isna()]
        emberdf2018tr = emberdf2018tr[~emberdf2018tr['avclass'].isna()]
        emberdf2018te = emberdf2018te[~emberdf2018te['avclass'].isna()]
        counts = emberdf2018tr.avclass.value_counts()
        valid_classes = counts[:10].index
        emberdf2018tr.loc[~emberdf2018tr['avclass'].isin(valid_classes),"avclass"] = 'others'
        emberdf2018te.loc[~emberdf2018te['avclass'].isin(valid_classes),"avclass"] = 'others'
        
        le = LabelEncoder()
        y_train = le.fit_transform(emberdf2018tr.avclass)
        y_test = le.transform(emberdf2018te.avclass)
        params = {
            "boosting": "gbdt",
            "objective": "multiclass",
            'metric': 'multi_logloss',
            "num_class": np.unique(y_train).size,
        }
        
        saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
        if model_id == "lightgbm":
            name = "multi_lightgbm"
            if not os.path.exists(os.path.join(saved_dir,name+".pkl")):
                lgbm_dataset = lgb.Dataset(x_train[:,:], y_train)
                base_lgb = lgb.train(params, lgbm_dataset)
                model_utils.save_model(model_id,base_lgb,saved_dir,name)
            else:
                base_lgb = model_utils.load_model(model_id,'ember',saved_dir,name)
            y_pred = base_lgb.predict(x_test[:,:])
            print(classification_report(y_test,y_pred.argmax(axis=1),digits=5))
        elif model_id == 'nn':
            name = "multi_nn"
            net = NN(x_train.shape[1], 'drebin', hidden=1024, moe=False)
            if not os.path.exists(os.path.join(saved_dir,name+".pkl")):
                net.fit(x_train,y_train,x_test,y_test,100,method) 
                net.save(saved_dir,name)
            else:
                net.load(save_dir,name)
            y_pred = net.predict(x_test)
            print(classification_report(y_test,y_pred.argmax(axis=1),digits=5))
        elif model_id == 'moe':
            from sklearn.preprocessing import StandardScaler,MinMaxScaler
            train_dis = pd.to_datetime(emberdf2018tr['appeared']).astype(np.int64)/1e18
            test_dis = pd.to_datetime(emberdf2018te['appeared']).astype(np.int64)/1e18
            scaler = MinMaxScaler()
            scaler.fit(np.array(train_dis).reshape(-1,1))
            t_train = scaler.transform(np.array(train_dis).reshape(-1,1))
            t_test = scaler.transform(np.array(test_dis).reshape(-1,1))
            x_train[:,-1] = t_train.reshape(-1)
            x_test[:,-1] = t_test.reshape(-1)
            name = "multi_moe"
            net = NN(x_train.shape[1], 'drebin', hidden=1024, moe=16)
            if not os.path.exists(os.path.join(saved_dir,name+".pkl")):
                net.fit(x_train,y_train,x_test,y_test,100,method) 
                net.save(saved_dir,name)
            else:
                net.load(save_dir,name)
            y_pred = net.predict(x_test)
            print(classification_report(y_test,y_pred.argmax(axis=1),digits=5))
    
