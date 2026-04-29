import os
import joblib
import torch
import torch.nn.functional as F

from core import utils
from sklearn.preprocessing import StandardScaler
from logger import logger
import numpy as np
import torch.nn as nn
class LargeNN(object):
    def __init__(self, n_features, data_id, hidden=1024):
        self.n_features = n_features
        self.normal = StandardScaler()
        self.hidden = hidden
        self.data_id = data_id
        self.lr = 0.001
        self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.exp = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999),weight_decay=1e-4)

    def fit(self, X, y, x_test, y_test, epoch, method):
        self.net.train()
        batch_size = 2048
        logger.debug("batch size:",batch_size)
        if self.data_id in ['ember','pdf']:
            logger.debug("It's EMBER data")
            #print(self.features_postproc_func(X))
            self.normal.fit(X)
            utils.train(self.normal.transform(X), y, self.normal.transform(x_test), y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        else:
            utils.train(X, y, x_test, y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        self.net.eval()

    def predict(self, X):
        if self.data_id in ['ember','pdf']:
            return utils.predict(self.net, self.normal.transform(X))[:,1]
        else:
            return utils.predict(self.net, X)[:,1]

    def build_model(self):
        hidden = self.hidden
        layer_sizes = None
        layers = []
        p = 0.5
        if layer_sizes is None:layer_sizes=[2048,2048,1024,1024,512,512,256,256]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(torch.nn.Linear(self.n_features,ls))
            else:
                layers.append(torch.nn.Linear(layer_sizes[i-1],ls))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(ls))
            layers.append(torch.nn.Dropout(p))
        layers.append(torch.nn.Linear(ls,2))
        net = torch.nn.Sequential(*tuple(layers))
        return net

    def save(self, save_path, file_name='nn'):
        joblib.dump(self.normal, os.path.join(save_path, file_name + '_scaler.pkl'))
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        self.normal = joblib.load(os.path.join(save_path, file_name + '_scaler.pkl'))
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl'),weights_only=True))

