import os
import joblib
import torch
import torch.nn.functional as F

from core import utils
from sklearn.preprocessing import StandardScaler
from logger import logger
import numpy as np
import torch.nn as nn

from core.moe.moe import MOENet
from core.moe.moe_o import MOENet_o
from core.moe.moe_wg import MOENet_wg

class NN(object):
    def __init__(self, n_features, data_id, hidden=1024, moe=False):
        self.n_features = n_features
        self.normal = StandardScaler()
        self.hidden = hidden
        self.data_id = data_id
        self.moe = moe
        self.lr = 0.001
        if moe and 'moe' in moe:
            self.net = self.build_moe(moe)
        else:
            self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.exp = None
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9,0.999))
        if moe == 'moe':
            #MoE prefers smaller learning rate and stronger regularization (better for fitting Gaussian curves).
            self.lr = 3e-4
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999),weight_decay=1e-3)
        else:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999),weight_decay=1e-3)

    def fit(self, X, y, x_test, y_test, epoch, method):
        self.net.train()
        batch_size = 2048 
        logger.debug("batch size:",batch_size)
        if self.data_id in ['ember','pdf']:
            logger.debug("It's EMBER data")
            self.normal.fit(X)
            utils.train(self.normal.transform(X), y, self.normal.transform(x_test), y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        else:
            utils.train(X, y, x_test, y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        self.net.eval()

    def predict(self, X):
        if self.moe:
            return self.moepredict(X)
        if self.data_id in ['ember','pdf']:
            return utils.predict(self.net, self.normal.transform(X))[:,1]
        else:
            return utils.predict(self.net, X)[:,1]

    def moepredict(self, X):
        #enforce a difference for the output probs
        net = self.net
        device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            if isinstance(net, torch.nn.Module):
                net.eval()
                y_hat = []
                for X_batch in utils.data_iter(2048,X):
                    X_batch = X_batch.to(device)
                    y_hat.append(net(X_batch))
                net.train()
        logits = torch.cat(y_hat, dim=0)
        probs = F.softmax(logits, dim=1)
        bad_rows = torch.abs(probs[:, 0] - 0.5) < 1e-6

        if bad_rows.any():
            bad_logits = logits[bad_rows]
            max_idx = bad_logits.argmax(dim=1)
            probs[bad_rows, max_idx] += 1e-05 
            probs[bad_rows, 1 - max_idx] -= 1e-05
        return probs.cpu()[:,1]

    def build_model(self):
        hidden = self.hidden
        layer_sizes = None
        layers = []
        p = 0.5
        if layer_sizes is None:layer_sizes=[hidden,512,256]
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

    def build_moe(self,moe):
        n_experts = 16
        if moe == "moe":#G-MoE
            nn = MOENet(input_dim=self.n_features, hidden=self.hidden, num_experts=n_experts, output_dim=2)
        elif moe == "moe_o":#vanilla MoE feeding features instead of time
            nn = MOENet_o(input_dim=self.n_features, hidden=self.hidden, num_experts=n_experts, output_dim=2)
        elif moe == "moe_wg": #replacing the gaussian gate with FNN
            nn = MOENet_wg(input_dim=self.n_features, hidden=self.hidden, num_experts=n_experts, output_dim=2)
        return nn
    
    def save(self, save_path, file_name='nn'):
        # Save the trained scaler so that it can be reused at test time
        joblib.dump(self.normal, os.path.join(save_path, file_name + '_scaler.pkl'))
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        self.normal = joblib.load(os.path.join(save_path, file_name + '_scaler.pkl'))
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl'),weights_only=True))

