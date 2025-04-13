import copy
import torch
import torch.nn as nn

from base_models import EmbbedingNet
from utility import *
from itertools import combinations
from collections import OrderedDict

class MASF(nn.Module):
    def __init__(self, args, model_class, model_args, embedding_args, nb_doms):
        super(MASF, self).__init__()
        
        self.model_class = model_class
        self.model_args = model_args
        self.embedding_args = embedding_args
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.embedding_lr = args.embedding_lr
        self.batch_size = args.batch_size
        self.global_weight = args.global_weight
        self.local_weight = args.local_weight
        self.clip_value = args.clip_value
        self.beta1 = args.global_weight
        self.beta2 = args.local_weight
        self.nb_classes = args.nb_classes
        self.nb_doms = nb_doms
        self.combination = list(combinations(range(nb_doms), 1))
        self.device = torch.device("cpu") if args.cuda=='cpu' else torch.device("cuda:{}".format(args.cuda))

        self.net = self.model_class(*self.model_args)
        self.embedding_net = EmbbedingNet(*self.embedding_args)
        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=self.update_lr)
        self.embedding_optim = torch.optim.Adam(self.embedding_net.parameters(), lr=self.embedding_lr)
        self.loss_func = nn.CrossEntropyLoss()

        self.IMP = args.IMP
        if self.IMP is True:
            self.log_sigma = nn.Parameter(torch.log(torch.FloatTensor([args.imp_param])), requires_grad=True)
            self.optim_log_sigma = torch.optim.RMSprop([self.log_sigma], lr=self.meta_lr)
        self.ATS = args.ATS
        if self.ATS is True:
            self.ats = args.ats_param
        self.AUG = args.AUG
        if self.AUG is True:
            self.aug_sigma = args.aug_param
            self.stds = [[[torch.zeros(self.net.feat_dim).to(self.device)] * self.nb_classes] * self.nb_doms] * len(self.combination)
            self.std_discount = 0.2

    def get_net_parameters(self):
        assert isinstance(self.net, nn.Module)
        params = OrderedDict()
        for name, param in self.net.named_parameters():
            params[name] = param
        return params
    
    def save_weights(self, weights=None):
        fast_weights = OrderedDict()
        model_fast_weights = self.get_net_parameters()  #  model parameter 
        for name in model_fast_weights:
            if weights is not None:
                fast_weights[name] = model_fast_weights[name].data - weights[name]  # save the delta of parameters
            else:
                fast_weights[name] = model_fast_weights[name].data
        
        with torch.cuda.device(self.device):
            return copy.deepcopy(fast_weights)
    
    def load_weights(self, fast_weights, weights=None):
        model_fast_weights = self.get_net_parameters()
        for name in model_fast_weights:
            if weights is not None:
                model_fast_weights[name].data.copy_(fast_weights[name] + weights[name])
            else:
                model_fast_weights[name].data.copy_(fast_weights[name])

    def load_weights_ats(self, fast_weights, diff_weights, w):
        model_fast_weights = self.get_net_parameters()
        for name in model_fast_weights:
            model_fast_weights[name].data.copy_(fast_weights[name] + diff_weights[name] * w)

    def forward(self, X_ori, y_ori, idx_batch=None):
        
        task_num, setsz, d = X_ori.size()

        losses_comb = []

        meta_test_diff_list = []

        base_weights = self.save_weights()

        losses_comb_last = None

        for idx_comb, test_indices in enumerate(self.combination):
            loss_task = []

            train_indices = list(set(range(task_num)) - set(test_indices))
                        
            X_meta_tr, y_meta_tr = [], []
            X_meta_te, y_meta_te = [], []

            for i in train_indices:
                X_meta_tr.append(X_ori[i])
                y_meta_tr.append(y_ori[i])

            for i in test_indices:
                X_meta_te.append(X_ori[i])
                y_meta_te.append(y_ori[i])

            for idx_tr, (inputs_tr, y_tr) in enumerate(zip(X_meta_tr, y_meta_tr)):
                feats = self.net.featureExtract(inputs_tr)
                if self.AUG is True:
                    nb_sample, d = feats.size()
                    assert d == self.net.feat_dim
                    alpha = (torch.randn(nb_sample, d) * self.aug_sigma + torch.ones(nb_sample, d)).to(self.device)
                    beta = torch.randn(nb_sample, d).to(self.device) * self.aug_sigma
                    feats = alpha * feats + beta
                    feats_cls = [[] for _ in range(self.nb_classes)]
                    y_cls = [[] for _ in range(self.nb_classes)]
                    for feat, label in zip(feats, y_tr):
                        feats_cls[label].append(feat)
                        y_cls[label].append(label)
                    feats_aft = []
                    y_aft = []
                    for c in range(self.nb_classes):
                        if len(feats_cls[c]) == 0:
                            continue
                        feats_c = torch.stack(feats_cls[c], dim=0)
                        y_c = torch.stack(y_cls[c], dim=0)
                        if feats_c.size(0) == 1:
                            feats_aft.append(feats_c)
                            y_aft.append(y_c)
                            continue
                        std_c = torch.std(feats_c, dim=0).detach()
                        self.stds[idx_comb][idx_tr][c] *= self.std_discount
                        self.stds[idx_comb][idx_tr][c] += (1 - self.std_discount) * std_c
                        noise = torch.randn(feats_c.size(0), d).to(self.device) * self.stds[idx_comb][idx_tr][c]
                        feats_aft.append(feats_c + noise)
                        y_aft.append(y_c)
                    feats = torch.cat(feats_aft, dim=0)
                    y_tr = torch.cat(y_aft, dim=0)
                logits = self.net.classify(feats)
                loss_task.append(F.cross_entropy(logits, y_tr))

            loss_task = torch.mean(torch.stack(loss_task))
            self.net_optim.zero_grad()
            loss_task.backward()
            self.net_optim.step()

            meta_trained_weights = self.save_weights()

            if self.AUG is True and idx_batch == 0:
                self.stds[idx_comb] = [[torch.ones(self.net.feat_dim).to(self.device)] * self.nb_classes] * self.nb_doms

            Z_meta_tr, Z_meta_te = [], []
            for idx_tr, (inputs_tr, y_tr) in enumerate(zip(X_meta_tr, y_meta_tr)):
                feats = self.net.featureExtract(inputs_tr)
                Z_meta_tr.append(feats)

            nb_doms_tr = len(X_meta_tr)
            for idx_te, (inputs_te, y_te) in enumerate(zip(X_meta_te, y_meta_te)):
                feats = self.net.featureExtract(inputs_te)
                Z_meta_te.append(feats)
            
            if self.IMP is not True:
                sc_train_list, sc_test_list = [], []
                for feats, labels in zip(Z_meta_tr, y_meta_tr):
                    sc_train = {}
                    feats_cls = [[] for _ in range(self.nb_classes)]
                    for feat,label in zip(feats, labels):
                        feats_cls[label].append(feat)
                    for c in range(self.nb_classes):
                        if len(feats_cls[c]) == 0:
                            continue
                        feats_c = torch.stack(feats_cls[c], dim=0)
                        feats_c = feats_c.mean(dim=0).unsqueeze(0)
                        sc_train[c] = F.softmax(self.net.classify(feats_c) / 2.0, dim=1)[0]
                    sc_train_list.append(sc_train)
                
                for feats, labels in zip(Z_meta_te, y_meta_te):
                    sc_test = {}
                    feats_cls = [[] for _ in range(self.nb_classes)]
                    for feat,label in zip(feats, labels):
                        feats_cls[label].append(feat)
                    for c in range(self.nb_classes):
                        if len(feats_cls[c]) == 0:
                            continue
                        feats_c = torch.stack(feats_cls[c], dim=0)
                        feats_c = feats_c.mean(dim=0).unsqueeze(0)
                        sc_test[c] = F.softmax(self.net.classify(feats_c) / 2.0, dim=1)[0]
                    sc_test_list.append(sc_test)

                loss_global = []
                for sc_train, y_tr in zip(sc_train_list, y_meta_tr):
                    for sc_test, y_te in zip(sc_test_list, y_meta_te):
                        loss_global_c = []
                        for c in range(self.nb_classes):
                            if c not in sc_train or c not in sc_test:
                                continue
                            KL_1 = torch.sum(sc_train[c] * torch.log(sc_train[c] / sc_test[c]))
                            KL_2 = torch.sum(sc_test[c] * torch.log(sc_test[c] / sc_train[c]))
                            loss_global_c.append((KL_1 + KL_2) / 2)
                        if len(loss_global_c) == 0:
                            continue
                        loss_global.append(torch.mean(torch.stack(loss_global_c)))

                if len(loss_global) == 0:
                    loss_global = 0
                else:
                    loss_global = torch.mean(torch.stack(loss_global))

                Z = torch.cat(Z_meta_tr + Z_meta_te, dim=0)
                y = torch.cat(y_meta_tr + y_meta_te, dim=0)

                embeddings = self.embedding_net(Z)
                
                a = None
                a, p, n = tripletMining(embeddings, y.view(-1))
                if a is not None:
                    loss_local = F.triplet_margin_loss(embeddings[a], embeddings[p], embeddings[n], margin=1.0)
                else:
                    loss_local = 0
                
                loss_meta = self.global_weight * loss_global + self.local_weight * loss_local
            
            else:

                mu_set_train_list, proto_set_train_list = [], []
                mu_set_test_list, proto_set_test_list = [], []

                for feats, labels in zip(Z_meta_tr, y_meta_tr):
                    mu_set_train, proto_set_train = IMP(feats, labels, self.nb_classes, self.log_sigma)
                    mu_set_train_list.append(self.net.classify(mu_set_train))
                    proto_set_train_list.append(proto_set_train)
                
                for feats, labels in zip(Z_meta_te, y_meta_te):
                    mu_set_test, proto_set_test = IMP(feats, labels, self.nb_classes, self.log_sigma)
                    mu_set_test_list.append(self.net.classify(mu_set_test))
                    proto_set_test_list.append(proto_set_test)
                
                loss_global = []
                loss_local = []
                for mu_set_train, proto_set_train, y_tr in zip(mu_set_train_list, proto_set_train_list, y_meta_tr):
                    for mu_set_test, proto_set_test, y_te in zip(mu_set_test_list, proto_set_test_list, y_meta_te):
                        loss_global_c = []
                        loss_local_c = []
                        for c in range(self.nb_classes):
                            if c not in proto_set_train or c not in proto_set_test:
                                continue
                            mu_set_c_train = mu_set_train[proto_set_train == c]
                            mu_set_c_test = mu_set_test[proto_set_test == c]
                            mu_set_c_train = mu_set_c_train.unsqueeze(1) 
                            mu_set_c_test = mu_set_c_test.unsqueeze(0)

                            KL_1 = torch.sum(mu_set_c_train * torch.log(mu_set_c_train / mu_set_c_test), dim=-1).mean()
                            KL_2 = torch.sum(mu_set_c_test * torch.log(mu_set_c_test / mu_set_c_train), dim=-1).mean()
                            loss_global_c.append((KL_1 + KL_2) / 2)

                            log_dis_local = torch.sum(torch.log(torch.norm(mu_set_c_train - mu_set_c_test, dim=-1) + 1e-18)).mean()
                            loss_local_c.append(log_dis_local)

                        if len(loss_global_c) > 0:
                            loss_global.append(torch.mean(torch.stack(loss_global_c)))
                        if len(loss_local) > 0:
                            loss_local.append(torch.mean(torch.stack(loss_local)))
                
                if len(loss_global) == 0:
                    loss_global = 0
                else:
                    loss_global = torch.mean(torch.stack(loss_global))

                if len(loss_local) == 0:
                    loss_local = 0
                else:
                    loss_local = torch.mean(torch.stack(loss_local))

                loss_meta = self.global_weight * loss_global + self.local_weight * loss_local
            
            losses_comb.append(loss_meta)
            
            self.net_optim.zero_grad()
            self.embedding_optim.zero_grad()
            loss_meta.backward()
            self.net_optim.step()
            self.embedding_optim.step()
            if self.IMP is True:
                self.optim_log_sigma.step()

            meta_test_diff = self.save_weights(weights=meta_trained_weights)

            if self.ATS is not True:
                self.load_weights(base_weights, weights=meta_test_diff)
            else:
                self.load_weights(base_weights)
                meta_test_diff_list.append(meta_test_diff)

        if self.ATS is not True:
            return 
        
        losses_comb = torch.stack(losses_comb).detach()

        if losses_comb_last is not None:
            loss_mean = torch.mean(losses_comb)
            loss_mean_last = torch.mean(losses_comb_last)
            loss_ats = torch.arctan(loss_mean_last - loss_mean)
            self.ats *= (1 - loss_ats * 0.01)

        ws = F.softmax(self.ats * losses_comb, dim=0) * len(self.combination)

        last_base_weights = base_weights
        for w, meta_test_diff in zip(ws, meta_test_diff_list):    
            self.load_weights_ats(last_base_weights, meta_test_diff, w)
            last_base_weights = self.save_weights()

        losses_comb_last = losses_comb

    def predict(self, X):
        return self.net(X)
    
    def featureExtract(self, X):
        return self.net.featureExtract(X)
    
    def train(self):
        self.net.train()
        self.embedding_net.train()

    def eval(self):
        self.net.eval()
        self.embedding_net.eval()
    