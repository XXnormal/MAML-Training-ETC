import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification
from tqdm import tqdm
from utility import getDataLoader

from dataset import *
from masf import MASF
from base_models import DFNet, DFTransformer, TransformerClassifier
import logging

def main(args):
    
    # 1. Load data

    train_types = args.train_types.split(',') if args.train_types is not None else None
    test_types = args.test_types.split(',') if args.test_types is not None else None
    
    if args.dataset == 'Tor-Obfuscation':
        X_tr_doms, y_tr_doms, X_va_doms, y_va_doms, X_te_doms, y_te_doms, train_doms, test_doms = LoadDataInDomains(train_types, test_types, args.model)
    elif args.dataset == 'Android-Device':
        X_tr_doms, y_tr_doms, X_va_doms, y_va_doms, X_te_doms, y_te_doms, train_doms, test_doms = LoadDataAndroidVersion(train_types, test_types, args.model)
    elif args.dataset == 'DoHBrw-NetEnv':
        X_tr_doms, y_tr_doms, X_va_doms, y_va_doms, X_te_doms, y_te_doms, train_doms, test_doms = LoadDataDoHBrwNetEnv(train_types, test_types, args.model)
    else:
        raise ValueError("Dataset not found")
    
    assert isinstance(train_doms, list)
    assert isinstance(test_doms, list)
    
    X_train = X_tr_doms
    y_train = y_tr_doms
    X_valid = X_va_doms
    y_valid = y_va_doms
    X_test = X_te_doms
    y_test = y_te_doms

    # 2. Create model
    if args.model == 'DF':
        model_name = DFNet
        model_args = (5000, args.nb_classes)
        embedding_args = (128, 64, 32)
    elif args.model == 'ARES':
        model_name = DFTransformer
        model_args = (5000, args.nb_classes)
        embedding_args = (256, 64, 32)
    else:
        raise ValueError("Model not found")

    device = torch.device("cpu") if args.cuda=='cpu' else torch.device("cuda:{}".format(args.cuda))
    nb_doms = len(train_doms)
    masf_model = MASF(args, model_name, model_args, embedding_args, nb_doms)

    log_path = os.path.join(args.log_dir, args.log_name+'.log')
    improved = ''
    if args.IMP:
        improved += '_IMP'
    if args.ATS:
        improved += '_ATS'
    if args.AUG:
        improved += '_AUG'

    if args.continue_training:
        print('Loading dict...')
        model_dict = torch.load(os.path.join(args.model_dir, 'model_{}{}.pt'.format(args.log_name, improved)), map_location=device)
        masf_model.load_state_dict(model_dict)
        del model_dict
    
    masf_model = masf_model.to(device)

    logger = logging.getLogger(args.log_name)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info("Train Def: {}, Test Def: {}".format(train_doms, test_doms))

    max_nb_samples = max([X.shape[0] for X in X_train])
    print("Dataset:", args.dataset, "Train X shape:", [X.shape for X in X_train])
    X_train = [np.concatenate(
        ([X] * (max_nb_samples // X.shape[0]) if max_nb_samples // X.shape[0] > 0 else []) + \
        ([X[:max_nb_samples % X.shape[0]]] if max_nb_samples % X.shape[0] > 0 else []), axis=0) for X in X_train]
    y_train = [np.concatenate(
        ([y] * (max_nb_samples // y.shape[0]) if max_nb_samples // y.shape[0] > 0 else []) + \
        ([y[:max_nb_samples % y.shape[0]]] if max_nb_samples % y.shape[0] > 0 else []), axis=0) for y in y_train]

    print('----------------------')

    f1 = torchmetrics.classification.MulticlassF1Score(args.nb_classes).to(device)
    acc = torchmetrics.classification.MulticlassAccuracy(args.nb_classes).to(device)
    # 3. Train model
    
    if args.dataset == 'TorDate':
        epochs = args.epochs // len(train_doms)
        test_epoch = min(epochs + 1, args.test_epoch)
    else:
        epochs = args.epochs
        test_epoch = args.test_epoch

    for epoch in tqdm(range(epochs)):
        train_loaders = tuple([getDataLoader(X, y, batch_size=args.batch_size, shuffle=True) for X, y in zip(X_train, y_train)])
        masf_model.train()
        for i, batches in enumerate(zip(*train_loaders)):
            X_batch, y_batch = [], []
            nb_samles = 0
            for X, y in batches:
                X_batch.append(X)
                y_batch.append(y)
                nb_samles += X.shape[0]
            if nb_samles < args.batch_size * 1.5:
                continue
            X_batch = torch.stack(X_batch, dim=0).to(device)
            y_batch = torch.stack(y_batch, dim=0).to(device)
            masf_model(X_batch, y_batch, idx_batch=i)

        if (epoch+1) % test_epoch == 0:
            logger.info("Epoch: {} Finished".format(epoch+1))
            masf_model.eval()
            torch.save(masf_model.state_dict(), os.path.join(args.model_dir, 'model_{}{}.pt'.format(args.log_name, improved)))
            with torch.no_grad():
                for X, y, dom in zip(X_valid, y_valid, train_doms):
                    valid_loader = getDataLoader(X, y, args.batch_size)
                    preds, ys = [], []
                    for X_batch, y_batch in valid_loader:
                        logits = masf_model.predict(X_batch.to(device))
                        preds.append(logits)
                        ys.append(y_batch.to(device))
                    preds = torch.cat(preds, dim=0)
                    ys = torch.cat(ys, dim=0)
                    acc_valid = acc(preds, ys)
                    f1_valid = f1(preds, ys)
                    logger.info("Epoch: {}, Valid Acc: {}, Valid F1: {}, Domain: {}".format(epoch+1, acc_valid, f1_valid, dom))
            
            with torch.no_grad():
                for X, y, dom in zip(X_test, y_test, test_doms):
                    test_loader = getDataLoader(X, y, args.batch_size)
                    preds, ys = [], []
                    for X_batch, y_batch in test_loader:
                        logits = masf_model.predict(X_batch.to(device))
                        preds.append(logits)
                        ys.append(y_batch.to(device))
                    preds = torch.cat(preds, dim=0)
                    ys = torch.cat(ys, dim=0)
                    acc_valid = acc(preds, ys)
                    f1_valid = f1(preds, ys)
                    logger.info("Epoch: {}, Test Acc: {}, Test F1: {}, Domain: {}".format(epoch+1, acc_valid, f1_valid, dom))
            
            masf_model.train()
    
    masf_model.eval()
    torch.save(masf_model.state_dict(), os.path.join(args.model_dir, 'model_{}{}.pt'.format(args.log_name, improved)))
    
    del masf_model
    torch.cuda.empty_cache()
    logger.removeHandler(fh)
    logger.removeHandler(ch)
    fh.close()
    ch.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--update_lr', type=float, help='update learning rate', default=0.001)
    argparser.add_argument('--meta_lr', type=float, help='meta learning rate', default=0.0002)
    argparser.add_argument('--embedding_lr', type=float, help='embedding learning rate', default=0.0001)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=256)
    argparser.add_argument('--global_weight', type=float, help='global weight', default=1)
    argparser.add_argument('--local_weight', type=float, help='local weight', default=0.005)
    argparser.add_argument('--epochs', type=int, help='epochs', default=100)
    argparser.add_argument('--nb_classes', type=int, help='number of classes', default=95)
    argparser.add_argument('--cuda', type=str, help='cuda', default=0)
    argparser.add_argument('--dataset', type=str, help='dataset', default='DF')
    argparser.add_argument('--log_dir', type=str, help='log directory', default='logs/MAML')
    argparser.add_argument('--log_name', type=str, help='log name', default='DF_masf_default.log')
    argparser.add_argument('--train_types', type=str, help='train types', default=None)
    argparser.add_argument('--test_types', type=str, help='test types', default=None)
    argparser.add_argument('--clip_value', type=float, help='clip value', default=2.0)
    argparser.add_argument('--test_epoch', type=int, help='test epoch', default=20)
    argparser.add_argument('--model', type=str, help='model', default='DF')
    argparser.add_argument('--model_dir', type=str, help='model directory', default='models')

    argparser.add_argument('--ATS', action='store_true', help='ATS', default=False)
    argparser.add_argument('--IMP', action='store_true', help='IMP', default=False)
    argparser.add_argument('--AUG', action='store_true', help='AUG', default=False)
    
    argparser.add_argument('--continue_training', action='store_true', help='new model', default=False)
    args = argparser.parse_args()

    main(args)
