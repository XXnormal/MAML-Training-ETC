import numpy as np
import os

DEFENCE = {
    'RegulaTor': ['RegulaTor_light_1', 'RegulaTor_light_3', 'RegulaTor_light_2'],
    'front': ['front_default', 'front_t2', 'front_t1'],
    'WTFPAD': ['WTFPAD_normal_0', 'WTFPAD_normal_2', 'WTFPAD_normal_1'],
    'TrafficSilver': ['TrafficSilver_WR', 'TrafficSilver_BD', 'TrafficSilver_BWR'],
}
DRIFT_DOHBRW_ENV = ['cn2cn', 'cn2kr', 'cn2us', 'exp']
DRIFT_ANDROID_VERSION = ['7', '8', '9', '10']

def getX(path, model_name, is_Tor):
    X = np.load(path).astype(np.float32)
    if model_name in ['DF', 'DFTF']:
        if is_Tor:
            X = np.sign(X)
        X = X[:, :5000].astype(np.float32)
        X = np.pad(X, ((0, 0), (0, 5000 - X.shape[1])), 'constant')
    else:
        X = X.astype(np.int64)
        if is_Tor:
            X = np.sign(X)
            X[X == -1] = 2
        else:
            X = np.abs(X)
            X[X > 1999] = 1999
    return X

def LoadDataGeneral(pre_dataset_dir, train_doms, test_doms, model_name, is_Tor, max_samples=10000000, nb_labels=-1, AAA=False):
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    for def_name in train_doms:
        X = getX(os.path.join(pre_dataset_dir, f'X_{def_name}.npy'), model_name, is_Tor)
        y = np.load(os.path.join(pre_dataset_dir, f'y_{def_name}.npy')).astype(np.int64)
        if nb_labels == -1:
            nb_labels = np.max(y) + 1
        else:
            X = X[y < nb_labels]
            y = y[y < nb_labels]
        nb_inst = X.shape[0]
        nb_train = min(int(nb_inst * 0.9), max_samples)
        nb_valid = min(int(nb_inst * 0.1), max_samples // 2)
        nb_per_cls = nb_train // nb_labels + 1
        nb_per_cls_valid = nb_valid // nb_labels + 1
        X_tmp, X_tmp_valid = [], []
        y_tmp, y_tmp_valid = [], []
        for i in range(nb_labels):
            Xi = X[y == i]
            yi = y[y == i]
            indices = np.random.permutation(yi.shape[0])
            X_tmp.append(Xi[indices[:nb_per_cls]])
            y_tmp.append(yi[indices[:nb_per_cls]])
            X_tmp_valid.append(Xi[indices[-nb_per_cls_valid:]])
            y_tmp_valid.append(yi[indices[-nb_per_cls_valid:]])
        X_train.append(np.concatenate(X_tmp, axis=0))
        y_train.append(np.concatenate(y_tmp, axis=0))
        X_valid.append(np.concatenate(X_tmp_valid, axis=0))
        y_valid.append(np.concatenate(y_tmp_valid, axis=0))
    
    for def_name in test_doms:
        X = getX(os.path.join(pre_dataset_dir, f'X_{def_name}.npy'), model_name, is_Tor)
        y = np.load(os.path.join(pre_dataset_dir, f'y_{def_name}.npy')).astype(np.int64)
        if nb_labels > 0:
            X = X[y < nb_labels]
            y = y[y < nb_labels]
        nb_inst = X.shape[0]
        indices = np.random.permutation(nb_inst)
        X_test.append(X[indices])
        y_test.append(y[indices])

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LoadDataInDomains(train_doms, test_doms, model_name, AAA=False):
    defence = DEFENCE
    if train_doms is not None:
        for train_dom in train_doms:
            assert train_dom in defence
    else:
        train_doms = [x for x in defence.keys() if x not in test_doms]
    
    new_train_doms = []
    new_test_doms = []
    for dom in train_doms:
        new_train_doms += defence[dom][:2]
        new_test_doms += defence[dom][2:]
    for dom in test_doms:
        new_test_doms += defence[dom]

    train_doms = new_train_doms
    test_doms = new_test_doms
    train_doms += ['NoDef']
    # Point to the directory storing data
    pre_dataset_dir = './DeepFingerprinting/Defence'
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataGeneral(
        pre_dataset_dir, train_doms, test_doms, model_name, True, max_samples=1000, AAA=AAA)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_doms, test_doms

def LoadDataDoHBrwNetEnv(train_doms, test_doms, model_name, AAA=False):
    # Point to the directory storing data
    pre_dataset_dir = f'./DoHBrwNetEnv'
    defence = DRIFT_DOHBRW_ENV
    
    for test_dom in test_doms:
        assert test_dom in defence
        if train_doms is not None:
            assert test_dom not in train_doms
    
    if train_doms is not None:
        for train_dom in train_doms:
            assert train_dom in defence
    else:
        train_doms = [x for x in defence if x not in test_doms]
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataGeneral(
        pre_dataset_dir, train_doms, test_doms, model_name, False, AAA=AAA)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_doms, test_doms

def LoadDataAndroidVersion(train_doms, test_doms, model_name, is_Tor=False, AAA=False):

    # Point to the directory storing data
    pre_dataset_dir = f'./NUDT_MobileTraffic/Android_Drift'
    defence = DRIFT_ANDROID_VERSION

    for test_dom in test_doms:
        assert test_dom in defence
        if train_doms is not None:
            assert test_dom not in train_doms
    
    if train_doms is not None:
        for train_dom in train_doms:
            assert train_dom in defence
    else:
        train_doms = [x for x in defence if x not in test_doms]
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataGeneral(
        pre_dataset_dir, train_doms, test_doms, model_name, is_Tor, max_samples=1000, nb_labels=4, AAA=AAA)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, train_doms, test_doms
