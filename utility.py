import torch 
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).sum().item()
        acc = correct / len(target)
        return acc

def getDataLoader(X, y=None, batch_size=128, shuffle=False):
    if y is None:
        try:
            loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=shuffle)
        except:
            loader = torch.utils.data.DataLoader(torch.from_numpy(X), batch_size=batch_size, shuffle=shuffle)
        return loader
    else:
        try:
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), 
                                                 batch_size=batch_size, shuffle=shuffle)
        except:
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), 
                                                 batch_size=batch_size, shuffle=shuffle)
        return loader

def optimize(loss, optimizers, loss_list=None):
    if loss_list is not None:
        loss_list.append(loss.item())
    for optimizer in optimizers:
        optimizer.zero_grad()
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()

def getSavePath(str):
    return os.path.join('/data/', str)

def EGDis2(E, G, X, a, b, i, j):
    dis = E(G(X[i][a])) - E(G(X[j][b]))
    print(torch.norm(dis))
    return torch.norm(E(G(X[i][a])) - E(G(X[j][b])), dim=1) ** 2

def save_model(model, device, name):
    model.to('cpu')
    torch.save(model.state_dict(), os.path.join(getSavePath('models'), name))
    model.to(device)
    
def load_model(model, device, name):
    model.load_state_dict(torch.load(os.path.join(getSavePath('models'), name)))
    model.to(device)

def pairMining(Z, y, margin):
    pair_indices = []
    for i in range(Z.size(0)):
        anchor = i
        positive_indices = np.where(y == y[i])[0]
        negative_indices = np.where(y != y[i])[0]
        ap_distances = torch.norm(Z[positive_indices] - anchor, dim=1) ** 2
        an_distances = torch.norm(Z[negative_indices] - anchor, dim=1) ** 2
        positive = positive_indices[torch.argmax(ap_distances)]
        negative = negative_indices[torch.argmin(an_distances)]
        ap_dis = torch.max(ap_distances)
        an_dis = torch.min(an_distances)
        if ap_dis > margin - an_dis:
            pair_indices.append([anchor, positive])
        else:
            pair_indices.append([anchor, negative])
    return pair_indices

def tripletMining(Z, y):
    """Triplet Mining"""
    triplets_indices = []
    for i in range(Z.size(0)):
        anchor = i
        positive_indices = torch.where(y == y[anchor])[0]
        negative_indices = torch.where(y != y[anchor])[0]
        if positive_indices.size(0) <= 1 or negative_indices.size(0) == 0:
            continue
        ap_distances = torch.norm(Z[positive_indices] - anchor, dim=1) ** 2
        positive = positive_indices[torch.argmax(ap_distances)].cpu().numpy()
        an_distances = torch.norm(Z[negative_indices] - anchor, dim=1) ** 2
        negative = negative_indices[torch.argmin(an_distances)].cpu().numpy()
        triplets_indices.append([anchor, positive, negative])
    if len(triplets_indices) == 0:
        return None, None, None
    apn = np.array(triplets_indices)
    return apn[:, 0], apn[:, 1], apn[:, 2]

def IMP(feats: torch.Tensor, labels, nb_classes, log_sigma: nn.Parameter, alpha=0.1):

    # get Lambda
    d = feats.size(1)
    rho = torch.std(feats, dim=0).mean()
    sigma = torch.exp(log_sigma).data[0]
    Lambda = -2 * sigma * np.log(alpha) + d * sigma * torch.log(1 + rho / sigma)

    mu_set = []
    proto_set = []
    feats_cls = [[] for _ in range(nb_classes)]
    for feat, label in zip(feats, labels):
        feats_cls[label].append(feat)
    for c in range(nb_classes):
        if len(feats_cls[c]) == 0:
            continue
        feats_c = torch.stack(feats_cls[c], dim=0)
        mu_c = torch.mean(feats_c, dim=0)
        mu_set.append(mu_c)
        proto_set.append(c)
    
    if len(mu_set) == 0:
        return None, None
    
    mu_set = torch.stack(mu_set)

    nb_per_proto = [0] * len(proto_set)

    for feat, label in zip(feats, labels):
        dist_min = torch.inf
        the_proto = None
        for i, (mu, proto) in enumerate(zip(mu_set, proto_set)):
            if label == proto:
                dist = torch.norm(feat - mu) ** 2
                if dist < dist_min:
                    dist_min = dist
                    the_proto = i
            
        if the_proto is None or dist_min > Lambda:
            mu_set = torch.cat([mu_set, feat.unsqueeze(0)], dim=0)
            proto_set.append(label.item())
            nb_per_proto.append(1)
        else:
            nb_per_proto[the_proto] += 1

    nb_per_proto = np.array(nb_per_proto)
    mu_set = mu_set[nb_per_proto > 0]
    proto_set = np.array(proto_set)[nb_per_proto > 0]
    
    feats = feats.unsqueeze(1) # (i, 1, d)
    mu_set = mu_set.unsqueeze(0)
    sigma = sigma.view(1, 1)
    dim = feats.size(-1)
    neg_dist = -torch.sum((feats - mu_set) ** 2, dim=-1)

    logits = neg_dist / 2.0 / sigma 
    norm_constant = 0.5 * dim * (torch.log(sigma) + np.log(2 * np.pi))
    logits = logits - norm_constant # (i, c)
    z = logits / torch.logsumexp(logits, dim=1).unsqueeze(1) # (i, c)

    z = z.unsqueeze(2) # (i, c, 1)

    mu_new = torch.sum(z * feats, dim=0) / torch.sum(z, dim=0) # (c, d)

    return mu_new, proto_set
