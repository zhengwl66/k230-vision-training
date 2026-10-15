import math
import numpy as np

def target(box,class_id):
    return dict(x=int(math.floor(box[0])),y=int(math.floor(box[1])),w=int(math.ceil(box[2]))-int(math.floor(box[0])),h=int(math.ceil(box[3]))-int(math.floor(box[1])),class_id=class_id)

def kernel(a,b,gamma):
    d=np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)
    return np.exp(-gamma*d)

def fit_rbf(features,labels,basis,gamma,regularization):
    a=kernel(features,basis,gamma);kb=kernel(basis,basis,gamma)
    return np.linalg.solve(a.T@a+regularization*kb+1e-5*np.eye(len(basis)),a.T@labels)
