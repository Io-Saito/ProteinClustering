"""
title: ProteinClustering
description: Network enhancement for weighted non-directional graph
author: Io V Saito
reference : 
    Wang, B., Pourshafeie, A., Zitnik, M. et al. 
    Network enhancement as a general method to denoise weighted biological networks. 
    Nat Commun 9, 3108 (2018).
    https://doi.org/10.1038/s41467-018-05469-x
"""

import numpy as np
import pandas as pd


class NetWorkEnhancement():

    def __init__(self, df,eps: float = 2e-16, k=None, alpha=0.9, diffusion=2):
        self.W = None
        self.diffusion = diffusion
        self.alpha = alpha
        self.eps = eps
        self.k = k
        self.cor = df.to_numpy(dtype=np.float64)
        self.node = df.columns
        if self.k == None:
            self.k = min(20, np.ceil(self.cor.size)/10)
        self.num = self.node.size
        return None

    def Enhance(self):
        print(f"Generating Network of {self.num} proteins...")
        W_ = self.cor*(1-np.eye(self.num))
        zeorindex = np.where((np.abs(W_).sum(axis=1) > 0))
        self.W = self.NE_DN(W_[zeorindex])
        self.W = (self.W+self.W.T)/2
        print("Building dominantset...")
        if np.unique(self.W).size == 2:
            self.P = self.W
        else:
            self.P = self.dominantset(np.abs(self.W), min(
                self.k, self.num))*np.sign(self.W)
        self.P = self.P + np.eye(self.P.shape[0]) + \
            np.diagflat(np.abs(self.P).sum(axis=0))
        print("Building TransitionField...")
        self.P = self.TransitionField(self.P)
        print("Eigenvalue Decomposing...")
        U, V = np.linalg.eig(self.P)
        d = U-self.eps
        d = (1-self.alpha)*d/(1-self.alpha*d**self.diffusion)
        D = np.diagflat(d)
        print("Calculating weight...")
        self.CalculatingWeight(D, V)
        print("Almost done...")
        self.W = self.W*(1-np.eye(self.num))/(1-np.diag(self.W))
        D = np.diagflat(np.sum(np.abs(W_[zeorindex]*self.num), axis=0))
        self.W = D@self.W
        self.W[np.where(self.W < 0)] = 0
        self.W = (self.W+self.W.T)/2
        return pd.DataFrame(self.W.astype(np.float), columns=self.node, index=self.node)

    def CalculatingWeight(self, D, V):
        self.W = (V@D)@V.T

    def dominantset(self, aff_matrix: np.array, NR_OF_N: int) -> np.array:
        A = np.sort(aff_matrix, axis=0)[::-1].T
        B = np.argsort(aff_matrix, axis=0)[::-1].T
        res = A[:, 0:NR_OF_N]
        inds = np.tile(np.arange(0, aff_matrix.shape[0]), (1, NR_OF_N))
        loc = B[:, 0: NR_OF_N]
        idx_ = loc.flatten("F")*aff_matrix.shape[0]+(inds.flatten("F"))
        PNN_array = np.zeros(shape=aff_matrix.size)
        PNN_array[idx_.astype(np.int)] = res.flatten("F")
        PNN_matrix1 = PNN_array.reshape(aff_matrix.shape)
        PNN_matrix = (PNN_matrix1+PNN_matrix1.T)/2
        return PNN_matrix

    def NE_DN(self, w: np.ndarray, type="avg") -> np.array:
        w = w * w.shape[0]
        D = np.sum(np.abs(w), axis=0)+self.eps
        if type == "ave" or type == "avg":
            D = 1/D
            D = np.diagflat(D)
            wn = D@w
        elif type == "gph":
            D = 1/np.sqrt(D)
            D = np.diagflat(D.shape[0])
            wn = D@w@D
        else:
            raise ValueError("Invalid type")
        return wn

    def TransitionField(self, W: np.array) -> np.array:
        zeroindex = np.argwhere(W.sum(axis=1) == 0)
        W = self.NE_DN(W*W.shape[0])
        W = W/np.sqrt(np.abs(W).sum(axis=1)+self.eps)
        W = W@W.T
        W[zeroindex, :] = 0
        W[:, zeroindex] = 0
        return W
