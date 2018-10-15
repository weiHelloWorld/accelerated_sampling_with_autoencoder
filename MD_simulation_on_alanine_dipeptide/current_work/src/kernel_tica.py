import numpy as np
from msmbuilder.decomposition.tica import tICA
from sklearn.kernel_approximation import Nystroem

"""modified from https://github.com/msmbuilder/msmbuilder/blob/master/msmbuilder/decomposition/ktica.py"""
"""reference: [1] Schwantes, Christian R., and Vijay S. Pande. J. Chem Theory Comput. 11.2 (2015): 600--608."""

class Kernel_tica(object):
    def __init__(self, n_components, lag_time,
                 gamma,             # gamma value for rbf kernel
                 n_components_nystroem=100,  # number of components for Nystroem kernel approximation
                 shrinkage = 0.1
                 ):
        self._n_components = n_components
        self._lag_time = lag_time
        self._n_components_nystroem = n_components_nystroem
        self._gamma = gamma
        self._nystroem = Nystroem(gamma=gamma, n_components=n_components_nystroem)
        self._tica = tICA(n_components=n_components, lag_time=lag_time, shrinkage=shrinkage)
        return

    def fit(self, sequence):
        sequence_transformed = self._nystroem.fit_transform(sequence)
        self._tica.fit([sequence_transformed])
        return

    def transform(self, sequence):
        return self._tica.transform(
            [self._nystroem.transform(sequence)])

    def fit_transform(self, sequence):
        self.fit(sequence)
        return self.transform(sequence)
