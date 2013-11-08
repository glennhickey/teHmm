#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import string
from .teModel import TEModel
from .emissions import IndependentMultinomialEmissionModel

from sklearn.hmm import _BaseHMM
from sklearn.hmm import MultinomialHMM
from sklearn.hmm import GaussianHMM
from sklearn.utils import check_random_state, deprecated
from sklearn.utils.extmath import logsumexp
from sklearn.base import BaseEstimator
from sklearn.hmm import cluster
from sklearn.hmm import _hmmc
from sklearn.hmm import normalize
from sklearn.hmm import NEGINF

"""
This class is based on the MultinomialHMM from sckikit-learn, but we make
the emission model a parameter. The custom emission model we support at this
point is a multi-*dimensional* multinomial. 
"""
class TEHmm(_BaseHMM):
    def __init__(self, emissionModel, n_components=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):
        """Create a hidden Markov model with multinomial emissions.
        emissionModel must have already been created"""
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)
        self.emissionModel = emissionModel

    def _compute_log_likelihood(self, obs):
        return self.emissionModel.allLogProbs(obs)

    def _generate_sample_from_state(self, state, random_state=None):
        return self.emissionModel.sample(state)

    def _init(self, obs, params='ste'):
        super(TEHmm, self)._init(obs, params=params)
        self.emissionModel.initParams()

    def _initialize_sufficient_statistics(self):
        stats = super(TEHmm, self)._initialize_sufficient_statistics()
        stats['obs'] = self.emissionModel.initStats()
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(TEHmm, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            self.emissionModel.accumulateStats(obs, stats['obs'], posteriors)

    def _do_mstep(self, stats, params):
        super(TEHmm, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionModel.maximize(stats['obs'])

    def fit(self, obs, **kwargs):
        err_msg = ("Input must be both positive integer array and "
                   "every element must be continuous, but %s was given.")

#        if not self._check_input_symbols(obs):
#            raise ValueError(err_msg % obs)

        return _BaseHMM.fit(self, obs, **kwargs)

"""
HMM Model for a transposable element.  Gets trained by a set of tracks.
Prototype implementation using scikit-learn
"""
class TEHMMModel(TEModel):
    def __init__(self):
        super(TEHMMModel, self).__init__()
        #: scikit-learn hmm
        self.hmm = None
        #: map from self.data -> self.flatData
        self.flatMap = None
        #: 1-d version of dataList (unique entry for each vector)
        self.flatDataList = None
        #: reverse version of flatMap
        self.flatMapBack = None

    def loadTrackData(self, tracksInfoPath, seqName, start, end,
                      forTraining = False):
        super(TEHMMModel, self).loadTrackData(tracksInfoPath, seqName,
                                              start, end, forTraining)
        self.flattenObservations()

    def save(self, modelPath):
        flatDataBackup = self.flatDataList
        self.flatData = None
        super(TEHMMModel, self).save(modelPath)
        self.flatDataList = flatDataBackup
        
    def flattenObservations(self):
        """ hack to flatten observations into 1-d array since scikit hmm
        doesn't support multidemnsional output """
        self.flatMap = dict()
        self.flatMapBack = dict()
        self.flatDataList = []
        for data in self.dataList:
            flatData = np.zeros((data.shape[0], ), dtype=np.int)
            for col in xrange(data.shape[0]):
                entry = tuple(data[col])
                if entry in self.flatMap:
                    val = self.flatMap[entry]
                else:
                    val = len(self.flatMap)
                    self.flatMap[entry] = val
                    self.flatMapBack[val] = entry
                flatData[col] = val
            self.flatDataList.append(flatData)

    def unflattenStates(self, states):
        """ transforms the 1-d array back into a multidimensional array """
        numTracks = len(self.tracks)        
        unpackedStates = np.empty((len(states), numTracks), np.int)
        for i in xrange(len(states)):
            uState = self.flatMapBack[states[i]]
            for j in xrange(numTracks):
                unpackedStates[i,j] = uState[j]
        return unpackedStates

    def create(self, numStates, numIter = 10):
        """ Create the sckit learn multinomial hmm """

        flatEmissionModel = IndependentMultinomialEmissionModel(
            numStates=numStates,
            numSymbolsPerTrack=self.getNumSymbolsPerTrack())
        
        self.hmm = TEHmm(emissionModel = flatEmissionModel,
                         n_components = numStates,
       # self.hmm = MultinomialHMM(n_components = numStates,
                                  n_iter = numIter)

    def train(self):
        """ Use EM to estimate best parameters from scratch (unsupervised)
        Note that tracks and track data must already be read.  Should be
        wrapped up in simpler interface later... """

        #self.hmm.fit(self.flatDataList)
        self.hmm.fit(self.dataList)

    def score(self):
        """ Return the log probability of the data """
        score = 0.0
        #for flatData in self.flatDataList:
        #    score += self.hmm.score(flatData)
        for data in self.dataList:
            score += self.hmm.score(data)
        return score

    def viterbi(self):
        """ Return the output of the Viterbi algorithm on the loaded
        data: a tuple of (log likelihood of best path, and the path itself)
        """
        output = []
        #for flatData in self.flatDataList:
        #    prob, states = self.hmm.decode(self.flatData)
        #    assert len(states) == len(self.flatData)
        #    ouput.append(prob, self.unflattenStates(states))
        for flatData in self.flatDataList:
            prob, states = self.hmm.decode(self.flatData)
            assert len(states) == len(self.flatData)
            ouput.append(prob, self.unflattenStates(states))

        return output
        
    def toText(self):
        s = "NumStates = %d\n" % self.hmm.n_components
        s += "Start probs = %s\n" % self.hmm.startprob_
        s += "Transitions =\n%s\n" % str(self.hmm.transmat_)
        s += "Emissions =\n%s\n" % str(self.hmm.emissionModel.getLogProbs())
        return s
