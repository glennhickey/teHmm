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

""" Generlization of the sckit-learn multinomial to k dimensions.  Ie that the
observations are k-dimensional vectors -- one element for each track.
The probability of an observation in this model is the product of probabilities
for each track because we make the simplifying assumption that the tracks are
independent """
class IndependentMultinomialEmissionModel(object):
    def __init__(self, numStates, numSymbolsPerTrack):
        self.numStates = numStates
        self.numTracks = len(numSymbolsPerTrack)
        self.numSymbolsPerTrack = numSymbolsPerTrack
        # [TRACK, STATE, SYMBOL]
        self.logProbs = []

    def initParams(self, params = None):
        """ initalize emission parameters such that all values are
        equally probable for each category"""
        self.logProbs = []
        #todo: numpyify
        if params is None:
            for i in xrange(self.numTracks):
                stateList = []
                for j in xrange(self.numStates):
                    stateList.append(
                        normalize(1. + np.zeros(self.numSymbolsPerTrack[i],
                                                dtype=np.float)))
            self.logProbs.append(stateList)
        
        else:
            self.logProbs = params
        
        assert len(self.logProbs) == self.numTracks
        for i in xrange(self.numTracks):
            assert len(self.logProbs[i]) == self.numStates
            for j in xrange(self.numStates):
                assert len(self.logProbs[i][j]) == self.numSymbolsPerTrack[i]

    def singleLogProb(self, state, singleObs):
        """ Compute the log probability of a single observation, obs given
        a state."""
        assert state < self.numStates
        logProb = 0. 
        for track in xrange(self.numTracks):
            obsSymbol = singleObs[track]
            assert obsSymbol < self.numSymbolsPerTrack[track]
            # independence assumption means we can just add the tracks
            logProb += self.logProbs[track][state][obsSymbol]
        return logProb

    def allLogProbs(self, obs):
        """ obs is an array of observation vectors.  return an array of log
        probabilities.  this output array contains the probabilitiy for
        each state for each observation"""
        allLogProbs = np.zeros((obs.shape[0], self.numStates), dtype=np.float)
        for i in xrange(len(obs)):
            for state in xrange(self.numStates):
                allLogProbs[i, state] = self.singleLogProb(state, obs[i])
        return allLogProbs
    
    def sample(self, state):
        return None
        ##TODO adapt below code for multidimensional input
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        rand = random_state.rand()
        symbol = (cdf > rand).argmax()
        return symbol

    def initStats(self):
        """ Initialize an array to hold the accumulation statistics
        looks something like obsStats[TRAC][STATE][SYMBOL] = total probability
        of that STATE/SYMBOL pair across all observations """
        obsStats = []
        for track in xrange(self.numTracks):
            obsStats.append(np.zeros((self.numStates,
                                      self.numSymbolsPerTrack[track]),
                                     dtype=np.float))
        return obsStats

    def accumulateStats(self, obs, obsStats, posteriors):
        """ For each observation, add the posterior probability of each state at
        that position, to the emission table.  Note that tracks are also
        treated completely independently here"""
        for i in xrange(len(obs)):
            for track in xrange(self.numTracks):
                for state in xrange(self.numStates):
                    obsVal = obs[i][track]
                    obsStats[track][state, obsVal] += posteriors[i, state]
        return obsStats
        
    def maximimze(self, stats):
        for track in xrange(self.numTracks):
            self.params[track] = (stats['obs']
                                  / stats['obs'].sum(1)[:, np.newaxis])
