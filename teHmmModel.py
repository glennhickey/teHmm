#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from teModel import TEModel
from sklearn.hmm import MultinomialHMM
from sklearn.hmm import GaussianHMM

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
        #: 1-d version of data (unique entry for each vector)
        self.flatData = None

    def loadTrackData(self, tracksInfoPath, seqName, start, end,
                      forTraining = False):
        super(TEHMMModel, self).loadTrackData(tracksInfoPath, seqName,
                                              start, end, forTraining)
        self.flattenObservations()

    def save(self, modelPath):
        flatDataBackup = self.flatData
        self.flatData = None
        super(TEHMMModel, self).save(modelPath)
        self.flatData = flatDataBackup
        
    def flattenObservations(self):
        """ hack to flatten observations into 1-d array since scikit hmm
        doesn't support multidemnsional output """
        self.flatMap = dict()
        self.flatData = np.zeros((self.data.shape[0], ), dtype=np.int)
        for col in xrange(self.data.shape[0]):
            entry = tuple(self.data[col])
            if entry in self.flatMap:
                val = self.flatMap[entry]
            else:
                val = len(self.flatMap)
                self.flatMap[entry] = val
            self.flatData[col] = val

    def create(self, numStates):
        """ Create the sckit learn multinomial hmm """

        # Create some crappy start values to try to get model to not crash
        startprob = []
        for i in xrange(numStates):
            startprob.append(1. / float(numStates))
        transmat = []
        for i in xrange(numStates):
            transmat.append(startprob)
        emissionrow = []
        for i in xrange(len(self.tracks)):
            emissionrow.append(1. / float(len(self.tracks)))
        emissionprob = []
        for i in xrange(numStates):
            emissionprob.append(emissionrow)

        self.hmm = MultinomialHMM(n_components = numStates,
                                  startprob = startprob,
                                  transmat = transmat)
        self.hmm.emissionprob_ = emissionprob

    def train(self):
        """ Use EM to estimate best parameters from scratch (unsupervised)
        Note that tracks and track data must already be read.  Should be
        wrapped up in simpler interface later... """

        self.hmm.fit([self.flatData])

    def score(self):
        """ Return the log probability of the data """
        return self.hmm.score(self.flatData)

    def viterbi(self):
        """ Return the output of the Viterbi algorithm on the loaded
        data: a tuple of (log likelihood of best path, and the path itself)
        """
        prob, states = self.hmm.decode(self.flatData)
        assert len(states) == len(self.flatData)
        return (prob, states)
        
    def toText(self):
        s = "NumStates = %d\n" % self.hmm.n_components
        s += "Start probs = %s\n" % self.hmm.startprob_
        s += "Transitions =\n%s\n" % str(self.hmm.transmat_)
        s += "Emissions =\n%s\n" % str(self.hmm.emissionprob_)
        return s
