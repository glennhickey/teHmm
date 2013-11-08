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
from .emission import IndependentMultinomialEmissionModel
from .track import TrackList, TrackTable, Track

from sklearn.hmm import _BaseHMM
from sklearn.hmm import MultinomialHMM
from sklearn.hmm import _hmmc
from sklearn.hmm import normalize
from sklearn.hmm import NEGINF
    

"""
This class is based on the MultinomialHMM from sckikit-learn, but we make
the emission model a parameter. The custom emission model we support at this
point is a multi-*dimensional* multinomial. 
"""
class MultitrackHmm(_BaseHMM):
    def __init__(self, emissionModel,
                 n_components=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):
        
        """Create a hidden Markov model that supports multiple tracks.
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
        
        #: emission model is specified as parameter here
        self.emissionModel = emissionModel
        #: a TrackList object specifying information about the tracks
        self.trackList = None

    def train(self, trackData):
        """ Use EM to estimate best parameters from scratch (unsupervised)"""
        self.trackList = trackData.getTrackList()
        self.fit(trackData.getTrackTableList())

    def logProb(self, trackData):
        """ Return the log probability of the data (one score for each
        interval"""
        logProbList = []
        for trackTable in trackData.getTrackTableList():
            totalLogProb.append(self.score(trackTable))
        return logProbList

    def viterbi(self, trackData):
        """ Return the output of the Viterbi algorithm on the loaded
        data: a tuple of (log likelihood of best path, and the path itself)
        (one data point of each interval of track data)
        """ 
        output = []
        for trackTable in trackData.getTrackTableList():
            prob, states = self.decode(trackTable)
            output.append((prob,states))
        return output

    def load(self, path):
        f = open(path, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, path):
        f = open(path, "wb")
        pickle.dump(self.__dict__, f, 2)
        f.close()
        
    def toText(self):
        s = "NumStates = %d\n" % self.n_components
        s += "Start probs = %s\n" % self.startprob_
        s += "Transitions =\n%s\n" % str(self.transmat_)
        s += "Emissions =\n%s\n" % str(self.emissionModel.getLogProbs())
        return s
    
    ###########################################################################
    #       SCIKIT LEARN BASEHMM OVERRIDES BELOW 
    ###########################################################################

    def _compute_log_likelihood(self, obs):
        return self.emissionModel.allLogProbs(obs)

    def _generate_sample_from_state(self, state, random_state=None):
        return self.emissionModel.sample(state)

    def _init(self, obs, params='ste'):
        super(MultitrackHmm, self)._init(obs, params=params)
        self.emissionModel.initParams()

    def _initialize_sufficient_statistics(self):
        stats = super(MultitrackHmm, self)._initialize_sufficient_statistics()
        stats['obs'] = self.emissionModel.initStats()
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultitrackHmm, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            self.emissionModel.accumulateStats(obs, stats['obs'], posteriors)

    def _do_mstep(self, stats, params):
        super(MultitrackHmm, self)._do_mstep(stats, params)
        if 'e' in params:
            self.emissionModel.maximize(stats['obs'])

    def fit(self, obs, **kwargs):
        return _BaseHMM.fit(self, obs, **kwargs)
