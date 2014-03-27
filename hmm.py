#!/usr/bin/env python
#Released under the MIT license, see LICENSE.txt
#Copyright (C) 2013 by Glenn Hickey

"""
 Class derived from _BaseHMM and MultinomialHMM from sklearn/tests/hmm.py
 See below:

Copyright (c) 2007-2014 the scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


import os
import sys
import numpy as np
import pickle
import string
import copy
import logging
import time
from collections import Iterable
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .emission import IndependentMultinomialEmissionModel
from .track import TrackList, TrackTable, Track
from .common import EPSILON, myLog, logger
from .basehmm import BaseHMM, check_random_state, NEGINF, ZEROLOGPROB, logsumexp
from .basehmm import normalize
from . import _hmm

"""
This class is based on the MultinomialHMM from sckikit-learn, but we make
the emission model a parameter. The custom emission model we support at this
point is a multi-*dimensional* multinomial. 
"""
class MultitrackHmm(BaseHMM):
    def __init__(self, emissionModel=None,
                 startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 state_name_map=None,
                 fudge=0.0,
                 fixTrans=False,
                 fixEmission=False,
                 fixStart=True,
                 forceUserTrans=None,
                 forceUserEmissions=None,
                 forceUserStart=None):
        if emissionModel is not None:
            n_components = emissionModel.getNumStates()
        else:
            n_components = 1

        """Create a hidden Markov model that supports multiple tracks.
        emissionModel must have already been created"""
        BaseHMM.__init__(self, n_components=n_components,
                          startprob=startprob,
                          transmat=transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)
        # remember init_params
        self.init_params = init_params
        #: emission model is specified as parameter here
        self.emissionModel = emissionModel
        #: a TrackList object specifying information about the tracks
        self.trackList = None
        #: a map between state values and names (track.CategoryMap)
        self.stateNameMap = state_name_map
        # little constant that gets added to frequencies during training
        # to prevent zero probabilities.  The bigger it is, the flatter
        # the distribution... (note that emission class has its own)
        self.fudge = fudge
        # freeze input transmat
        self.fixTrans = fixTrans
        # freeze input EmissionModel
        self.fixEmission = fixEmission
        # freeze input Start Probs
        self.fixStart = fixStart
        # keep track of number of EM iterations performed
        self.current_iteration = None
        # userTransition file (as line list) to apply after each iteration
        self.forceUserTrans = forceUserTrans
        if forceUserTrans is not None:
            with open(forceUserTrans) as f:
                self.forceUserTrans = f.readlines()
        # userEmissions file (as line list) to apply after each iteration
        self.forceUserEmissions = forceUserEmissions
        if forceUserEmissions is not None:
            with open(forceUserEmissions) as f:
                self.forceUserEmissions = f.readlines()
        # userStart file (as line list) to apply after each iteration
        self.forceUserStart = forceUserStart
        if forceUserStart is not None:
            with open(forceUserStart) as f:
                self.forceUserStart = f.readlines()

    def train(self, trackData):
        """ Use EM to estimate best parameters from scratch (unsupervised)"""
        self.trackList = trackData.getTrackList()
        self.fit(trackData.getTrackTableList())
        self.validate()

    def supervisedTrain(self, trackData, bedIntervals):
        """ Train directly from set of known states (4th column in the
        bedIntervals provided.  We assume that the most likely parameters
        are also just the expected values, which works for our basic
        multinomial distribution. Note that the states should already
        have been mapped to integers"""
        # NOTE bedIntervals must be sorted!
        self.trackList = trackData.getTrackList()
        N = self.emissionModel.getNumStates()
        transitionCount = self.fudge + np.zeros((N,N), np.float)
        freqCount = self.fudge + np.zeros((N,), np.float)
        prevInterval = None
        logger.debug("beginning supervised transition stats")
        for interval in bedIntervals:
            state = int(interval[3])
            assert state < N
            transitionCount[state,state] += interval[2] - interval[1] - 1
            freqCount[state] += interval[2] - interval[1]
            if prevInterval is not None and prevInterval[0] == interval[0]:
                if interval[1] < prevInterval[2]:
                    raise RuntimeError(
                        "Overlapping or out of order training intervals"
                        " detected: %s and %s."
                        (prevInterval, interval))
                elif interval[1] == prevInterval[2]:
                    transitionCount[prevInterval[3], state] += 1
            prevInterval = interval
        for row in xrange(len(transitionCount)):
            transitionCount[row] /= np.sum(transitionCount[row])
        self.transmat_ = np.copy(transitionCount)
        # scikit learn is too chicken to have 0-probs.  so we go back and
        # hack them in if necessary
        self._log_transmat = myLog(transitionCount)
        freqCount /= np.sum(freqCount)
        self.startprob_ = freqCount
        self.emissionModel.supervisedTrain(trackData, bedIntervals)
        self.validate()
        

    def logProb(self, trackData):
        """ Return the log probability of the data (one score for each
        interval"""
        logProbList = []
        for trackTable in trackData.getTrackTableList():
            totalLogProb.append(self.score(trackTable))
        return logProbList

    def viterbi(self, trackData, numThreads = 1):
        """ Return the output of the Viterbi algorithm on the loaded
        data: a tuple of (log likelihood of best path, and the path itself)
        (one data point of each interval of track data)
        """
        # Thread interface provided but not implemented:
        assert numThreads == 1
        output = []
        for trackTable in trackData.getTrackTableList():
            logger.debug("Beginning hmm viterbi decode")
            prob, states = self.decode(trackTable)
            logger.debug("Done hmm viterbi decode")
            if self.stateNameMap is not None:
                states = map(self.stateNameMap.getMapBack, states)
            output.append((prob,states))
        return output
        
    def __str__(self):
        states = [x for x in xrange(self.n_components)]
        if self.stateNameMap is not None:
            states = map(self.stateNameMap.getMapBack, states)
        s = "\nNumStates = %d:\n%s\n" % (self.n_components, str(states))
        sp = [(states[i], self.startprob_[i])
              for i in xrange(self.n_components)] 
        s += "\nStart probs =\n%s\n" % str(sp)
        s += "\nTransitions =\n%s\n" % str(self.transmat_)
        s += "\nlogTransitions = \n%s\n" % str(myLog(self.transmat_))
        em = self.emissionModel         
        s += "\nNumber of symbols per track=\n%s\n" % str(
            em.getNumSymbolsPerTrack())
        s += "\nEmissions =\n"
        emProbs = em.getLogProbs()
        for state, stateName in enumerate(states):
            s += "State %s:\n" % stateName
            for trackNo in xrange(em.getNumTracks()):
                track = self.trackList.getTrackByNumber(trackNo)
                s += "  Track %d %s (%s):\n" % (track.getNumber(),
                                                track.getName(),
                                                track.getDist())
                numSymbolsPerTrack =  em.getNumSymbolsPerTrack()
                for idx, symbol in enumerate(em.getTrackSymbols(trackNo)):
                    symbolName = track.getValueMap().getMapBack(symbol)
                    prob = np.exp(emProbs[trackNo][state][symbol])
                    if prob > 0.0000005:
                        logval = str(myLog(prob))
                        if prob == 0.0:
                            logval = "-inf"
                        s += "    %s) %s: %f (log=%s)\n" % (symbol, symbolName,
                                                            prob, logval)
        return s

    def getTrackList(self):
        return self.trackList

    def getStartProbs(self):
        return self.startprob_

    def getTransitionProbs(self):
        return self.transmat_

    def getStateNameMap(self):
        return self.stateNameMap

    def getEmissionModel(self):
        return self.emissionModel

    def validate(self):
        assert len(self.startprob_) == self.emissionModel.getNumStates()
        assert not isinstance(self.startprob_[0], Iterable)
        assert len(self.transmat_) == self.emissionModel.getNumStates()
        assert len(self.transmat_[0]) == self.emissionModel.getNumStates()
        assert_array_almost_equal(np.sum(self.startprob_), 1.)
        for i in xrange(self.emissionModel.getNumStates()):
            assert_array_almost_equal(np.sum(self.transmat_[i]), 1.)
        self.emissionModel.validate()
        
    def applyUserEmissions(self, userEmLines):
        """ Modify the emission distribution with user specified values
        read directly from text file into line array"""
        assert self.stateNameMap is not None
        assert self.trackList is not None
        assert self.emissionModel is not None
        self.emissionModel.applyUserEmissions(userEmLines, self.stateNameMap,
                                              self.trackList)
        
    def applyUserTrans(self, userTransLines):
        """ Modify the transtion probability matrix so that it contains the
        probabilities specified by the given text file.  If a stateNameMap
        (catMap)
        is provided, it is used (and missing values trigger errors).  If the map
        is None, then one is created.  If the transmap is None, one is created
        as well (with default values being flat distribution.
        The modified transMat and catMap are returned as a tuple, can can be
        applied to the hmm."""
        logger.debug("Applying user transitions to HMM")

        N = self.n_components
        mask = np.zeros((N, N), dtype=np.int8)
        transMat = self.transmat_
        catMap = self.stateNameMap

        # init the transmap if ncessary
        if transMat is None:
            transMat = 1. / float(N) + np.zeros((N, N), dtype=np.float)

        # read the probabilities into the transmap
        f = userTransLines    
        for line in f:
            if len(line.lstrip()) > 0 and line.lstrip()[0] is not "#":
                toks = line.split()
                assert len(toks) == 3
                prob = float(toks[2])
                fromState = toks[0]
                toState = toks[1]
                if not catMap.has(fromState) or not catMap.has(toState):
                    raise RuntimeError("Cannot apply transition %s->%s to model"
                                       " since at least one of the states was "
                                       "not found in the supervised data." % (
                                           fromState, toState))
                fid = catMap.getMap(fromState)
                tid = catMap.getMap(toState)
                # remember that this is a user transition
                mask[fid, tid] = 1
                # set the trnasition probability in the matrix
                transMat[fid, tid] = prob

        # normalize all other probabilities (ie that are unmaksed) so that they
        # add up.
        for fid in xrange(N):
            # total probability of learned states
            curTotal = 0.0
            # total probability of learned states after normalization
            tgtTotal = 1.0
            for tid in xrange(N):
                if mask[fid, tid] == 1:
                    tgtTotal -= transMat[fid, tid]
                else:
                    curTotal += transMat[fid, tid]
            if tgtTotal < -EPSILON:
                raise RuntimeError("User defined probability %f from state %s "
                                   "exceeds 1" % (tgtTotal,
                                                  catMap.getMapBack(fid)))
            for tid in xrange(N):
                if mask[fid, tid] == 0:
                    if tgtTotal == 0.:
                        transMat[fid, tid] = 0.
                    else:
                        transMat[fid, tid] *= (tgtTotal / curTotal)

        # reset back to make sure logs get updated too
        self.transmat_ = transMat

        
    def applyUserStarts(self, userStartLines):
        """ modify a HMM that was constructed using supervisedTrain() so that
        it contains the start probabilities specified in the userStartPath File."""
        logger.debug("Applying user starts to HMM")
        f = userStartLines
        startProbs = self.startprob_

        N = self.n_components
        if startProbs is None:
            startProbs = 1. / float(len(self.stateNameMap)) + np.zeros((N))
        mask = np.zeros(startProbs.shape, dtype=np.int8)

        # scan file and set values in logProbs matrix
        for line in f:
            if len(line.lstrip()) > 0 and line.lstrip()[0] is not "#":
                toks = line.split()
                assert len(toks) == 2
                stateName = toks[0]
                prob = float(toks[1])
                if not self.stateNameMap.has(stateName):
                    raise RuntimeError("State %s not found in supervised data" %
                                       stateName)
                state = self.stateNameMap.getMap(stateName)
                startProbs[state] = prob
                mask[state] = 1

        # normalize all other probabilities (ie that are unmaksed) so that they
        # add up.
        # total probability of learned states
        curTotal = 0.0
        # total probability of learned states after normalization
        tgtTotal = 1.0            

        for state in xrange(N):
            if mask[state] == 1:
                tgtTotal -= startProbs[state]
            else:
                curTotal += startProbs[state]

            if tgtTotal < 0.:
                raise RuntimeError("User defined start probabiliies exceed 1")

        for state in xrange(N):
            # same correction as applyUserTransmissions()....
            if mask[state] == 0:
                if tgtTotal == 0.:
                    startProbs[state] = 0.
                else:
                    startProbs[state] *= (tgtTotal / curTotal)

        self.startprob_ = startProbs

                
    ###########################################################################
    #       SCIKIT LEARN BASEHMM OVERRIDES BELOW 
    ###########################################################################

    def _compute_log_likelihood(self, obs):
        return self.emissionModel.allLogProbs(obs)

    def _generate_sample_from_state(self, state, random_state=None):
        return self.emissionModel.sample(state)
        
    def _init(self, obs, params='ste'):
        if self.fixTrans is True:
            self.params = self.params.replace("t", "")
        if self.fixEmission is True:
            self.params = self.params.replace("e", "")
        if self.fixStart is True:
            self.params = self.params.replace("s", "")
        super(MultitrackHmm, self)._init(obs, params=params)
        self.random_state = check_random_state(self.random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(MultitrackHmm, self)._initialize_sufficient_statistics()
        stats['obs'] = self.emissionModel.initStats()
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        logger.debug("%d: beginning MultitrackHMM E-step" %
                      self.current_iteration)
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            logger.debug("beginning Transition E-substep")
            n_observations, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_observations > 1:
                logsum_lneta = np.zeros((n_components, n_components))

                lnP = logsumexp(fwdlattice[-1])
                _hmm._log_sum_lneta(n_observations, n_components, fwdlattice,
                                     self._log_transmat, bwdlattice,
                                     framelogprob, lnP, logsum_lneta)

                stats["trans"] += np.exp(logsum_lneta)

        if 'e' in params:
            logger.debug("beginning Emissions E-substep")
            self.emissionModel.accumulateStats(obs, stats['obs'], posteriors)

        logger.debug("ending MultitrackHMM E-step")

    def _do_mstep(self, stats, params):
        logger.debug("%d: beginning MultitrackHMM M-step" %
                      self.current_iteration)
        self.validate()
        if self.startprob_prior is None:
            self.startprob_prior = 1.0
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if 's' in params:
            self.startprob_ = normalize(
                np.maximum(self.startprob_prior - 1.0 + stats['start'], 1e-20))

        if 't' in params:
            lastMat = copy.deepcopy(self.transmat_)
            transmat_ = self.transmat_prior - 1.0 + stats['trans']
            for row in xrange(len(transmat_)):
                rowSum = np.sum(transmat_[row])
                if rowSum < EPSILON:
                    # orphaned state.  dont zap just leave values from
                    # last iteration
                    transmat_[row] = lastMat[row]
                else:
                    transmat_[row] = transmat_[row] / rowSum
            self.transmat_ = transmat_

        if 'e' in params:
            self.emissionModel.maximize(stats['obs'])
        logger.debug("%d: ending MultitrackHMM M-step" %
                      self.current_iteration)
        self.current_iteration += 1

        # apply the force user params if specified
        if self.forceUserTrans is not None:
            self.applyUserTrans(self.forceUserTrans)
        if self.forceUserEmissions is not None:
            self.applyUserEmissions(self.forceUserEmissions)
        if self.forceUserStart is not None:
            self.applyUserStarts(self.forceUserStart)

        self.validate()

    def fit(self, obs, **kwargs):
        self.current_iteration = 1
        return BaseHMM.fit(self, obs, **kwargs)

    # Getting annoyed with epsilons being added by scikit learn
    # so redo tranmat property to allow zeros (should probably do
    # for start probs as well at some point)
    def _get_transmat(self):
        """Matrix of transition probabilities."""
        return np.exp(self._log_transmat)

    def _set_transmat(self, transmat):
        if transmat is None:
            transmat = np.tile(1.0 / self.n_components,
                               (self.n_components, self.n_components))

        if (np.asarray(transmat).shape
                != (self.n_components, self.n_components)):
            raise ValueError('transmat must have shape '
                             '(n_components, n_components)')
        if not np.all(np.allclose(np.sum(transmat, axis=1), 1.0)):
            raise ValueError('Rows of transmat must sum to 1.0')

        self._log_transmat = myLog(np.asarray(transmat).copy())

    transmat_ = property(_get_transmat, _set_transmat)

    def _get_startprob(self):
        """Mixing startprob for each state."""
        return np.exp(self._log_startprob)

    def _set_startprob(self, startprob):
        if startprob is None:
            startprob = np.tile(1.0 / self.n_components, self.n_components)
        else:
            startprob = np.asarray(startprob, dtype=np.float)

        if len(startprob) != self.n_components:
            raise ValueError('startprob must have length n_components')
        if not np.allclose(np.sum(startprob), 1.0):
            raise ValueError('startprob must sum to 1.0')

        self._log_startprob = myLog(np.asarray(startprob).copy())

    startprob_ = property(_get_startprob, _set_startprob)        
    
    def _do_viterbi_pass(self, framelogprob):
        """ Viterbi dynamic programming.  Overrides the original version
        which is still in basehmm.py, to use the faster Cython code """
        n_observations, n_components = framelogprob.shape
        state_sequence, logprob = _hmm._viterbi(
            n_observations, n_components, self._log_startprob,
            self._log_transmat, framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        """ Forward dynamic programming.  Overrides the original version
        which is still in basehmm.py, to use the faster Cython code """
        n_observations, n_components = framelogprob.shape
        logger.debug("beginning Forward pass on %d x %d matrix" % (
            n_observations, n_components))
        fwdlattice = np.zeros((n_observations, n_components))
        _hmm._forward(n_observations, n_components, self._log_startprob,
                       self._log_transmat, framelogprob, fwdlattice)
        lp = logsumexp(fwdlattice[-1])
        logger.debug("Forward log prob %f" % lp)
        return lp, fwdlattice

    def _do_backward_pass(self, framelogprob):
        """ Backward dynamic programming.  Overrides the original version
        which is still in basehmm.py, to use the faster Cython code """
        n_observations, n_components = framelogprob.shape
        logger.debug("beginning Backward pass on %d x %d matrix" % (
            n_observations, n_components))
        bwdlattice = np.zeros((n_observations, n_components))
        _hmm._backward(n_observations, n_components, self._log_startprob,
                        self._log_transmat, framelogprob, bwdlattice)
        lp = logsumexp(bwdlattice[0])
        logger.debug("Backward log prob + start %f" % (lp +
                     logsumexp(self._log_startprob)))
        return bwdlattice

