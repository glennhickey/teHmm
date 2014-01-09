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
import copy
import itertools
import logging
import time
from numpy.testing import assert_array_equal, assert_array_almost_equal
from operator import mul
from ._emission import canFast, fastAllLogProbs, fastAccumulateStats, fastUpdateCounts
from .track import TrackTable
from .common import EPSILON, myLog
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
    def __init__(self, numStates, numSymbolsPerTrack, params = None,
                 zeroAsMissingData = True, fudge = 0.0):
        self.numStates = numStates
        self.numTracks = len(numSymbolsPerTrack)
        self.numSymbolsPerTrack = numSymbolsPerTrack
        # [TRACK, STATE, SYMBOL]
        self.logProbs = None
        self.zeroAsMissingData = zeroAsMissingData
        self.initParams(params)
        # little constant that gets added to frequencies during training
        # to prevent zero probabilities.  The bigger it is, the flatter
        # the distribution...
        self.fudge = fudge

    def getLogProbs(self):
        return self.logProbs

    def getNumStates(self):
        return self.numStates

    def getNumTracks(self):
        return self.numTracks

    def getNumSymbolsPerTrack(self):
        return self.numSymbolsPerTrack

    def getTrackSymbols(self, track):
        offset = 0
        if self.zeroAsMissingData is True:
            offset = 1
        for i in xrange(offset, self.numSymbolsPerTrack[track] + offset):
            yield i
        
    def getSymbols(self):
        """ iterate over all possible vectors of symbosl """
        if self.numTracks == 1:
            for i in self.getTrackSymbols(0):
                yield [i]
        else:
            valArrays = []
            for track in xrange(self.numTracks):
                if self.numSymbolsPerTrack[track] > 0:
                    valArrays.append(
                        [x for x in self.getTrackSymbols(track)])
                else:
                    valArrays.append([0])
            for val in itertools.product(*valArrays):
                yield val
    
    def initParams(self, params = None):
        """ initalize emission parameters such that all values are
        equally probable for each category.  if params is specifed, then
        assume it is the emission probability matrix and set our log probs
        to the log of it."""
        offset = 0
        if self.zeroAsMissingData:
            offset = 1
        logging.debug("Creating emission matrix with %d entries" %
                      (self.numTracks * self.numStates *
                      (offset + max(self.numSymbolsPerTrack))))
        self.logProbs = np.zeros((self.numTracks, self.numStates,
                                  offset + max(self.numSymbolsPerTrack)),
                                 dtype=np.float)

        logging.debug("Begin track by track emission matrix init")
        for i in xrange(self.numTracks):
            stateList = []
            logStateList = []
            for j in xrange(self.numStates):
                if params is None:
                    dist = normalize(1. + np.zeros(
                        self.numSymbolsPerTrack[i], dtype=np.float))
                else:
                    dist = np.array(params[i][j], dtype=np.float)
                # tack a 1 at the front of dist.  it'll be our uknown value
                if self.zeroAsMissingData == True:
                    dist = np.append([1.], dist)
                for k in xrange(len(dist)):
                    self.logProbs[i, j, k] = np.log(dist[k])
                    
        logging.debug("Validating emission matrix")
        assert len(self.logProbs) == self.numTracks
        for i in xrange(self.numTracks):
            assert len(self.logProbs[i]) == self.numStates
            for j in xrange(self.numStates):
                    assert (len(self.logProbs[i][j]) >=
                            self.numSymbolsPerTrack[i] + offset)
        self.validate()

    def singleLogProb(self, state, singleObs):
        """ Compute the log probability of a single observation, obs given
        a state."""
        logProb = 0.    
        for track, obsSymbol in enumerate(singleObs):
            # independence assumption means we can just add the tracks
            logProb += self.logProbs[track][state][obsSymbol]
        return logProb

    def allLogProbs(self, obs):
        """ obs is an array of observation vectors.  return an array of log
        probabilities.  this output array contains the probabilitiy for
        each state for each observation"""
        logging.debug("%s Computing multinomial log prob for %d %d-track "
                      "observations" % (time.strftime("%H:%M:%S"),
                                        obs.shape[0], self.getNumTracks()))
        obsLogProbs = np.zeros((obs.shape[0], self.numStates), dtype=np.float)
        if canFast(obs):
            logging.debug("Cython log prob enabled")
            fastAllLogProbs(obs, self.logProbs, obsLogProbs)
        else:
            for i in xrange(len(obs)):
                for state in xrange(self.numStates):
                    obsLogProbs[i, state] = self.singleLogProb(state, obs[i])
        logging.debug("%s Done computing log prob" % time.strftime("%H:%M:%S"))
        return obsLogProbs
    
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
            obsStats.append(self.fudge +
                            np.zeros((self.numStates,
                                      self.numSymbolsPerTrack[track]+1),
                                     dtype=np.float))
        return obsStats

    def accumulateStats(self, obs, obsStats, posteriors):
        """ For each observation, add the posterior probability of each state at
        that position, to the emission table.  Note that tracks are also
        treated completely independently here"""
        assert obs.shape[1] == self.numTracks
        logging.debug("%s Begin emission.accumulateStast for %d obs" % (
            time.strftime("%H:%M:%S"), len(obs)))
        if canFast(obs):
            logging.debug("Cython emission.accumulateStats enabled")
            fastAccumulateStats(obs, obsStats, posteriors)
        else:
            for i in xrange(len(obs)):
                for track in xrange(self.numTracks):
                    for state in xrange(self.numStates):
                        obsVal = obs[i,track]
                        obsStats[track][state, obsVal] += posteriors[i, state]
        logging.debug("%s Done emission.accumulateStast for %d obs" % (
            time.strftime("%H:%M:%S"), len(obs)))
        return obsStats
        
    def maximize(self, obsStats):
        for track in xrange(self.numTracks):
            for state in xrange(self.numStates):
                totalSymbol = 0.0
                for symbol in self.getTrackSymbols(track):
                    totalSymbol += obsStats[track][state, symbol]
                for symbol in self.getTrackSymbols(track):
                    denom = max(self.fudge, totalSymbol)
                    if denom != 0.:
                        symbolProb = obsStats[track][state, symbol] / denom
                    else:
                        symbolProb = 0.
                    self.logProbs[track][state][symbol] = myLog(symbolProb)
        self.validate()

    def validate(self):
        """ make sure everything sums to 1 """
        numSymbols = reduce(lambda x,y : max(x,1) * max(y,1),
                            self.numSymbolsPerTrack, 1)
        if numSymbols >= 500000:
            logging.warning("Unable two validate emission model because"
                            " there are too many (%d) symbosl" % numSymbols)
            return
        
        allSymbols = [x for x in self.getSymbols()]
        assert len(allSymbols) == numSymbols
        assert isinstance(self.logProbs, np.ndarray)
        assert len(self.logProbs.shape) == 3
        for state in xrange(self.numStates):
            total = 0.
            for val in allSymbols:
                assert len(val) == self.numTracks
                total += np.exp(self.singleLogProb(state, val))
            if len(allSymbols) > 0:
                assert_array_almost_equal(total, 1.)                    
            
    def supervisedTrain(self, trackData, bedIntervals):
        """ count the various emissions for each state.  Note that the
        iteration in this function assumes that both trackData and
        bedIntervals are sorted."""
        logging.debug("%s beginning supervised emission stats" % (
             time.strftime("%H:%M:%S")))
        trackTableList = trackData.getTrackTableList()
        numTables = len(trackTableList)
        assert numTables > 0
        assert len(bedIntervals) > 0
        obsStats = self.initStats()
        
        lastHit = 0
        for interval in bedIntervals:
            hit = False
            for tableIdx in xrange(lastHit, numTables):
                table = trackTableList[tableIdx]
                overlap = table.getOverlap(interval)
                if overlap is not None:
                    lastHit = tableIdx
                    hit = True
                    if canFast(table) is True:
                        fastUpdateCounts(overlap, table, obsStats)
                    else:
                        self.__updateCounts(overlap, table, obsStats)
                elif hit is True:
                    break
        logging.debug("%s beginning supervised emission max" % (
            time.strftime("%H:%M:%S")))
        self.maximize(obsStats)
        logging.debug("%s done supervised emission" % (
                          time.strftime("%H:%M:%S")))

        self.validate()

    def __updateCounts(self, bedInterval, trackTable, obsStats):
        """ Update the emission counts in obsStats using statistics from the
        known hidden states in bedInterval"""
        for pos in xrange(bedInterval[1], bedInterval[2]):
            # convert to position within track table
            tablePos = pos - trackTable.getStart()
            emissions = trackTable[tablePos]
            state = bedInterval[3]
            for track in xrange(self.getNumTracks()):
                obsStats[track][state, emissions[track]] += 1.

""" Simple pair emission model that supports emitting 1 or two states
simultaneousy.  Based on a normal emission but makes a simple distribution
for pairs """
class PairEmissionModel(object):
   def __init__(self, emissionModel, pairPriors):
       # base emissionmodel
       self.em = emissionModel
       # input observations can be linked as candidate pairs.  pairsPrior
       # models our confidence in these linkings (for each state).  For exameple,
       # if pairPrior is 0.95 for the LTR state, then emitting two linked
       # symbols is pr[emit obs1] X pr[emit obs2] X 0.95.  If they are not linked
       # then the prob is pr[emit obs1] X pr[emit obs2] X 0.05, etc.
       #
       # if None then ignored entirely
       pp = []
       for i in pairPriors:
           if i is None:
               pp.append([0, 0])
           elif i == 1:
               pp.append([NEGINF, 0.])
           else:
               pp.append([np.log(1. - i), np.log(i)])
       self.logPriors = np.array(pp, dtype=np.float)
       assert len(self.logPriors) == self.em.getNumStates()
       assert self.logPriors.shape == (self.em.getNumStates(), 2)
       
   def pairLogProb(self, state, logProb1, logProb2, match):
       """ compute the pair probability from two independent emission logprobs
       given a state and whether or not there is a match.
       Note that this function should eventually be in _cfg.pyx or something"""
       assert match == 0 or match == 1
       assert state < len(self.logPriors)
       return logProb1 + logProb2 + self.logPriors[state, int(match)]
   
