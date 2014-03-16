#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt

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
from .track import TrackTable, BinaryMap
from .common import EPSILON, myLog, logger
from .basehmm import normalize, NEGINF

""" Generlization of the sckit-learn multinomial to k dimensions.  Ie that the
observations are k-dimensional vectors -- one element for each track.
The probability of an observation in this model is the product of probabilities
for each track because we make the simplifying assumption that the tracks are
independent """
class IndependentMultinomialEmissionModel(object):
    def __init__(self, numStates, numSymbolsPerTrack, params = None,
                 zeroAsMissingData = True, fudge = 0.0, normalizeFac = 0.0,
                 randomize=False):
        self.numStates = numStates
        self.numTracks = len(numSymbolsPerTrack)
        self.numSymbolsPerTrack = numSymbolsPerTrack
        # [TRACK, STATE, SYMBOL]
        self.logProbs = None
        self.zeroAsMissingData = zeroAsMissingData
        # little constant that gets added to frequencies during training
        # to prevent zero probabilities.  The bigger it is, the flatter
        # the distribution...
        self.fudge = fudge
        # normalization factor
        # 0: emission scores left as-is (no normalization)
        # 1: emission probabiliy is divided by number of tracks: ie
        #    emission probability and transition probability are equally
        #    weighted no matter how many tracks there are
        # k: emission probability is divided by (number of tracks / k):
        # (transform into constant to add when doing logprobs)
        # (ie logprob --> logprob + self.normalizeFac)
        self.normalizeFac = 1.
        if normalizeFac > 0:
            self.normalizeFac = float(normalizeFac) / float(self.numTracks)
        self.initParams(params=params, randomize=randomize)

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
    
    def initParams(self, params = None, randomize=False):
        """ initalize emission parameters such that all values are
        equally probable for each category.  if params is specifed, then
        assume it is the emission probability matrix and set our log probs
        to the log of it."""
        offset = 0
        if self.zeroAsMissingData:
            offset = 1
        logger.debug("Creating emission matrix with %d entries" %
                      (self.numTracks * self.numStates *
                      (offset + max(self.numSymbolsPerTrack))))
        self.logProbs = np.zeros((self.numTracks, self.numStates,
                                  offset + max(self.numSymbolsPerTrack)),
                                 dtype=np.float)

        logger.debug("Begin track by track emission matrix init (random=%s)" %
                      randomize)
        for i in xrange(self.numTracks):
            stateList = []
            logStateList = []
            for j in xrange(self.numStates):
                if params is None:
                    if randomize is False:
                        dist = normalize(1. + np.zeros(
                        self.numSymbolsPerTrack[i], dtype=np.float))
                    else:
                        dist = normalize(np.random.random_sample(
                            self.numSymbolsPerTrack[i]))
                else:
                    dist = np.array(params[i][j], dtype=np.float)
                # tack a 1 at the front of dist.  it'll be our uknown value
                if self.zeroAsMissingData == True:
                    dist = np.append([1.], dist)
                for k in xrange(len(dist)):
                    self.logProbs[i, j, k] = np.log(dist[k])
                    
        logger.debug("Validating emission matrix")
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
        logProb = 0.0
        for track, obsSymbol in enumerate(singleObs):
            # independence assumption means we can just add the tracks
            logProb += self.logProbs[track][state][int(obsSymbol)]
        return logProb * self.normalizeFac

    def allLogProbs(self, obs):
        """ obs is an array of observation vectors.  return an array of log
        probabilities.  this output array contains the probabilitiy for
        each state for each observation"""
        logger.debug("Computing multinomial log prob for %d %d-track "
                      "observations" % (obs.shape[0], self.getNumTracks()))
        obsLogProbs = np.zeros((obs.shape[0], self.numStates), dtype=np.float)
        if canFast(obs):
            logger.debug("Cython log prob enabled")
            fastAllLogProbs(obs, self.logProbs, obsLogProbs, self.normalizeFac)
        else:
            for i in xrange(len(obs)):
                for state in xrange(self.numStates):
                    obsLogProbs[i, state] = self.singleLogProb(state, obs[i])
        logger.debug("Done computing log prob")
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
        obsStats = np.zeros((self.numTracks, self.numStates,
                             np.max(self.numSymbolsPerTrack) + 1), 
                             dtype=np.float)
        for track in xrange(self.numTracks):
            for state in xrange(self.numStates):
                for symbol in xrange(self.numSymbolsPerTrack[track] + 1):
                    obsStats[track, state, symbol] += self.fudge
        return obsStats

    def accumulateStats(self, obs, obsStats, posteriors):
        """ For each observation, add the posterior probability of each state at
        that position, to the emission table.  Note that tracks are also
        treated completely independently here"""
        assert obs.shape[1] == self.numTracks
        logger.debug("Begin emission.accumulateStast for %d obs" % len(obs))
        if canFast(obs):
            logger.debug("Cython emission.accumulateStats enabled")
            fastAccumulateStats(obs, obsStats, posteriors)
        else:
            for i in xrange(len(obs)):
                for track in xrange(self.numTracks):
                    obsVal = obs[i,track]                    
                    for state in xrange(self.numStates):
                        obsStats[track, state, int(obsVal)] += posteriors[i, state]
        logger.debug("Done emission.accumulateStast for %d obs" % len(obs))
        return obsStats
        
    def maximize(self, obsStats):
        for track in xrange(self.numTracks):
            for state in xrange(self.numStates):
                totalSymbol = 0.0
                for symbol in self.getTrackSymbols(track):
                    totalSymbol += obsStats[track, state, symbol]
                lastMat = copy.deepcopy(self.logProbs[track][state])
                trackSum = 0
                for symbol in self.getTrackSymbols(track):
                    denom = max(self.fudge, totalSymbol)
                    if denom != 0.:
                        symbolProb = obsStats[track, state, symbol] / denom
                    else:
                        symbolProb = 0.
                    # no longer want to have absolute zero emissions
                    # as it can lead to unrecognizable strings (we elect to
                    # allow for epsilon in emissions but keep the 0s in
                    # transitions) so we override logZero
                    trackSum += symbolProb
                    self.logProbs[track][state][symbol] = myLog(symbolProb,
                                                                logZeroVal=-1e6)
                if trackSum < EPSILON:
                    # orphaned state/track has no emission. just leave as was
                    self.logProbs[track][state] = lastMat
        self.validate()

    def validate(self):
        """ make sure everything sums to 1 """
        numSymbols = reduce(lambda x,y : max(x,1) * max(y,1),
                            self.numSymbolsPerTrack, 1)
        if numSymbols >= 500000:
            logger.debug("Warning-Unable to validate emission model because"
                            " there are too many (%d) symbols" % numSymbols)
            return
        if self.normalizeFac != 1.0:
            # sum-to-one doesn't work for normalizeFac.  Should eventually
            # just incorporate into check below, however.
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
        logger.debug("%beginning supervised emission stats")
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
        logger.debug("beginning supervised emission max")
        self.maximize(obsStats)
        logger.debug("done supervised emission")
        
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
                obsStats[track, state, emissions[track]] += 1.

    def applyUserEmissions(self, userEmLines, stateMap, trackList):
        """ modify a HMM that was constructed using supervisedTrain() so that
        it contains the emission probabilities specified in the userEmPath File.
        """
        logger.debug("Applying user emissions Emission Model")
        f = userEmLines
        logProbs = self.getLogProbs()
        mask = np.zeros(logProbs.shape, dtype=np.int8)

        # scan file and set values in logProbs matrix
        for line in f:
            if len(line.lstrip()) > 0 and line.lstrip()[0] is not "#":
                toks = line.split()
                assert len(toks) == 4
                stateName = toks[0]
                trackName = toks[1]
                symbolName = toks[2]
                prob = float(toks[3])
                if not stateMap.has(stateName):
                    raise RuntimeError("User Emission: State %s not found" %
                                       stateName)
                state = stateMap.getMap(stateName)
                track = trackList.getTrackByName(trackName)
                if track is None:
                    raise RuntimeError("Track %s (in user emissions) not found" %
                                       trackName)
                symbolMap = track.getValueMap()
                track = track.getNumber()
                if isinstance(symbolMap, BinaryMap):
                    # hack in conversion for binary data, where map expects
                    # either None or non-None
                    if symbolName == "0" or symbolName == "None":
                        symbolName = None
                    symbol = symbolMap.getMap(symbolName)
                else:
                    # be careful to catch excpetion when scaling non-numeric value
                    try:
                        hasSymbol = symbolMap.has(symbolName)
                        symbol = symbolMap.getMap(symbolName)
                    except:
                        hasSymbol = False
                        symbol = symbolMap.getMissingVal()
                    if not hasSymbol:
                        logger.warning("Track %s Symbol %s not found in"
                                       "data (setting as null value)" %
                                       (trackName, symbolName))
                        
                assert symbol in self.getTrackSymbols(track)
                logProbs[track, state, symbol] = myLog(prob)
                mask[track, state, symbol] = 1

        # easier to work outside log space
        probs = np.exp(logProbs)

        # normalize all other probabilities (ie that are unmaksed) so that they
        # add up.
        for track in xrange(self.getNumTracks()):
            for state in xrange(self.getNumStates()):
                # total probability of learned states
                curTotal = 0.0
                # total probability of learned states after normalization
                tgtTotal = 1.0            
                for symbol in self.getTrackSymbols(track):
                    if mask[track, state, symbol] == 1:
                        tgtTotal -= probs[track, state, symbol]
                    else:
                        curTotal += probs[track, state, symbol]
                    if tgtTotal < 0.:
                        raise RuntimeError("User defined prob from state %s"
                                           " for track %s exceeds 1" %
                                           (stateMap.getMapBack(state),
                                            symbolMap.getMapBack(symbol)))

                # special case:
                # we have set probabilities that total < 1 and no remaining
                # probabilities to boost with factor. ex (1, 0, 0, 0) ->
                #(0.95, 0, 0, 0)  (where the first prob is being set)
                additive = False
                if curTotal == 0. and tgtTotal < 1.:
                    additive = True
                    numUnmasked = len(mask[track, state]) - \
                      np.sum(mask[track,state])
                    assert numUnmasked > 0
                    addAmt = (1. - tgtTotal) / float(numUnmasked)
                else:
                    assert curTotal > 0.
                    multAmt = tgtTotal / curTotal

                # same correction as applyUserTransmissions()....
                for symbol in self.getTrackSymbols(track):
                    if mask[track, state, symbol] == 0:
                        if tgtTotal == 0.:
                            probs[track, state, symbol] = 0.
                        elif additive is False:
                            probs[track, state, symbol] *= multAmt
                        else:
                            probs[track, state, symbol] += addAmt

        # Make sure we set our new log probs back into object
        self.logProbs = myLog(probs)

        self.validate()
                
                
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
   
