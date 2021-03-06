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
import logging
from collections import Iterable
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .emission import IndependentMultinomialEmissionModel
from .track import TrackList, TrackTable, Track
from .hmm import MultitrackHmm
from .common import EPSILON, LOGZERO, myLog, logger
from ._cfg import fastCykTable
from .basehmm import normalize
from .basehmm import NEGINF

""" Generalize MultitrackHmm (hmm.py) to a Stochastic Context Free Grammer
(CFG) while preserving more or less the same interface (and using the same
emission model).  The CFG is diferentiated by passing in a list of states
(which are all integers at this point) that trigger pair emissions (for example
these could be states corresponding to LTR or TSD).  All other states are
treated as HMM states (ie emit one column at a time left-to-right.  For now
these sets are disjoint, ie the nest states always emit pairs and the left to
right don't though that could be relaxed in theory.  For now only the
supervised training is supported, and we donn't implement full EM algorithm.

All productions look like one of the following (nonterminals are capitals):

X -> Y Z
X -> a
X -> aa
X -> a Y a

The first two are Chmosky Normal Form, but we add the last two kinds because
we want to explicitly model pair emissions.
"""
class MultitrackCfg(object):
    def __init__(self, emissionModel, pairEmissionModel,
                 nestStates=[], state_name_map = None):
        """ For now we take in an emission model (that was used for the HMM)
        and a list singling out a few states as nested / pair emission states"""
        self.emissionModel = emissionModel
        self.pairEmissionModel = pairEmissionModel
        self.logProbs1 = None
        self.logProbs2 = None
        self.startProbs = None
        self.defAlignmentSymbol = 0
        self.PAIRFLAG = -2
        self.trackList = None
        self.stateNameMap = state_name_map

        # all states that can emit a column
        self.M = self.emissionModel.getNumStates()
        self.emittingStates = np.arange(self.M)

        # all states flagged as nest enabled (used by training)
        self.nestStates = np.array(nestStates)

        # all states that are LR only (ie the rest)
        self.hmmStates = []
        for i in self.emittingStates:
            if i not in self.nestStates:
                self.hmmStates.append(i)
        self.hmmStates = np.array(self.hmmStates)
        assert len(self.hmmStates) + len(self.nestStates) == self.M
        
        # start with a basic flat distribution
        self.initParams()

    def getLogProbTables(self):
        """ Direct access to the production probability tables.  The first
        talble is for CNF type production X->YZ, and the second one is
        specifically for nested pair emission productions X->xYx"""
        return self.logProbs1, self.logProbs2

    def getStartProbs(self):
        """ Get the (non-log to be compatible with hmm) start probabilities"""
        return np.exp(self.startProbs)

    def getTrackList(self):
        return self.trackList

    def viterbi(self, trackData, numThreads = 1):
        """ Return the output of the Viterbi algorithm on the loaded
        data: a tuple of (log likelihood of best path, and the path itself)
        (one data point of each interval of track data)
        """ 
        output = []
        alignmentTrackTableList = trackData.getAlignmentTrackTableList()
        alignmentTable = None
        for i, trackTable in enumerate(trackData.getTrackTableList()):
            if len(alignmentTrackTableList) > 0:
               alignmentTable = alignmentTrackTableList[i]
            prob, states = self.decode(trackTable,
                                       alignmentTrack=alignmentTable,
                                       numThreads=numThreads)
            if self.stateNameMap is not None:
                states = map(self.stateNameMap.getMapBack, states)
            output.append((prob,states))
        return output

    def createTables(self):
        """ all probabailities of the form X -> Y Z (note that we will only
        ever use a small subset.  will keep track using a shortcut table for
        iterations """
        self.logProbs1 = NEGINF + np.zeros((self.M, self.M, self.M),
                                        dtype = np.float)

        """ all probabilities of the form X -> aYa (ie Y is nested inside of X)
        note that we are not counting probability of emitting a """
        self.logProbs2 = NEGINF + np.zeros((self.M, self.M), dtype = np.float)

    def createHelperTables(self):
        """ to avoid iterating over all possible productions when many may
        have zero probaiblities, we keep arrays of nozero productions """
        self.helper1 = []
        self.helperDim1 = np.zeros((self.M,), dtype=np.int32)
        maxRow = 0
        for state in self.emittingStates:
            stateList = []
            for next1 in self.emittingStates:
                for next2 in self.emittingStates:
                    if self.logProbs1[state, next1, next2] > NEGINF:
                        stateList.append([next1, next2])
            maxRow = max(len(stateList), maxRow)
            self.helperDim1[state] = len(stateList)
            self.helper1.append(stateList)
        npHelper1 = -1 + np.zeros((self.M, maxRow, 2), dtype=np.int32)
        for state in xrange(len(self.helper1)):
            for i, entry in enumerate(self.helper1[state]):
                assert len(entry) == 2
                npHelper1[state, i] = entry
        self.helper1 = npHelper1

        self.helper2 = []
        self.helperDim2 = np.zeros((self.M,), dtype=np.int32)
        maxRow = 0
        for state in self.emittingStates:
            stateList = []
            for nextState in self.emittingStates:
                if self.logProbs2[state, nextState] > NEGINF:
                    stateList.append(nextState)
            maxRow = max(len(stateList), maxRow)
            self.helperDim2[state] = len(stateList)
            self.helper2.append(stateList)
        npHelper2 = -1 + np.zeros((self.M, maxRow), dtype=np.int32)
        for state in xrange(len(self.helper2)):
            for i, entry in enumerate(self.helper2[state]):
                npHelper2[state, i] = entry
        self.helper2 = npHelper2
        

    def initParams(self):
        """ Allocate the production (transition) probability matrix and
        initialize it to a flat distribution where everything has equal prob."""
        self.createTables()

        # flat start probs over all states
        oneOfAny = myLog(1. / self.M)
        self.startProbs = oneOfAny + np.zeros((self.M,), dtype=np.float)

        # hmm states flat of X - > X Y (where Y is any state)
        for state in self.hmmStates:
            for nextState in self.emittingStates:
                self.logProbs1[state, state, nextState] = oneOfAny

        # cfg states X -> a Y a (prob Y nested in X).  X is a nested state
        # but Y is any state
        # (-1 below to correct counting for state == nextState case)
        oneOfCfgRHS = myLog(1. / (3.0 * self.M - 1.0))
        for state in self.nestStates:
            for nextState in self.emittingStates:
                self.logProbs1[state, state, nextState] = oneOfCfgRHS
                self.logProbs1[state, nextState, state] = oneOfCfgRHS
                self.logProbs2[state, nextState] = oneOfCfgRHS
        
        # create shortcut tables:
        self.createHelperTables()
       
        self.validate()

    def validate(self):
        """ check that all the probabilities in the production matrix rows add
        up to one"""
        
        total = 0.
        for state in self.emittingStates:
            total += np.exp(self.startProbs[state])
        assert_array_almost_equal(total, 1.)
        
        for origin in self.emittingStates:
            total = 0.
            for next1 in self.emittingStates:
                for next2 in self.emittingStates:
                    total += np.exp(self.logProbs1[origin, next1, next2])
                total += np.exp(self.logProbs2[origin, next1])
            assert_array_almost_equal(total, 1.)

        # now verify the helper tables exhibit the same behaviour
        for origin in self.emittingStates:
            total = 0.
            for i in xrange(self.helperDim1[origin]):
                total += np.exp(self.logProbs1[origin,
                                               self.helper1[origin, i][0],
                                               self.helper1[origin, i][1]])
            for i in xrange(self.helperDim2[origin]):
                total += np.exp(self.logProbs2[origin,
                                               self.helper2[origin, i]])
            assert_array_almost_equal(total, 1.)                

    def __initDPTable(self, obs, alignmentTrack):
        """ Create the 2D dynamic programming table for CYK etc. and initialise
        all the 1-length entries for each (emitting) state"""

        # Create a dynamic programming traceback (for CYK) table to remember
        #which states got used 
        self.tb = -1 + np.zeros((len(obs), len(obs), self.M, 3),
                                dtype = np.int64)

        self.dp = NEGINF + np.zeros((len(obs), len(obs), self.M),
                                      dtype = np.float)
        self.emLogProbs = self.emissionModel.allLogProbs(obs)
        # todo: how fast is this type of loop?
        assert len(self.emLogProbs) == len(obs)
        for i in xrange(len(obs)):
            for j in self.hmmStates:
                self.dp[i,i,j] = self.emLogProbs[i,j]
        baseMatch = alignmentTrack is not None
        # pair emissions where emitted columns are right beside eachother
        for i in xrange(len(obs)-1):
            match = baseMatch and\
                    alignmentTrack[i,0] != self.defAlignmentSymbol and\
                    alignmentTrack[i,0] == alignmentTrack[i+1,0]
            for j in self.nestStates:
                self.dp[i,i+1,j] = self.pairEmissionModel.pairLogProb(
                    j, self.emLogProbs[i,j], self.emLogProbs[i+1,j], match)
                self.tb[i, i+1, j] = [self.PAIRFLAG, 0, 0]

    def __cyk(self, obs, alignmentTrack = None, numThreads=1):
        """ Do the CYK dynamic programming algorithm (like viterbi) to
        compute the maximum likelihood CFG derivation of the observations."""
        self.__initDPTable(obs, alignmentTrack)
        if isinstance(obs, TrackTable):
            obs = obs.getNumPyArray()
        assert obs.dtype == np.uint8
        if alignmentTrack is None:
            alignmentTrack = np.ndarray((0,1), dtype = np.uint16)
        if isinstance(alignmentTrack, TrackTable):
            alignmentTrack = alignmentTrack.getNumPyArray()
        assert alignmentTrack.dtype == np.uint16
        fastCykTable(self, obs, alignmentTrack, numThreads)
        score = max([self.startProbs[i] + self.dp[0, len(obs)-1, i]  \
                     for i in self.emittingStates])
        return score

    def __traceBack(self, obs):
        """ depth first search to determine most likely states from the
        traceback table (self.tb) that was constructed during __cyk"""
        trace = -1 + np.zeros(len(self.dp))
        top = np.argmax([self.startProbs[i] + self.dp[0, len(obs)-1, i]\
                             for i in xrange(self.M)])
        tbRecurseStack = [(0, len(self.tb) - 1, top, trace)]
        while len(tbRecurseStack) > 0:
            i, j, state, trace = tbRecurseStack.pop()
            self.assigned = 0
            assert i >= 0
            assert j >= i
            size = j - i + 1
            if size == 1:
                trace[i] = state
                self.assigned += 1
            else:
                (k, r1State, r2State) = self.tb[i, j, state]
                if k == self.PAIRFLAG:
                    #cheap hack, k set to -2 to flag a pair emission
                    trace[i] = state
                    trace[j] = state
                    assert r1State == r2State
                    if size > 2:
                        tbRecurseStack.append((i+1, j-1, r1State, trace))
                else:
                    tbRecurseStack.append((i, k, r1State, trace))
                    tbRecurseStack.append((k+1, j, r2State, trace))

        assert self.assigned <= len(obs)
        return trace
            
    def decode(self, obs, alignmentTrack = None, defAlignmentSymbol=0,
               numThreads=1):
        """ return tuple of log prob and most likely state sequence.  same
        as in the hmm. """
        self.defAlignmentSymbol = defAlignmentSymbol
        if numThreads > 1:
            logger.info("%d threads activated for CYK" % numThreads)
        return self.__cyk(obs, alignmentTrack,
                          numThreads=numThreads), self.__traceBack(obs)

    def supervisedTrain(self, trackData, bedIntervals):
        """ Production porbabilites determined by frequencies two states
        are adjacent in the training data.  In fact, we mostly piggyback
        off the HMM training for now, using left-right adjacency as proxy
        for nesting events as well (not sure there's much way around this
        in fact..."""
        self.trackList = trackData.getTrackList()
        self.initParams()        

        hmm = MultitrackHmm(self.emissionModel)
        hmm.supervisedTrain(trackData, bedIntervals)

        assert (self.startProbs.shape == hmm.getStartProbs().shape)
        self.startProbs = myLog(hmm.getStartProbs())

        # map hmm transitions to X -> X Y productions
        # for each hmm transition, P(X->Y) we let
        # P(X->XY) = = P(X->YX) = P(X->Y) /2
        hmmProbs = hmm.transmat_
        for lState in self.hmmStates:
            for rState in self.emittingStates:
                hp = hmmProbs[lState, rState]
                if lState != rState:
                    hp /= 2.
                self.logProbs1[lState, lState, rState] = myLog(hp)
                self.logProbs1[lState, rState, lState] = myLog(hp)

        # like above but we also split across the table2
        for lState in self.nestStates:
            for rState in self.emittingStates:
                hp = hmmProbs[lState, rState]
                if lState != rState:
                    hp /= 3.
                else:
                    hp /= 2.
                self.logProbs1[lState, lState, rState] = myLog(hp)
                self.logProbs1[lState, rState, lState] = myLog(hp)
                self.logProbs2[lState, rState] = myLog(hp)

        # make sure the helper tables are up to date
        self.createHelperTables()
            
    def __str__(self):
        """ Pretty print model.  Note -- too much code duplicated here and in
        hmm (ie especially emission stuff..) TODO: merge up somehow"""
        hmmStates = self.hmmStates
        nestStates = self.nestStates
        states = xrange(self.M)
        if self.stateNameMap is not None:
            states = map(self.stateNameMap.getMapBack, states)
            hmmStates = map(self.stateNameMap.getMapBack, self.hmmStates)
            nestStates = map(self.stateNameMap.getMapBack, self.nestStates)
        s = "\nNumStates = %d:\nSingle:%s\nPair:%s\n" % (self.M, str(hmmStates),
                                                         str(nestStates))
        sp = [(states[i], self.startProbs[i])
              for i in xrange(self.M)] 
        s += "\nStart probs =\n%s\n" % str(sp)
        s += "\nlogTable1 = \n%s\n" % str(self.logProbs1)
        s += "\nlogTable2 = \n%s\n" % str(self.logProbs2)
        em = self.emissionModel         
        s += "\nNumber of symbols per track=\n%s\n" % str(
            em.getNumSymbolsPerTrack())
        s += "\nEmissions =\n"
        emProbs = em.getLogProbs()
        for state, stateName in enumerate(hmmStates):
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
                    if idx <= 2 or prob > 0.01:
                        s += "    %s) %s: %f (log=%f)\n" % (symbol, symbolName,
                                                            prob, myLog(prob))
        return s

