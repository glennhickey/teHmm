#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
# Class derived from _BaseHMM and MultinomialHMM from sklearn/tests/hmm.py
# (2010 - 2013, scikit-learn developers (BSD License))
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

from sklearn.hmm import _BaseHMM
from sklearn.hmm import MultinomialHMM
from sklearn.hmm import _hmmc
from sklearn.hmm import normalize
from sklearn.hmm import NEGINF
from sklearn.utils import check_random_state, deprecated

""" Generalize MultitrackHmm (hmm.py) to a Stochastic Context Free Grammer
(CFG) while preserving more or less the same interface (and using the same
emission model).  The CFG is diferentiated by passing in a list of states
(which are all integers at this point) that trigger pair emissions (for example
these could be states corresponding to LTR or TSD).  All other states are
treated as HMM states (ie emit one column at a time left-to-right.  For now
these sets are disjoint, ie the nest states always emit pairs and the left to
right don't though that could be relaxed in theory.  For now only the
supervised training is supported, and we donn't implement full EM algorithm."""
class MultitrackCfg(object):
    def __init__(self, emissionModel=None,
                 nestStates=[]):
        self.emissionModel = emissionModel
        self.logProbs = None
        self.startProbs = None

        # all states that can emit a column
        self.emittingStates = [x for x in xrange(emissionModel.getNumStates())]

        # states that can emit but are not hmm states.  we keep track of their
        # id and index in a dictionary
        self.nestStates = dict()
        self.nestBack = dict()
        for idx, state in enumerate(list(nestStates)):
            self.nestStates[state] = idx
            self.nestBack[idx] = state
        
        # all the LR states (HMM states)
        self.hmmStates = []
        for i in self.emittingStates:
            if i not in nestStates:
                self.hmmStates.append(i)

        # we need to add a pair of CFG states for each nest state
        # for Chmosky Normal Form.
        # In particular, the CNF states are constructed as followed for
        # some nest State X (that emits x).
        # CNF(X)_1 -> X CNF(X)_2 (100% of the time)
        # CNF(X)_2 -> Y X (where Y is any HMM or CNF_1 state)
        # (and X is only reachable as a RHS of the above two productions)
        self.cnfStates = []
        for i in self.nestStates:
            self.cnfStates.append(self.getCnf(i))
            assert self.getBase(self.getCnf(i)) == i
            self.cnfStates.append(self.getCnf(i) + 1)
            assert self.getBase(self.getCnf(i) + 1) == i
            
        # total number of states
        self.M = len(self.hmmStates) + len(self.cnfStates) + \
                 len(self.nestStates)

        # sanity check to make sure we dont have same state twice
        assert len(set(self.hmmStates).union(
            set(self.cnfStates)).union(set(self.nestStates))) == self.M

        self.initParams()

    def getCnf(self, state):
        """ Map an emitting nested state to its first CNF state"""
        return len(self.emittingStates) + 2 * self.nestStates[state]

    def getBase(self, cnf):
        """ Map a CNF state to its corresponding emitting state """
        cnfIdx = (cnf - len(self.emittingStates)) / 2
        return self.nestBack[cnfIdx]

    def initParams(self):
        """ Allocate the production (transition) probability matrix and
        initialize it to a flat distribution where everything has equal prob."""
        self.startProbs = NEGINF + np.zeros((self.M,), dtype=np.float)
        for state in self.hmmStates:
            self.startProbs[state] = np.log(
                    1. / (len(self.emittingStates)))
        for i in xrange(0, len(self.cnfStates), 2):
            cnf1 = self.cnfStates[i]
            self.startProbs[cnf1] = np.log(
                    1. / (len(self.emittingStates)))
        
        self.logProbs = NEGINF + \
                        np.zeros((self.M, self.M, self.M), dtype=np.float)

        # do the hmm style productions (X -> XY)
        for state in self.hmmStates:
            # we can always got left to right to an hmm state
            for nextState in self.hmmStates:
                self.logProbs[state, state, nextState] = np.log(
                    1. / (len(self.emittingStates)))
            # we can also go left to right to a CNF1 state (which essentially
            # opens up a new nest pair)
            for nextState in self.nestStates:
                cnf1 = self.getCnf(nextState)
                self.logProbs[state, state, cnf1] = np.log(
                    1. / (len(self.emittingStates)))

        # do the cfg stype productions (CNF_1 -> XCNF_2; CNF_2 -> YX)
        assert len(self.cnfStates) % 2 == 0
        for i in xrange(0, len(self.cnfStates), 2):
            cnf1 = self.cnfStates[i]
            cnf2 = self.cnfStates[i + 1]
            base = self.getBase(cnf1)
            assert base == self.getBase(cnf2)
            # we can nest any hmm state
            self.logProbs[cnf1, base, cnf2] = 0.
            for nextState in self.hmmStates:
                self.logProbs[cnf2, nextState, base] = np.log(
                    1. / (len(self.emittingStates)))
            # we can also nest a CNF1 state (which essentially
            # opens up a new nest pair)
            for nextState in self.nestStates:
                nextCnf1 = self.getCnf(nextState)
                self.logProbs[cnf2, nextCnf1, base] = np.log(
                    1. / (len(self.emittingStates)))

        self.validate()

    def validate(self):
        """ check that all the probabilities in the production matrix rows add
        up to one"""
        
        total = 0.
        for state in xrange(self.M):
            total += np.exp(self.startProbs[state])
        assert_array_almost_equal(total, 1.)
        
        for origin in xrange(self.M):
            total = 0.
            for next1 in xrange(self.M):
                for next2 in xrange(self.M):
                    total += np.exp(self.logProbs[origin, next1, next2])
            if origin in self.nestStates:
                assert_array_almost_equal(total, 0.)
            else:
                assert_array_almost_equal(total, 1.)

    def __initDPTable(self, obs):
        """ Create the 2D dynamic programming table for CYK etc. and initialise
        all the 1-length entries for each (emitting) state"""
        self.table = NEGINF + np.zeros((len(obs), len(obs), self.M),
                                      dtype = np.float)
        emLogProbs = self.emissionModel.allLogProbs(obs)
        # todo: how fast is this type of loop?
        assert len(emLogProbs) == len(obs)
        for i in xrange(len(obs)):
            for j in xrange(len(emLogProbs[i])):
                self.table[i,i,j] = emLogProbs[i,j]

    def __initTraceBackTable(self, obs):
        """ Create a dynamic programming traceback (for CYK) table to remember
        which states got used """
        self.tb = -1 + np.zeros((len(obs), len(obs), self.M, 3),
                                dtype = np.int64)

    def __cyk(self, obs):
        """ Do the CYK dynamic programming algorithm (like viterbi) to
        compute the maximum likelihood CFG derivation of the observations."""
        self.__initDPTable(obs)
        self.__initTraceBackTable(obs)
        
        for size in xrange(2, len(obs) + 1):
            for i in xrange(len(obs) + 1 - size):
                j = i + size - 1
                for k in xrange(i, i + size - 1):
                    # test that X_ij -> Yi_k Zk+1_j (everything inclusive)
                    # TODO: optimize loops for hmm vs cfg states the brute
                    # for below is extremely wasteful (O(M^3)) insead of
                    # O(grammar size)
                    for lState in xrange(self.M):
                        for r1State in xrange(self.M):
                            for r2State in xrange(self.M):
                                lp = self.logProbs[lState, r1State, r2State] + \
                                     self.table[i, k, r1State] +\
                                     self.table[k+1, j, r2State]
                                if lp > self.table[i, j, lState]:
                                    self.table[i, j, lState] = lp
                                    self.tb[i, j, lState] = [k, r1State, r2State]

        score = max([self.startProbs[i] + self.table[0, len(obs)-1, i]  \
                     for i in xrange(self.M)])
        return score

    def __traceBack(self, obs):
        """ depth first search to determine most likely states from the
        traceback table (self.tb) that was constructed during __cyk"""
        trace = -1 + np.zeros(len(self.table))
        top = np.argmax([self.startProbs[i] + self.table[0, len(obs)-1, i]\
                             for i in xrange(self.M)])
        self.assigned = 0
        def tbRecursive(i, j, state, trace):
            size = j - i + 1
            if size == 1:
                trace[i] = state
                self.assigned += 1
            else:
                (k, r1State, r2State) = self.tb[i, j, state]
                assert k != j
                tbRecursive(i, k, r1State, trace)
                tbRecursive(k+1, j, r2State, trace)
        tbRecursive(0, len(self.tb) - 1, top, trace)
        assert self.assigned == len(obs)
        return trace
            
    def decode(self, obs):
        """ return tuple of log prob and most likely state sequence.  same
        as in the hmm. """
        return self.__cyk(obs), self.__traceBack(obs)
                                
                
            
            
