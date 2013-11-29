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
                self.logProbs[state, state, nextState] = np.log(
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
        for origin in xrange(self.M):
            total = 0.
            for next1 in xrange(self.M):
                for next2 in xrange(self.M):
                    total += np.exp(self.logProbs[origin, next1, next2])
            if origin in self.nestStates:
                assert_array_almost_equal(total, 0.)
            else:
                assert_array_almost_equal(total, 1.)
