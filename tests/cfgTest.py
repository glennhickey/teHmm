#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
# Contains some fragments of code from sklearn/tests/test_hmm.py
# (2010 - 2013, scikit-learn developers (BSD License))
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal

from teHmm.tracksInfo import TracksInfo
from teHmm.track import *
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.emission import IndependentMultinomialEmissionModel

from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfoPath
from teHmm.tests.emissionTest import getBedStates

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testInit(self):
        emissionModel = IndependentMultinomialEmissionModel(
            10, [3], zeroAsMissingData=False)
        cfg = MultitrackCfg(emissionModel)
        cfg = MultitrackCfg(emissionModel, [3,8])
        cfg.validate()

    def testDefaultVsHmm(self):
        emissionModel = IndependentMultinomialEmissionModel(
            10, [3], zeroAsMissingData=False)
        hmm = MultitrackHmm(emissionModel)
        cfg = MultitrackCfg(emissionModel)

    def testDefaultHmmViterbi(self):
        emissionModel = IndependentMultinomialEmissionModel(
            5, [3], zeroAsMissingData=False)
        hmm = MultitrackHmm(emissionModel)
        cfg = MultitrackCfg(emissionModel)
        obs = np.array([[0],[0],[1],[2]], dtype=np.int16)
        hmmProb, hmmStates = hmm.decode(obs)
        cfgProb, cfgStates = cfg.decode(obs)
        assert_array_almost_equal(hmmProb, cfgProb)

    def testTraceBack(self):
        # a model with 2 states.  state 0 has a .75 chance of emitting 0
        # state 1 has a 0.95 chance of emitting 1
        emissionModel = IndependentMultinomialEmissionModel(
            2, [2], zeroAsMissingData=False)
        emProbs = np.zeros((1, 2, 2), dtype=np.float)
        emProbs[0,0] = [0.75, 0.25]
        emProbs[0,1] = [0.05, 0.95]
        emissionModel.logProbs = emProbs

        hmm = MultitrackHmm(emissionModel)
        cfg = MultitrackCfg(emissionModel)
        obs = np.array([[0],[0],[1],[0]], dtype=np.int16)
        hmmProb, hmmStates = hmm.decode(obs)
        cfgProb, cfgStates = cfg.decode(obs)
        assert_array_almost_equal(hmmProb, cfgProb)
        assert_array_almost_equal(hmmStates, [0, 0, 1, 0])
        assert_array_almost_equal(hmmStates, cfgStates)
        
        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

