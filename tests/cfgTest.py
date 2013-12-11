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
from teHmm.trackIO import readBedIntervals
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.emission import IndependentMultinomialEmissionModel, PairEmissionModel

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
        pairModel = PairEmissionModel(emissionModel, [1.0] *
                                      emissionModel.getNumStates())
        cfg = MultitrackCfg(emissionModel, pairModel)
        cfg = MultitrackCfg(emissionModel, [3,8])
        cfg.validate()

    def testDefaultVsHmm(self):
        emissionModel = IndependentMultinomialEmissionModel(
            10, [3], zeroAsMissingData=False)
        hmm = MultitrackHmm(emissionModel)
        pairModel = PairEmissionModel(emissionModel, [1.0] *
                                      emissionModel.getNumStates())
        cfg = MultitrackCfg(emissionModel, pairModel)


    def testDefaultHmmViterbi(self):
        emissionModel = IndependentMultinomialEmissionModel(
            5, [3], zeroAsMissingData=False)
        hmm = MultitrackHmm(emissionModel)
        pairModel = PairEmissionModel(emissionModel, [1.0] *
                                      emissionModel.getNumStates())
        cfg = MultitrackCfg(emissionModel, pairModel)

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
        emissionModel.logProbs = np.log(emProbs)

        hmm = MultitrackHmm(emissionModel)
        pairModel = PairEmissionModel(emissionModel, [1.0] *
                                      emissionModel.getNumStates())
        cfg = MultitrackCfg(emissionModel, pairModel)

        obs = np.array([[0],[0],[1],[0]], dtype=np.int16)
        hmmProb, hmmStates = hmm.decode(obs)
        cfgProb, cfgStates = cfg.decode(obs)
        assert_array_almost_equal(hmmProb, cfgProb)
        assert_array_almost_equal(hmmStates, [0, 0, 1, 0])
        assert_array_almost_equal(hmmStates, cfgStates)

    def testBasicNesting(self):
        # a model with 3 states.  state 0 has a .75 chance of emitting 0
        # state 1 has a 0.95 chance of emitting 1
        # state 2 has a 0.90 chance of emitting 1
        emissionModel = IndependentMultinomialEmissionModel(
            3, [2], zeroAsMissingData=False)
        emProbs = np.zeros((1, 3, 2), dtype=np.float)
        emProbs[0,0] = [0.75, 0.25]
        emProbs[0,1] = [0.05, 0.95]
        emProbs[0,2] = [0.01, 0.90]
        emissionModel.logProbs = np.log(emProbs)
        
        # state 1 is a nested pair state! 
        pairModel = PairEmissionModel(emissionModel, [1.0] *
                                      emissionModel.getNumStates())
        cfg = MultitrackCfg(emissionModel, pairModel, nestStates = [1])


        obs = np.array([[0],[0],[1],[0]], dtype=np.int16)
        cfgProb, cfgStates = cfg.decode(obs)
        # 1 is a pair only state.  no way it should be here
        assert 1 not in cfgStates
        assert_array_equal(cfgStates, [0,0,2,0])

        obs = np.array([[1],[0],[0],[1]], dtype=np.int16)
        cfgProb, cfgStates = cfg.decode(obs)
        assert_array_equal(cfgStates, [2,0,0,2])

        alignment = np.array([[1],[0],[0],[1]], dtype=np.int16)
        cfgProb, cfgStates = cfg.decode(obs, alignmentTrack = alignment,
                                        defAlignmentSymbol=0)
        assert_array_equal(cfgStates, [1,0,0,1])

        alignment = np.array([[1],[0],[0],[2]], dtype=np.int16)
        cfgProb, cfgStates = cfg.decode(obs, alignmentTrack = alignment,
                                        defAlignmentSymbol=0)
        assert_array_equal(cfgStates, [2,0,0,2])
                               
    def testSupervisedLearn(self):
        intervals = readBedIntervals(getTestDirPath("truth.bed"), ncol=4)
        truthIntervals = []
        for i in intervals:
            truthIntervals.append((i[0], i[1], i[2], int(i[3])))

        allIntervals = [(truthIntervals[0][0],
                        truthIntervals[0][1],
                        truthIntervals[-1][2])]
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(3), allIntervals)
        assert len(trackData.getTrackTableList()) == 1

        em = IndependentMultinomialEmissionModel(
            4, trackData.getNumSymbolsPerTrack())
        hmm = MultitrackHmm(em)
        hmm.supervisedTrain(trackData, truthIntervals)
        hmm.validate()
        pairModel = PairEmissionModel(em, [1.0] *
                                      em.getNumStates())
        cfg = MultitrackCfg(em, pairModel, nestStates = [1])

        cfg.supervisedTrain(trackData, truthIntervals)
        cfg.validate()

        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

