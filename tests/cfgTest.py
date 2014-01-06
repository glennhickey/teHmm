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
                               
    def testHmmSupervisedLearn(self):
        """ Pretty much copied from the HMM unit test.  We try to recapitualte
        all results with a CFG with no nest states, which should be same as
        HMM"""
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
        # set the fudge to 1 since when the test was written this was
        # hardcoded default
        em = IndependentMultinomialEmissionModel(
            4, trackData.getNumSymbolsPerTrack(),
			  fudge = 1.0)
        hmm = MultitrackHmm(em)
        hmm.supervisedTrain(trackData, truthIntervals)
        hmm.validate()
        pairModel = PairEmissionModel(em, [1.0] *
                                      em.getNumStates())
        # Test validates with neststate just for fun
        cfg = MultitrackCfg(em, pairModel, nestStates = [1])

        cfg.supervisedTrain(trackData, truthIntervals)
        cfg.validate()

        # Then reload as an hmm-equivalent
        cfg = MultitrackCfg(em, pairModel, nestStates = [])

        cfg.supervisedTrain(trackData, truthIntervals)
        cfg.validate()

        # check emissions, they should basically be binary. 
        trackList = cfg.getTrackList()
        emp = np.exp(em.getLogProbs())
        ltrTrack = trackList.getTrackByName("ltr")
        track = ltrTrack.getNumber()
        cmap = ltrTrack.getValueMap()
        s0 = cmap.getMap(None)
        s1 = cmap.getMap(0)
        # we add 1 to all frequencies like emission trainer
        assert_array_almost_equal(emp[track][0][s0], 36. / 37.) 
        assert_array_almost_equal(emp[track][0][s1], 1 - 36. / 37.)
        assert_array_almost_equal(emp[track][1][s0], 1 - 6. / 7.) 
        assert_array_almost_equal(emp[track][1][s1], 6. / 7.)
        assert_array_almost_equal(emp[track][2][s0], 26. / 27.) 
        assert_array_almost_equal(emp[track][2][s1], 1. - 26. / 27.)
        assert_array_almost_equal(emp[track][3][s0], 1. - 6. / 7.)
        assert_array_almost_equal(emp[track][3][s1], 6. / 7.)

        insideTrack = trackList.getTrackByName("inside")
        track = insideTrack.getNumber()
        cmap = insideTrack.getValueMap()
        s0 = cmap.getMap(None)
        s1 = cmap.getMap("Inside")
        assert_array_almost_equal(emp[track][0][s0], 36. / 37.) 
        assert_array_almost_equal(emp[track][0][s1], 1 - 36. / 37.)
        assert_array_almost_equal(emp[track][1][s0], 6. / 7.)
        assert_array_almost_equal(emp[track][1][s1], 1 - 6. / 7.)
        assert_array_almost_equal(emp[track][2][s0], 1. - 26. / 27.)
        assert_array_almost_equal(emp[track][2][s1], 26. / 27.) 
        assert_array_almost_equal(emp[track][3][s0], 6. / 7.)
        assert_array_almost_equal(emp[track][3][s1], 1. - 6. / 7.)

        # crappy check for start probs.  need to test transition too!
        freq = [0.0] * em.getNumStates()
        total = 0.0
        for interval in truthIntervals:
           state = interval[3]
           freq[state] += float(interval[2]) - float(interval[1])
           total += float(interval[2]) - float(interval[1])

        sprobs = cfg.getStartProbs()
        assert len(sprobs) == em.getNumStates()
        for state in xrange(em.getNumStates()):
            assert_array_almost_equal(freq[state] / total, sprobs[state])

        # transition probabilites
        # from eyeball:
        #c	0	5	0   0->0 +4   0->1 +1    0-> +5
        #c	5	10	1   1->1 +4   1->2 +1    1-> +5
        #c	10	35	2   2->2 +24  2->3 +1    2-> +25
        #c	35	40	3   3->3 +4   3->0 +1    3-> +5
        #c	40	70	0   0->0 +29             0-> +19
        realTransProbs = np.array([
            [33. / 34., 1. / 34., 0., 0.],
            [0., 4. / 5., 1. / 5., 0.],
            [0., 0., 24. / 25., 1. / 25.],
            [1. / 5., 0., 0., 4. / 5.]
            ])
            
        tprobs = np.exp(cfg.getLogProbTables()[0])
        assert tprobs.shape == (em.getNumStates(), em.getNumStates(),
                                em.getNumStates())
        for i in xrange(em.getNumStates()):
            for j in xrange(em.getNumStates()):
                fbTot = tprobs[i, i, j]
                if i != j:
                    fbTot += tprobs[i, j, i]
                assert_array_almost_equal(fbTot, realTransProbs[i,j])
        prob, states = cfg.decode(trackData.getTrackTableList()[0])
        for truthInt in truthIntervals:
            for i in xrange(truthInt[1], truthInt[2]):
                # gah, just realized that ltr track is binary, which means
                # ltr states can be either 1 or 3.  need to fix test properly
                # but just relax comparison for now.
                if truthInt[3] == 1 or truthInt[3] == 3:
                    assert states[i] == 1 or states[i] == 3
                else:
                    assert states[i] == truthInt[3]


        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

