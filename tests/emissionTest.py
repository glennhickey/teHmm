#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import numpy as np
import math

from teHmm.emissions import IndependentMultinomialEmissionModel
from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfo

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def createSimpleModel1(self):
        #two states, 1 track, 2 symbols
        em = IndependentMultinomialEmissionModel(numStates=2,
                                                 numSymbolsPerTrack = [2])
        state1 = [math.log(0.2), math.log(0.8)]
        state2 = [math.log(0.5), math.log(0.5)]
        track1 = [state1, state2]
        em.initParams([track1])
        return em

    def createSimpleModel2(self):
        #2 states, 2 tracks, 2 symbols in track 0, and 3 symbols in track 2
        em = IndependentMultinomialEmissionModel(numStates = 2,
                                                 numSymbolsPerTrack=[2, 3])
        state1track1 = [math.log(0.2), math.log(0.8)]
        state2track1 = [math.log(0.5), math.log(0.5)]
        state1track2 = [math.log(0.1), math.log(0.3), math.log(0.6)]
        state2track2 = [math.log(0.7), math.log(0.1), math.log(0.2)]
 
        track1 = [state1track1, state2track1]
        track2 = [state1track2, state2track2]
        em.initParams([track1, track2])
        return em
        
    def testSingleObs(self):
        em = self.createSimpleModel1()
        assert em.singleLogProb(0, [0]) == math.log(0.2)
        assert em.singleLogProb(0, [1]) == math.log(0.8)
        assert em.singleLogProb(1, [0]) == math.log(0.5)
        assert em.singleLogProb(1, [1]) == math.log(0.5)

        em = self.createSimpleModel2()
        assert em.singleLogProb(0, [0, 1]) == math.log(0.2) + math.log(0.3)
        assert em.singleLogProb(0, [1, 0]) == math.log(0.8) + math.log(0.1)
        assert em.singleLogProb(1, [1, 2]) == math.log(0.5) + math.log(0.2)


    def testAllObs(self):
        em = self.createSimpleModel1()
        
        truth = np.array([[math.log(0.2), math.log(0.5)],
                          [math.log(0.2), math.log(0.5)],
                          [math.log(0.8), math.log(0.5)]])
        obs = np.array([[0], [0], [1]])
        assert np.array_equal(em.allLogProbs(obs), truth)

    def testInitStats(self):
        em = self.createSimpleModel1()
        obsStats = em.initStats()
        assert len(obsStats) == 1
        assert obsStats[0].shape == (2, 2)

        em = self.createSimpleModel2()
        obsStats = em.initStats()
        assert len(obsStats) == 2
        assert obsStats[0].shape == (2, 2)
        assert obsStats[1].shape == (2, 3)

    def testAccumulateStats(self):
        em = self.createSimpleModel1()
        obsStats = em.initStats()
        obs = np.array([[0], [0], [1]])
        posteriors = np.array([[0.01, 0.02], [0.01, 0.02], [0.3, 0.4]])
        em.accumulateStats(obs, obsStats, posteriors)
        assert obsStats[0][0][0] == 0.01 + 0.01
        assert obsStats[0][1][0] == 0.02 + 0.02
        assert obsStats[0][0][1] == 0.3
        assert obsStats[0][1][1] == 0.4


def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

