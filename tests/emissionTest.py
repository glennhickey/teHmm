#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import numpy as np
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal
import itertools

from teHmm.emission import IndependentMultinomialEmissionModel
from teHmm.track import TrackData
from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfoPath

def getBedStates():
    return [
        ("scaffold_1", 0,10, 0),
        ("scaffold_1", 10, 30, 1),
        ("scaffold_1", 1000, 1040, 0),
        ("scaffold_1", 100000, 100020, 0),
        ("scaffold_2", 6000120, 6000122, 1)]

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def createSimpleModel1(self):
        #two states, 1 track, 2 symbols
        em = IndependentMultinomialEmissionModel(numStates=2,
                                                 numSymbolsPerTrack = [2])
        state1 = [0.2, 0.8]
        state2 = [0.5, 0.5]
        track1 = [state1, state2]
        em.initParams([track1])
        return em

    def createSimpleModel2(self):
        #2 states, 2 tracks, 2 symbols in track 0, and 3 symbols in track 2
        em = IndependentMultinomialEmissionModel(numStates = 2,
                                                 numSymbolsPerTrack=[2, 3])
        state1track1 = [0.2, 0.8]
        state2track1 = [0.5, 0.5]
        state1track2 = [0.1, 0.3, 0.6]
        state2track2 = [0.7, 0.1, 0.2]
 
        track1 = [state1track1, state2track1]
        track2 = [state1track2, state2track2]
        em.initParams([track1, track2])
        return em
        
    def testSingleObs(self):
        em = self.createSimpleModel1()
        assert em.singleLogProb(0, [1]) == math.log(0.2)
        assert em.singleLogProb(0, [2]) == math.log(0.8)
        assert em.singleLogProb(1, [1]) == math.log(0.5)
        assert em.singleLogProb(1, [2]) == math.log(0.5)
        assert em.singleLogProb(1, [0]) == 0

        em = self.createSimpleModel2()
        assert em.singleLogProb(0, [1, 2]) == math.log(0.2) + math.log(0.3)
        assert em.singleLogProb(0, [2, 1]) == math.log(0.8) + math.log(0.1)
        assert em.singleLogProb(1, [2, 3]) == math.log(0.5) + math.log(0.2)


    def testAllObs(self):
        em = self.createSimpleModel1()
        
        truth = np.array([[math.log(0.2), math.log(0.5)],
                          [math.log(0.2), math.log(0.5)],
                          [math.log(0.8), math.log(0.5)]])
        obs = np.array([[1], [1], [2]])
        assert np.array_equal(em.allLogProbs(obs), truth)

    def testInitStats(self):
        em = self.createSimpleModel1()
        obsStats = em.initStats()
        assert len(obsStats) == 1
        assert obsStats[0].shape == (2, 2+1)

        em = self.createSimpleModel2()
        obsStats = em.initStats()
        assert len(obsStats) == 2
        assert obsStats[0].shape == (2, 2+1)
        assert obsStats[1].shape == (2, 3+1)

    def testAccumultestats(self):
        em = self.createSimpleModel1()
        obsStats = em.initStats()
        obs = np.array([[0], [0], [1]])
        posteriors = np.array([[0.01, 0.02], [0.01, 0.02], [0.3, 0.4]])
        em.accumulateStats(obs, obsStats, posteriors)
        assert obsStats[0][0][0] == 0.01 + 0.01
        assert obsStats[0][1][0] == 0.02 + 0.02
        assert obsStats[0][0][1] == 0.3
        assert obsStats[0][1][1] == 0.4

    def testSupervisedTrain(self):
        bedIntervals = getBedStates()
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(), bedIntervals)
        assert len(trackData.getTrackTableList()) == len(bedIntervals)
        em = IndependentMultinomialEmissionModel(
            2, trackData.getNumSymbolsPerTrack())
        em.supervisedTrain(trackData, bedIntervals)

        # count frequency of symbols for a given track
        for track in xrange(3):            
            counts = [dict(), dict()]
            totals = [0, 0]

            # init to ones like we do in emisisonModel
            for i in em.getTrackSymbols(track):
                counts[0][i] = 1
                counts[1][i] = 1
                totals[0] += 1
                totals[1] += 1
                
            for tableIdx, table in enumerate(trackData.getTrackTableList()):
                state = bedIntervals[tableIdx][3]
                count = counts[state]
                for i in xrange(len(table)):
                    val = table[i][track]
                    if val in count:
                        count[val] += 1
                        totals[state] += 1

            # compute track frequency from model by marginalizing and compare
            for state in xrange(2):
                for val in counts[state]:
                    frac = float(counts[state][val]) / float(totals[state])
                    prob = 0.0
                    for val3d in em.getSymbols():
                        if val3d[track] == val:
                            prob += np.exp(em.singleLogProb(state, val3d))
                    assert_array_almost_equal(prob, frac)

        


def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

