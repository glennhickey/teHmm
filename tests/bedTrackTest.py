#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os

from teHmm.tracksInfo import TracksInfo
from teHmm.track import *
from teHmm.tests.common import getTestDirPath

class TestCase(unittest.TestCase):

    def setUp(self):
        self.tempFiles = []
        unittest.TestCase.setUp(self)
    
    def tearDown(self):
        for tempFile in self.tempFiles:
            os.remove(tempFile)
        unittest.TestCase.tearDown(self)

    def getTracksInfo(self):
        tracksInfo = TracksInfo()
        pm = dict()
        pm["cb"] = getTestDirPath("tests/data/alyrata_chromBand.bed")
        pm["kmer"] = getTestDirPath("tests/data/alyrata_kmer14.bed")
        pm["te2"] = getTestDirPath("tests/data/alyrata_scaffold1_teii3.7.bed")
        tracksInfo.pathMap = pm
        return tracksInfo

    def testBedQuery(self):
        ti = self.getTracksInfo()
        track = Track("cb", 0, None)
        data = np.zeros((1, 20), np.int)
        bedData = BedTrackData("scaffold_1", 3000050, 3000070, data, track)
        bedData.loadBedInterval(ti.pathMap["cb"], False)
        for i in xrange(10):
            assert data[0,i] == 4
        for i in xrange(11,20):
            assert data[0,i] == 5

    def testCatMap(self):
        ti = self.getTracksInfo()
        track = Track("cb", 0, TrackCategoryMap())
        data = np.zeros((1, 20), np.int)
        bedData = BedTrackData("scaffold_1", 3000050, 3000070, data, track)
        bedData.loadBedInterval(ti.pathMap["cb"], useScore=True,
                                updateMap=True)
        for i in xrange(10):
            assert data[0,i] == 1
        for i in xrange(11,20):
            assert data[0,i] == 2

        

        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

