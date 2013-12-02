#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os

from teHmm.trackIO import *
from teHmm.track import *
from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase

def getTracksInfoPath(idx = 1):
    if idx == 1:
        return getTestDirPath("tests/data/tracksInfo.xml")
    else:
        return getTestDirPath("tests/data/tracksInfo%d.xml" % idx)

def getTrackList(idx = 1):
    return TrackList(getTracksInfoPath(idx))

def getStatesPath():
    return getTestDirPath("tests/data/states.bed")

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testBedQuery(self):
        ti = getTrackList()
        cbBedPath = ti.getTrackByName("cb").getPath()
        bedData = readTrackData(cbBedPath, "scaffold_1", 3000050, 3000070)
        for i in xrange(10):
            assert int(bedData[i]) == 4
        for i in xrange(11,20):
            assert int(bedData[i]) == 5

    def testBedIntervals(self):
        ti = getTrackList()
        cbBedPath = ti.getTrackByName("cb").getPath()
        intervals = readBedIntervals(cbBedPath, 3, "scaffold_1", 0, 3000060)
        assert len(intervals) == 2
        assert intervals[0] == ("scaffold_1", 0, 2000040)
        assert intervals[1] == ("scaffold_1", 2000040, 3000060)

    def testTrackTable(self):
        for TableType in [IntegerTrackTable]:
            table = TableType(3, "scaffold_1", 3000050, 3000070)
            mappedTable = TableType(3, "scaffold_1", 3000050, 3000070)
            catMap = CategoryMap()
            catMap.update(1000)
            catMap.update(66)
            catMap.update(0)
            catMap.update(1)
            catMap.update(2)
            assert table.getNumTracks() == 3
            assert len(table) == 3000070 - 3000050
            for track in xrange(3):
                table.writeRow(track, [track] * len(table))
                mappedTable.writeRow(
                    track,
                    [catMap.getMap(x) for x in [track] * len(table)])
            assert catMap.getMap(0) == 3
            assert catMap.getMap(1) == 4
            assert catMap.getMap(3) == catMap.getMissingVal()

            for col in xrange(len(table)): 
                for track in xrange(3):                
                    assert table[col][track] == track
                    assert mappedTable[col][track] == track + 3

    def testTrackData(self):
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(),
                                [("scaffold_1", 0, 200004),
                                ("scaffold_1", 2000040, 3000060)])
        assert trackData.getNumTracks() == 3
        trackList = trackData.getTrackList()
        assert len(trackList) == 3
        assert trackList.getTrackByName("cb").name == "cb"
        assert trackList.getTrackByName("cb").getValCol() == 4
        assert trackList.getTrackByName("cb").getBinSize() == 1
        assert trackList.getTrackByName("cb").getDist() == "multinomial"
        assert trackList.getTrackByName("kmer").name == "kmer"
        assert trackList.getTrackByName("blin") == None

        tableList = trackData.getTrackTableList()
        assert len(tableList) == 2
        assert tableList[0].getNumTracks() == 3
        assert tableList[1].getNumTracks() == 3
        assert len(tableList[0]) == 200004
        assert len(tableList[1]) == 3000060 - 2000040

        cbTrack = trackList.getTrackByName("cb")
        
        for i in xrange(len(tableList[0])):
            assert tableList[0][i][cbTrack.number] == 2

        for i in xrange(len(tableList[1])):
            assert tableList[1][i][cbTrack.number] == 3

    def testReadStates(self):
        bedIntervals = readBedIntervals(getStatesPath(), ncol=4)
        for interval in bedIntervals:
            assert int(interval[3]) == 0 or int(interval[3]) == 1

    def testBinaryTrack(self):
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(2),
                                [("scaffold_1", 0, 200004),
                                ("scaffold_Q", 2000040, 3000060)])
        assert trackData.getNumTracks() == 3
        trackList = trackData.getTrackList()
        assert len(trackList) == 3
        assert trackList.getTrackByName("cb").name == "cb"
        assert trackList.getTrackByName("cb").getDist() == "binary"
        assert trackList.getTrackByName("kmer").name == "kmer"
        assert trackList.getTrackByName("blin") == None

        tableList = trackData.getTrackTableList()
        assert len(tableList) == 2
        assert tableList[0].getNumTracks() == 3
        assert tableList[1].getNumTracks() == 3
        assert len(tableList[0]) == 200004
        assert len(tableList[1]) == 3000060 - 2000040

        cbTrack = trackList.getTrackByName("cb")
        
        for i in xrange(len(tableList[0])):
            assert tableList[0][i][cbTrack.number] == 2

        for i in xrange(len(tableList[1])):
            assert tableList[1][i][cbTrack.number] == 1

    def testAlignmentTrack(self):
        trackData = TrackData()
        trackData.loadTrackData(getTracksInfoPath(1),
                                [("scaffold_1", 0, 200004),
                                ("scaffold_Q", 2000040, 3000060)])
        assert trackData.getNumTracks() == 3
        trackList = trackData.getTrackList()
        alignmentTrack = trackList.getAlignmentTrack() is not None
        assert alignmentTrack is not None

        trackTableList = trackData.getTrackTableList()
        assert trackTableList[0].getNumTracks() == 3
        alignmentTableList = trackData.getAlignmentTrackTableList()
        assert alignmentTableList[0].getNumTracks() == 1

        trackData2 = TrackData()
        trackData2.loadTrackData(getTracksInfoPath(2),
                                 [("scaffold_1", 0, 200004),
                                  ("scaffold_Q", 2000040, 3000060)])
        trackList2 = trackData2.getTrackList()
        alignmentTrack2 = trackList2.getAlignmentTrack()
        assert alignmentTrack2 is None


def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

