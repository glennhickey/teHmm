#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import numpy as np
import math
import copy
import scipy

from teHmm.track import TrackList, TrackData
from teHmm.trackIO import readTrackData, getMergedBedIntervals
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Produce a bed file of genome segments which are atomic"
        " elements with resepect to the hmm. ie each segment emits a single"
        " state.")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("allBed", help="Bed file spanning entire genome")
    parser.add_argument("outBed", help="Output segments")
    parser.add_argument("--thresh", help="Number of tracks that can change "
                        "before a new segment formed.  Increasing this value"
                        " increases the expected lengths of output segments",
                        type=int, default=0)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    # read query intervals from the bed file
    logger.info("loading training intervals from %s" % args.allBed)
    mergedIntervals = getMergedBedIntervals(args.allBed, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.allBed)

    # read the tracks, while intersecting them with the query intervals
    logger.info("loading tracks %s" % args.tracksInfo)
    trackData = TrackData()
    trackData.loadTrackData(args.tracksInfo, mergedIntervals)

    # segment the tracks
    segmentTracks(trackData, args)

    cleanBedTool(tempBedToolPath)

def segmentTracks(trackData, args):
    """ produce a segmentation of the data based on the track values. start with
    a hacky prototype..."""
    oFile = open(args.outBed, "w")

    trackTableList = trackData.getTrackTableList()
    # for every non-contiguous region
    for trackTable in trackTableList:
        interval = [trackTable.getChrom(), trackTable.getStart(),
                    trackTable.getEnd()]
        curInterval = (interval[0], interval[1], interval[1] + 1)
        intervalLen = interval[2] - interval[1]
        # scan each column (base) in region, and write new bed segment
        # if necessary (ie too much change in track values)
        for i in xrange(1, intervalLen):
            if isNewSegment(trackTable, i, args) is True:
                oFile.write("%s\t%d\t%d\n" % (interval[0], interval[1],
                                              interval[1] + i))
                interval[1] = interval[1] + i
                interval[2] = interval[1] + 1
        # write last segment
        if interval[1] < trackTable.getEnd():
            oFile.write("%s\t\%d\t%d\n" % (interval[0], interval[1],
                                            trackTable.getEnd()))        
    
    oFile.close()

def isNewSegment(trackTable, i, args):
    """ may be necessary to cythonize this down the road """
    assert i > 0
    assert i < len(trackTable)

    # faster to just call pdist(trackTable[i-1:i], 'hamming')? 
    col = trackTable[i]
    prev = trackTable[i-1]
    difCount = 0
    for j in xrange(len(col)):
        if col[j] != prev[j]:
            difCount += 1

    return difCount > args.thresh
                
if __name__ == "__main__":
    sys.exit(main())
