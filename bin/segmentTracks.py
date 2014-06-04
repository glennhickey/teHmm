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
        " state.  Output intervals are assigned name 0 1 0 1 etc.")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("allBed", help="Bed file spanning entire genome")
    parser.add_argument("outBed", help="Output segments")
    parser.add_argument("--thresh", help="Number of tracks that can change "
                        "before a new segment formed.  Increasing this value"
                        " increases the expected lengths of output segments",
                        type=int, default=0)
    parser.add_argument("--cutTracks", help="Create a new segment if something"
                        " changes in one of these tracks (as specified by "
                        "comman-separated list), overriding --thresh options"
                        " if necessary.  For example, --cutTracks tsd,chaux"
                        " would invoke a new segment everytime the value at"
                        "either of these tracks changed", default=None)
    parser.add_argument("--comp", help="Strategy for comparing columns for the "
                        "threshold cutoff.  Options are [first, prev], where"
                        " first compares with first column of segment and "
                        "prev compares with column immediately left",
                        default="prev")
    parser.add_argument("--ignore", help="Comma-separated list of tracks to "
                        "ignore (the FASTA DNA sequence would be a good "
                        "candidate", default=None)
    parser.add_argument("--maxLen", help="Maximum length of a segment (<= 0 means"
                        " no max length applied",
                        type=int, default=0)
    parser.add_argument("--fixLen", help="Just make segments of specifed fixed "
                        "length ignoring other parameters and logic (<= 0 means"
                        " no fixed length applied",
                        type=int, default=0)
    parser.add_argument("--stats", help="Write some statistics to specified "
                        "file. Of the form <trackName> <Diff> <DiffPct> "
                        " where <Diff> is the number of times a track differs"
                        " between two consecutive segments, and <DiffPct> "
                        " is the average perecentage of all such differences "
                        "accounted for by the track", default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    if args.comp != "first" and args.comp != "prev":
        raise RuntimeError("--comp must be either first or prev")
    if args.fixLen > 0:
        if args.ignore is not None or args.maxLen > 0 \
           or args.thresh != 0 or args.cutTracks is not None:
            raise RuntimeError("--fixLen not compatible with other options")
        
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

    # process the --cutTracks option
    trackList = trackData.getTrackList()
    cutList = np.zeros((len(trackList)), np.int)
    if args.cutTracks is not None:
        cutNames = args.cutTracks.split(",")
        for name in cutNames:
            track = trackList.getTrackByName(name)
            if track is None:
                raise RuntimeError("cutTrack %s not found" % name)
            trackNo = track.getNumber()
            assert trackNo < len(cutList)
            cutList[trackNo] = 1
    args.cutList = cutList

    # process the --ignore option
    ignoreList = np.zeros((len(trackList)), np.int)
    if args.ignore is not None:
        ignoreNames = args.ignore.split(",")
        for name in ignoreNames:
            track = trackList.getTrackByName(name)
            if track is None:
                raise RuntimeError("ignore track %s not found" % name)
            trackNo = track.getNumber()
            assert trackNo < len(ignoreList)
            ignoreList[trackNo] = 1
            if args.cutList[trackNo] == 1:
                raise RuntimeError("Same track (%s) cant be cut and ignored" %
                                  name)
    args.ignoreList = ignoreList

    # segment the tracks
    stats = dict()
    segmentTracks(trackData, args, stats)
    writeStats(trackData, args, stats)

    cleanBedTool(tempBedToolPath)

def segmentTracks(trackData, args, stats):
    """ produce a segmentation of the data based on the track values. start with
    a hacky prototype..."""
    oFile = open(args.outBed, "w")
    prevMode = args.comp == "prev"

    trackTableList = trackData.getTrackTableList()
    # for every non-contiguous region
    for trackTable in trackTableList:
        start = trackTable.getStart()
        end = trackTable.getEnd()
        chrom = trackTable.getChrom()
        interval = [chrom, start, start + 1]
        intervalLen = end - start
        # scan each column (base) in region, and write new bed segment
        # if necessary (ie too much change in track values)
        count = 0
        pi = 0
        curLen = 0
        for i in xrange(1, intervalLen):
            curLen += 1
            if isNewSegment(trackTable, pi, i, curLen, args, stats) is True:
                oFile.write("%s\t%d\t%d\t%d\n" % (chrom, interval[1], start + i,
                                                  count % 2))
                interval[1] = start + i
                interval[2] = interval[1] + 1
                count += 1
                pi = i
                curLen = 0
            if prevMode is True:
                pi = i
        # write last segment
        if interval[1] < trackTable.getEnd():
            oFile.write("%s\t%d\t%d\t%d\n" % (chrom, interval[1],
                                              trackTable.getEnd(),
                                              count % 2))        
    
    oFile.close()

def isNewSegment(trackTable, pi, i, curLen, args, stats):
    """ may be necessary to cythonize this down the road """
    assert i > 0
    assert i < len(trackTable)
    assert pi >= 0
    assert pi < i

    if args.fixLen > 0:
        return curLen >= args.fixLen
    if args.maxLen > 0 and curLen >= args.maxLen:
        return True

    # faster to just call pdist(trackTable[i-1:i], 'hamming')? 
    col = trackTable[i]
    prev = trackTable[pi]
    difCount = 0
    cutTrackFound = False
    for j in xrange(len(col)):
        if args.ignoreList[j] == 0 and col[j] != prev[j]:
            difCount += 1
            # cutList track is different
            if args.cutList[j] == 1:
                cutTrackFound = True
                if args.stats is None:
                    return True

    retVal = cutTrackFound is True or difCount > args.thresh
    
    # second pass (with difCount in hand) for stats if wanted
    if args.stats is not None and retVal is True:
        for j in xrange(len(col)):
            if args.ignoreList[j] == 0 and col[j] != prev[j]:
                if j not in stats:
                    stats[j] = (0, 0)
                stats[j] = (stats[j][0] + 1, stats[j][1] + 1. / float(difCount))
                 

    return retVal

def writeStats(trackData, args, stats):
    """ write the cutting statistics if wanted"""
    if args.stats is not None:
        trackList = trackData.getTrackList()
        statFile = open(args.stats, "w")
        for trackNo, stat in stats.items():
            trackName = trackList.getTrackByNumber(trackNo).getName()
            difCount = stat[0]
            avgDifPct = float(stat[1]) / float(stat[0])
            statFile.write("%s\t%d\t%f\n" % (trackName, difCount, avgDifPct))
        statFile.close()
        
                
if __name__ == "__main__":
    sys.exit(main())
