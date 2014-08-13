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

from teHmm.track import TrackList, TrackData
from teHmm.trackIO import readTrackData, readBedIntervals
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand, getLocalTempPath

filTok = "F\tll"

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Set the score column of each bed interval in input to "
        "(MODE, BINNED) average value of the intersection region in another track). "
        "Can be used, for instance, to assign a copy number of each RepeatModeler "
        "prediction...")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("inBed", help="BED file to annotate")
    parser.add_argument("track", help="Track to use for annotation")
    parser.add_argument("outBed", help="Path for output, annotated BED file")
    parser.add_argument("--name", help="Set ID field (column 4 instead of 5)",
                        action="store_true", default=False)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    # read the tracks list
    trackList = TrackList(args.tracksInfo)
    track = trackList.getTrackByName(args.track)
    if track is None:
        raise RuntimeError("Can't find track %s" % args.track)
    # make temporary tracks list with just our track so we can keep using
    # tracks list interface but not read unecessary crap.
    singleListPath = getLocalTempPath("Temp_secScore", ".bed")
    trackList.trackList = [track]
    trackList.saveXML(singleListPath)

    obFile = open(args.outBed, "w")

    # trackData interface not so great at cherry picking intervals.
    # need to merge them up and use segmentation interface    
    filledIntervals, mergedIntervals = fillGaps(args.inBed)

    # read track into trackData
    trackData = TrackData()
    logger.info("loading track %s" % singleListPath)
    trackData.loadTrackData(singleListPath, mergedIntervals,
                            segmentIntervals=filledIntervals)

    # finally, write the annotation
    writeAnnotatedIntervals(trackData, filledIntervals, mergedIntervals, obFile,
                             args)

    runShellCommand("rm -f %s" % singleListPath)
    obFile.close()
    cleanBedTool(tempBedToolPath)

def fillGaps(inBed):
    """ Make two interval sets from a given bed file:
      filledIntervals: Set of intervals with intervals added between consecutive
                       intervals on same seq (ala addBedGaps.y)
      mergedIntervals: Set of intervals spanning each continuous region from
                       above (ala getMergeItnervals)
    probably reimplementing stuff but oh well """
    filledIntervals = []
    mergedIntervals = []
    intervals = readBedIntervals(inBed, ncol=4, sort=True)
    if len(intervals) == 0:
        return [], []
    
    prevInterval = None
    for interval in intervals:
        if prevInterval is not None and prevInterval[0] == interval[0] and\
          prevInterval[2] != interval[1]:
          # update fill for discontinuity
          assert prevInterval[2] < interval[1]
          filledIntervals.append((interval[0], prevInterval[2], interval[1],
                                     filTok))
        if prevInterval is None or prevInterval[0] != interval[0]:
            # update merge for new sequence
            mergedIntervals.append(interval)
        else:
            # extend merge for same sequence
            mergedIntervals[-1] = (mergedIntervals[-1][0],
                                   mergedIntervals[-1][1],
                                   interval[2],
                                   mergedIntervals[-1][3])

        # update fill with current interval
        filledIntervals.append(interval)
        prevInterval = interval
        
    return filledIntervals, mergedIntervals
        
        
def writeAnnotatedIntervals(trackData, filledIntervals, mergedIntervals, outBed,
                            args):
    """ for each non-fill itnerval in filledIntervals, write out an interval
    in the output with the score taken from the trackData"""
    track = trackData.getTrackList().getTrackByName(args.track)
    assert track is not None
    valMap = track.getValueMap()
    iCount = 0
    for i, trackTable in enumerate(trackData.getTrackTableList()):
        assert trackTable.getChrom() == mergedIntervals[i][0]
        assert trackTable.getStart() == mergedIntervals[i][1]
        assert trackTable.getEnd() == mergedIntervals[i][2]
        chrom = trackTable.getChrom()

        for j, segOffset in enumerate(trackTable.getSegmentOffsets()):
            start = trackTable.getStart() + segOffset
            end = start + trackTable.getSegmentLength(j)
            name = filledIntervals[iCount][3]
            assert chrom == filledIntervals[iCount][0]
            assert start == filledIntervals[iCount][1]
            assert end == filledIntervals[iCount][2]
            if filledIntervals[iCount][3] != filTok:
                mappedVal = trackTable[j][0]
                val = valMap.getMapBack(mappedVal)
                if args.name is False:
                    if name is None or len(name) == 0:
                        name = val
                    outBed.write("%s\t%s\t%s\t%s\t%s\n" % (chrom, start, end,
                                                           name, val))
                else:
                    outBed.write("%s\t%s\t%s\t%s\n" % (chrom, start, end, val))
            iCount += 1
    assert iCount == len(filledIntervals)
            
     
if __name__ == "__main__":
    sys.exit(main())
