#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import itertools
import copy
import numpy as np

from teHmm.common import runShellCommand
from teHmm.common import runParallelShellCommands
from teHmm.track import TrackList
from pybedtools import BedTool, Interval
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLogLevelString, setLogLevel
from teHmm.bin.compareBedStates import extractCompStatsFromFile
from teHmm.track import TrackData, INTEGER_ARRAY_TYPE
from teHmm.trackIO import readBedIntervals, getMergedBedIntervals

""" Dump some track data from the XML file to an ASCII matrix
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Write track data into ASCII dump.  Row i corresponds"
        " to the ith position found when scanning query BED IN SORTED ORDER."
        "Column j corresponds to the jth track in the XML file. --map option"
        " used to write internal integer format used by HMM.  Unobserved values"
        " written as \"None\" if default attribute not specified or track not"
        " binary.  Rounding can occur if scaling parameters present.\n\n"
        "IMPORTANT: values stored in 8bit integers internally.  Any track with"
        " more than 256 different values will get clamped (with a warning)")

    parser.add_argument("tracks", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("query", help="BED region(s) to dump. SCANNED IN"
                        " SORTED ORDER")
    parser.add_argument("output", help="Path of file to write output to")
    parser.add_argument("--map", help="Apply name mapping, including"
                        " transformation specified in scale, logScale"
                        ", etc. attributes, that HMM uses internally"
                        ". Important to note that resulting integers"
                        " are just unique IDs.  ID_1 > ID_2 does not"
                        " mean anything", action="store_true",
                        default=False)
    parser.add_argument("--segment", help="Treat each interval in query"
                        " as a single segment (ie with only one data point)"
                        ".  In this case, query should probably have been"
                        " generated with segmentTracks.py",
                        action="store_true",
                        default=False)
    parser.add_argument("--noPos", help="Do not print genomic position"
                        " (first 2 columnts)", action="store_true",
                        default=False)
    parser.add_argument("--noMask", help="Ignore mask tracks",
                        default=False, action="store_true")
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)

    # make sure output writeable
    outFile = open(args.output, "w")

    # need to remember to fix this, disable as precaution for now
    assert args.noMask is True or args.segment is False
    
    # read query intervals from the bed file
    logger.info("loading query intervals from %s" % args.query)
    mergedIntervals = getMergedBedIntervals(args.query, ncol=3)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.query)

    # read the segment intervals from the (same) bed file
    segIntervals = None
    if args.segment is True:
        logger.info("loading segment intervals from %s" % args.query)
        segIntervals = readBedIntervals(args.query, sort=True)

    # read all data from track xml
    logger.info("loading tracks %s" % args.tracks)
    trackData = TrackData()
    trackData.loadTrackData(args.tracks, mergedIntervals,
                            segmentIntervals=segIntervals,
                            applyMasking = not args.noMask)

    # dump the data to output
    dumpTrackData(trackData, outFile, args.map, not args.noPos)
    outFile.close()


def dumpTrackData(trackData, outFile, doMapping, doPosition):
    """ do the dump"""
    
    # make a list to category tables for convenience
    # as it turns out that getMapBack() is inordinately expensive. 
    mapTableList = []
    for track in trackData.getTrackList():
        table = dict()
        vm = track.getValueMap()
        for i in xrange(np.iinfo(INTEGER_ARRAY_TYPE).max + 1):
            table[i] = vm.getMapBack(i)
        mapTableList.append(table)
        
    # scan column by column
    for trackTable in trackData.getTrackTableList():
        segmentOffsets = trackTable.getSegmentOffsets()
        maskOffsets = trackTable.getMaskRunningOffsets()
        for pos in xrange(len(trackTable)):
            column = trackTable[pos]
            if doMapping is False:
                # since we don't want the internal mapped values, we
                # need to map them back 
                mappedCol = [mapTableList[x][column[x]] for x in
                             xrange(len(column))]
                column = mappedCol
            column = [str(x) for x in column]
            if doPosition is True:
                currentPos = pos
                if segmentOffsets != None:
                    currentPos = segmentOffsets[pos]
                if maskOffsets is not None:
                    currentPos += maskOffsets[pos]
                outFile.write("%s,%d," % (trackTable.getChrom(),
                              trackTable.getStart() + currentPos))
            outFile.write(",".join(column))
            outFile.write("\n")

if __name__ == "__main__":
    sys.exit(main())
