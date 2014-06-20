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
from teHmm.common import initBedTool, cleanBedTool, getLocalTempPath
from teHmm.track import TrackList
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLogLevelString, setLogLevel
from teHmm.trackIO import getMergedBedIntervals, readBedIntervals



""" Merge some tracks together naively to use as a baseline for benchmarking
the HMM.  Might be interesting to eventually do something more clever
in terms of overlap filtering... Current approach is:

for each track:
  use confusion matrix to rename with accumulated tracks (if exists)
  merge with accumulated tracks

remove overlaps (using overly-simple removeBedOverlaps.py)
add "Outside States"

NOTE: just realized that order tracks are added is pretty important.  not quite
what to do about it now.  maybe change fitting to be more symmetric -- ie try
fitting in both directions and choosing the best mapped?  In any case,
use with trackSelection script should help cover this a bit... 

"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Combine a bunch of non-numeric BED tracks into"
        " single file using fitStateNames.py to try to keep names "
        "consistent.  Idea is to be used as baseline to compare"
        " hmm to (via base-by-base statistics, primarily, since"
        " this procedure could induce some fragmentation)")

    parser.add_argument("tracksXML", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("regionBed", help="BED file representing "
                        "target region (best if whole genome)")
    parser.add_argument("outBed", help="Output bed")
    parser.add_argument("--tracks", help="Comma-separated list of "
                        "track names to use.  All tracks will be"
                        " used by default", default=None)
    parser.add_argument("--outside", help="Name to give non-annotated"
                        "regions", default="Outside")
    parser.add_argument("--fitThresh", help="Min map percentage (0,1)"
                        " in order to rename (see --qualThresh option"
                        "of fitStateNames.py", type=float,
                        default=0.5)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    inputTrackList = TrackList(args.tracksXML)
    iter = 0

    # get regionBed where all intervals are merged when possible
    regionIntervals = getMergedBedIntervals(args.regionBed, sort=True)
    tempRegionPath = getLocalTempPath("Temp", "_reg.bed")
    tempRegionFile = open(tempRegionPath, "w")
    for interval in regionIntervals:
        tempRegionFile.write("\t".join([str(x) for x in interval]) + "\n")
    tempRegionFile.close()

    # accumulate tracks in temp file
    tempOutPath = getLocalTempPath("Temp", "_out.bed")
    
    for track in inputTrackList:
        if track.shift is not None or track.scale is not None or\
          track.logScale is not None or track.dist == "gaussian" or\
          os.path.splitext(track.getPath())[1].lower() != ".bed":
          logger.warning("Skipping numeric track %s" % track.getName())
        elif args.tracks is None or track.getName() in args.tracks.split(","):
            combineTrack(track, tempOutPath, tempRegionPath, iter, args)
            iter += 1

    # nothing got written, make everything outside
    if iter == 0:
        tempOutFile = open(tempOutPath, "w")
        for interval in regionIntervals:
            tempOutFile.write("%s\t%s\t%s\t%s\n" % (interval[0], interval[1],
                                                   interval[2], args.outside))
        tempOutFile.close()

    runShellCommand("mv %s %s" % (tempOutPath, args.outBed))
                
    cleanBedTool(tempBedToolPath)

def combineTrack(track, outPath, tempRegionPath, iter, args):
    """ merge track with outPath """

    # make sure track is of form chrom start end state
    tempColPath = getLocalTempPath("Temp", "_col.bed")
    tempColFile = open(tempColPath, "w")
    vc = track.getValCol() + 1
    if track.getDist() == "binary":
        assert track.getName() != args.outside
        vc = 3
    bedIntervals = readBedIntervals(track.getPath(), vc,
                                    sort = True)
    for bedInterval in bedIntervals:
        outStr = "\t".join([str(x) for x in bedInterval])
        if track.getDist() == "binary":
            # state name = track name for binary track
            outStr += "\t%s" % track.getName()
        outStr += "\n"
        tempColFile.write(outStr)
    tempColFile.close()

    # intersect the target region
    tempIntersectPath = getLocalTempPath("Temp", "_int.bed")
    runShellCommand("intersectBed -a %s -b %s > %s" % (
        tempColPath, tempRegionPath, tempIntersectPath))

    # add the outside states
    tempGappedPath = getLocalTempPath("Temp", "_gap.bed")
    runShellCommand("addBedGaps.py --state %s %s %s %s" % (
        args.outside, tempRegionPath, tempIntersectPath, tempGappedPath))

    # fit the names with previous interations' result
    tempFitPath = getLocalTempPath("Temp", "_fit.bed")
    if iter == 0:
        runShellCommand("cp %s %s" % (tempGappedPath, tempFitPath))
    else:
        runShellCommand("fitStateNames.py %s %s %s --qualThresh %f --ignoreTgt %s" % (
            outPath, tempGappedPath, tempFitPath, args.fitThresh, args.outside))

    # now merge into outPath
    runShellCommand("cat %s >> %s" % (tempFitPath, outPath))
    runShellCommand("removeBedOverlaps.py %s > %s" % (outPath, tempColPath))
    runShellCommand("mv %s %s" % (tempColPath, outPath))

    # clean crap (note tempCol should already be gone)
    runShellCommand("rm -f %s" % tempColPath)
    runShellCommand("rm -f %s" % tempIntersectPath)
    runShellCommand("rm -f %s" % tempGappedPath)
    runShellCommand("rm -f %s" % tempFitPath)
    
if __name__ == "__main__":
    sys.exit(main())
