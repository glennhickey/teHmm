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
import copy
import ast
import itertools
from collections import defaultdict

from teHmm.trackIO import readBedIntervals, getMergedBedIntervals
from teHmm.common import intersectSize, initBedTool, cleanBedTool, runShellCommand
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLocalTempPath
from teHmm.track import TrackList
from teHmm.bin.compareBedStates import cutOutMaskIntervals

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fill in masked intervals of an hmm prediction "
        "(from teHmmEval.py) with state corresponding to surrounding"
        " intervals.")

    parser.add_argument("tracksXML", help="XML track list (used to id masking"
                        " tracks")
    parser.add_argument("allBed", help="Target scope.  Masked intervals outside"
                        " of these regions will not be included")
    parser.add_argument("inBed", help="TE prediction BED file.  State labels"
                        " should probably be mapped (ie with fitStateNames.py)")
    parser.add_argument("outBed", help="Output BED.  Will be equivalent to"
                        " the input bed except all gaps corresponding to "
                        "masked intervals will be filled")
    parser.add_argument("--maxLen", help="Maximum length of a masked interval"
                        " to fill (inclusive). Use --delMask option with same value"
                        "if running compareBedStates.py after.",
                        type=int, default=sys.maxint)
    parser.add_argument("--default", help="Default label to give to masked "
                        "region if no label can be determined", default="0")
    parser.add_argument("--tgts", help="Only relabel gaps that "
                        "are flanked on both sides by the same state, and this state"
                        " is in this comma- separated list. --default used for other"
                        " gaps.  If not targetst specified then all states checked.",
                        default=None)
    parser.add_argument("--oneSidedTgts", help="Only relabel gaps that "
                        "are flanked on at least one side by a state in this comma-"
                        "separated list --default used for other gaps",
                         default=None)
    parser.add_argument("--onlyDefault", help="Add the default state (--default) no"
                        " no all masked gaps no matter what. ie ignoring all other "
                        "logic", action="store_true", default=False)
    parser.add_argument("--cut", help="Cut out gaps for masked tracks from the input."
                        " By default, the input is expected to come from the HMM "
                        "with mask intervals already absent, and will crash on with"
                        " an assertion error if an overlap is detected.",
                        action="store_true", default=False)

    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    # make sets
    tgtSet = set()
    if args.tgts is not None:
        tgtSet = set(args.tgts.split(","))
    oneSidedTgtSet = set()
    if args.oneSidedTgts is not None:
        oneSidedTgtSet = set(args.oneSidedTgts.split(","))
    assert len(tgtSet.intersection(oneSidedTgtSet)) == 0

    # read the track list
    trackList = TrackList(args.tracksXML)
    maskTracks = trackList.getMaskTracks()

    # read the input bed
    inBed = args.inBed
    if args.cut is True:
        inBed = cutOutMaskIntervals(inBed, -1, args.maxLen + 1, args.tracksXML)
    inputIntervals = readBedIntervals(inBed, ncol = 4, sort = True)
    if args.cut is True:
        runShellCommand("rm -f %s" % inBed)
    if len(maskTracks) == 0 or len(inputIntervals) == 0:
        runShellCommand("cp %s %s" % (args.inBed, args.outBed))
        logger.warning("No mask tracks located in %s or"
                       " %s empty" % (args.tracksXML, args.inBed))
        return 0


    # make a temporary, combined, merged masking bed file
    tempMaskBed = getLocalTempPath("Temp_mb", ".bed")
    for maskTrack in maskTracks:
        assert os.path.isfile(maskTrack.getPath())
        runShellCommand("cat %s | setBedCol.py 3 mask >> %s" % (maskTrack.getPath(),
                                                                tempMaskBed))
    maskedIntervals = getMergedBedIntervals(tempMaskBed, sort = True)
    resolvedMasks = 0

    if len(inputIntervals) == 0:
        logger.warning("No mask tracks located in %s" % args.tracksXML)
        return
    inputIdx = 0
    rightFlank = inputIntervals[inputIdx]

    tempOutMask = getLocalTempPath("Temp_om", ".bed")
    tempOutMaskFile = open(tempOutMask, "w")

    for maskIdx, maskInterval in enumerate(maskedIntervals):
        if maskInterval[2] - maskInterval[1] > args.maxLen:
            continue
        # find candidate right flank
        while rightFlank < maskInterval:
            if inputIdx == len(inputIntervals) - 1:
                rightFlank = None
                break
            else:
                inputIdx += 1
                rightFlank = inputIntervals[inputIdx]

        # candidate left flank
        leftFlank = None
        if inputIdx > 0:
            leftFlank = inputIntervals[inputIdx - 1]

        # identify flanking states if the intervals perfectly abut
        leftState = None
        if leftFlank is not None:
            if leftFlank[0] == maskInterval[0] and leftFlank[2] == maskInterval[1]:
                leftState = str(leftFlank[3])
            else:
                assert intersectSize(leftFlank, maskInterval) == 0
        rightState = None
        if rightFlank is not None:
            if rightFlank[0] == maskInterval[0] and rightFlank[1] == maskInterval[2]:
                rightState = str(rightFlank[3])
            else:
                assert intersectSize(rightFlank, maskInterval) == 0
            
        # choose a state for the mask interval
        maskState = str(args.default)
        if args.onlyDefault is True:
            pass
        elif leftState is not None and leftState == rightState:
            if len(tgtSet) == 0 or leftState in tgtSet:
                maskState = leftState
        elif leftState in oneSidedTgtSet:
            maskState = leftState
        elif rightState in oneSidedTgtSet:
            maskState = rightState
        
        # write our mask interval
        tempOutMaskFile.write("%s\t%d\t%d\t%s\n" % (maskInterval[0], maskInterval[1],
                                                    maskInterval[2], maskState))

    
    tempOutMaskFile.close()    
    tempMergePath1 = getLocalTempPath("Temp_mp", ".bed")
    tempMergePath2 = getLocalTempPath("Temp_mp", ".bed")
    runShellCommand("cp %s %s ; cat %s >> %s" % (args.inBed, tempMergePath1,
                                                 tempOutMask, tempMergePath1))
    runShellCommand("cat %s | sortBed > %s" % (tempMergePath1, tempMergePath2))
    tempScopePath = getLocalTempPath("temp_all", ".bed")
    runShellCommand("mergeBed -i %s |sortBed > %s" % (args.allBed, tempScopePath))
    runShellCommand("intersectBed -a %s -b %s > %s" % (tempMergePath2, tempScopePath,
                                                       args.outBed))

    runShellCommand("rm -f %s" % " ".join([tempMaskBed, tempOutMask, tempMergePath1,
                                      tempMergePath2, tempScopePath]))
    cleanBedTool(tempBedToolPath)


    
if __name__ == "__main__":
    sys.exit(main())
