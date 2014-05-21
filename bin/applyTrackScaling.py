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
from teHmm.trackIO import readTrackData, getMergedBedIntervals
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand

"""

Generate a directory with a copy of all the track files from a tracks XML that have been
scaled (if applicable) according to the scaling parameters.  The scaled tracks can
then be loaded into the Browser to help with debugging....

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Automatically set the scale attributes of numeric tracks"
        " within a given tracks.xml function using some simple heuristics. ")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("chromSizes", help="2-column chrom sizes file as needed"
                        " by bedGraphToBigWig")
    parser.add_argument("queryBed", help="Region(s) to apply scaling to")
    parser.add_argument("outputDir", help="Output directory")
    parser.add_argument("--tracks", help="Comma-separated list of tracks "
                        "to process. If not set, all tracks with a scaling"
                        " attribute are processed", default=None)
    parser.add_argument("--skip", help="Comma-separated list of tracks to "
                        "skip.", default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()


    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    trackNames = []
    if args.tracks is not None:
        trackNames = args.tracks.split(",")
    skipNames = []
    if args.skip is not None:
        skipNames = args.skip.split(",")
    
    mergedIntervals = getMergedBedIntervals(args.queryBed)

    trackData = TrackData()
    trackData.loadTrackData(args.tracksInfo, mergedIntervals)
    trackList = trackData.getTrackList()

    for track in trackList:
        if track.getName() not in skipNames and\
          (track.getName() in trackNames or len(trackNames) == 0):
          if track.getScale() is not None or\
            track.getLogScale() is not None or\
            track.getShift() is not None or\
            track.getDelta() is True:
            writeScaledTrack(trackData, track, args)

    cleanBedTool(tempBedToolPath)

def writeScaledTrack(trackData, track, args):
    """ Go base-by-base, writing the unscaled value to the output"""
    fname, fext = os.path.splitext(os.path.basename(track.getPath()))
    outBed = os.path.join(args.outputDir, fname + "_scale" + ".bed")
    outBigWig = os.path.join(args.outputDir, fname + "_scale" + ".bw")
    outFile.open(outBed, "w")
    
    trackNo = track.getNumber()
    valMap = track.getValueMap()

    for trackTable in trackData.getTrackTableList():
        chrom = trackTable.getChrom()
        start = trackTable.getStart()
        for i in xrange(len(trackTable)):
            binnedVal = trackTable[i][trackNo]
            unbinnedVal = valMap.getMapBack(binnedVal)
            
            outBed.write("%s\t%d\t\%d\t\%f\n" % (
                chrom,
                start + i,
                start + i + 1,
                unbinnedVal))

    outFile.close()

    #make a .bw copy
    hasBedGraphToBigWig = False
    try:
        runShellCommand("bedGraphToBigWig")
        hasBedGraphToBigWig = True
    except:
        pass
    if hasBedGraphToBigWig is True:
        runShellCommand("bedGraphToBigWig %s %s %s" % outBed, args.chromSizes,
                        outBigWig)
            
if __name__ == "__main__":
    sys.exit(main())
