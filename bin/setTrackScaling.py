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

from teHmm.track import TrackList
from teHmm.trackIO import readTrackData
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Automatically set the scale attributes of numeric tracks"
        " within a given tracks.xml function using some simple heuristics. ")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("numBins", help="Maximum number of bins after scaling",
                        default=10, type=int)
    parser.add_argument("outputTracks", help="Path to write modified tracks XML"
                        " to.")
    parser.add_argument("--tracks", help="Comma-separated list of tracks "
                        "to process. If not set, all"
                        " tracks listed as having a multinomial distribution"
                        " (since this is the default value, this includes "
                        "tracks with no distribution attribute) will be"
                        " processed.", default=None)
    parser.add_argument("--skip", help="Comma-separated list of tracks to "
                        "skip.", default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    trackNames = []
    if args.tracks is not None:
        trackNames = args.tracks.split(",")
    skipNames = []
    if args.skip is not None:
        skipNames = args.skip.split(",")
    
    trackList = TrackList(args.tracksInfo)
    outTrackList = copy.deepcopy(trackList)

    for track in trackList:
        trackExt = os.path.splitext(track.getPath())[1]
        isFasta = len(trackExt) >= 3 and trackExt[:3].lower() == ".fa"
        if track.getName() not in skipNames and\
          (track.getName() in trackNames or len(trackNames) == 0) and\
          (track.getDist() == "multinomial" or
           track.getDist() == "sparse_multinomial") and\
          not isFasta:
          try:
            setTrackScale(track, args.numBins)
          except ValueError as e:
            logger.warning("Skipping (non-numeric?) track %s due to: %s" % (
              track.getName(), str(e)))

    trackList.saveXML(args.outputTracks)
    cleanBedTool(tempBedToolPath)

def setTrackScale(track, numBins):
    """ Modify the track XML element in place with the heuristically
    computed scaling paramaters below """
    data = readTrackIntoFloatArray(track)
    if len(data) > numBins:
        scaleType, scaleParam = computeScale(data, numBins)
        # round down so xml file doesnt look too ugly
        if scaleParam > 1e-4:
            scaleParam = float("%.4f" % scaleParam)
        if scaleType == "scale":
            logger.info("Setting track %s scale to %f" % (track.getName(),
                                                          scaleParam))
            track.setScale(scaleParam)
        elif scaleType == "logScale":
            logger.info("Setting track %s logScale to %f" % (track.getName(),
                                                             scaleParam))
            track.setLogScale(scaleParam)
    
def readTrackIntoFloatArray(track):
    """ use the track API to directly read an entire data file into memory
    as an array of floats"""
    numLines = int(runShellCommand("wc -l %s" % track.getPath()).split()[0])
    assert numLines > 0
    logger.debug("Allocating track array of size %d" % numLines)
    data = np.finfo(np.float).max + np.zeros((numLines), dtype=np.float)
    data = readTrackData(track.getPath(), outputBuf=data, valCol=track.getValCol())
    lastIdx = len(data) - 1
    for i in xrange(1, len(data)):
        if data[-i] != np.finfo(np.float).max:
            lastIdx = len(data) - i
            break
    data = data[:lastIdx]
    assert data[-1] != np.finfo(np.float).max
    return data

def histVariance(data, bins, fromLog = False):
    """ use histogram variance as a proxy for quality of binning"""
    freq, bins = np.histogram(data, bins)
    if bins[0] <= 0 or fromLog is False:
        assert np.sum(freq) == len(data)
    else:
        tempBins = np.zeros((len(bins)+1))
        tempBins[1:] = bins
        tempFreq, tempBins = np.histogram(data, tempBins)
        assert np.sum(np.sum(tempFreq) == len(data))
    return np.var(freq)

def computeScale(data, numBins):
    """ very simple heuristic to compute reasonable scaling"""
    minVal, maxVal = np.amin(data), np.amax(data)
    range = maxVal - minVal
    logger.debug("Min=%f Max=%f" % (minVal, maxVal))

    #NOTE: the -2.0 when computing the linear binSize and logBase
    # are very conservative measures to insure that we dont under
    # bin on each side due to rounding.  Can result in bins that
    # are too large when binsize, say, divides evently into the
    # range.  Should be optimized down the road when have more time.

    # try linear scale
    binSize = float(range) / float(numBins - 2.0)
    minBin = np.floor(minVal / binSize) * binSize
    linearBins = [minBin] * numBins
    for i in xrange(1, numBins):
        linearBins[i] = linearBins[i-1] + binSize
    logger.debug("Linear bins %s" % linearBins)
    linearVar = histVariance(data, sorted(linearBins))
    linearScale = 1.0 / binSize
    logger.debug("Linear scale=%f has variance=%f" % (linearScale, linearVar))
    
    # try log scale
    logVar = sys.maxint

    # for the purposes of scaling (see track.CategoryMap.__scale()), we
    # assume log(0) == 0.  Therefore there is effectively always a 0-bin
    # for logScaling, and we essentially ignore 0-values below. 
    if minVal == 0:
        # second smallest value
        newMin = sys.maxint
        for i in xrange(len(data)):
            if data[i] > minVal and data[i] < newMin:
                newMin = data[i]
        minVal = newMin
    # dont support negative numbers in log mode for now
    if maxVal != 0.0 and minVal != sys.maxint:
        ratio = float(maxVal) / float(minVal)
        logBase = np.power(ratio, 1. / float(numBins - 2.00))
        minBin = np.power(logBase, np.floor(np.log(minVal) / np.log(logBase)))
        logBins = [minBin] * numBins
        for i in xrange(1, numBins):
            logBins[i] = logBins[i-1] * logBase
        logger.debug("Log bins %s" % logBins)
        logVar = histVariance(data, sorted(logBins), fromLog = True)
        logger.debug("Log base=%f has variance=%f" % (logBase, logVar))

    ret = "scale", linearScale
    if logVar < linearVar:
        ret = "logScale", logBase
    return ret
            
if __name__ == "__main__":
    sys.exit(main())
