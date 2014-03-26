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
from teHmm.trackIO import readTrackData, getMergedBedIntervals
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
    parser.add_argument("allBed", help="Bed file spanning entire genome")
    parser.add_argument("outputTracks", help="Path to write modified tracks XML"
                        " to.")
    parser.add_argument("--numBins", help="Maximum number of bins after scaling",
                        default=10, type=int)
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

    allIntervals = getMergedBedIntervals(args.allBed)

    for track in trackList:
        trackExt = os.path.splitext(track.getPath())[1]
        isFasta = len(trackExt) >= 3 and trackExt[:3].lower() == ".fa"
        if track.getName() not in skipNames and\
          (track.getName() in trackNames or len(trackNames) == 0) and\
          (track.getDist() == "multinomial" or
           track.getDist() == "sparse_multinomial") and\
          not isFasta:
          try:
              setTrackScale(track, args.numBins, allIntervals)
          except ValueError as e:
              logger.warning("Skipping (non-numeric?) track %s due to: %s" % (
                  track.getName(), str(e)))

    trackList.saveXML(args.outputTracks)
    cleanBedTool(tempBedToolPath)

def setTrackScale(track, numBins, allIntervals):
    """ Modify the track XML element in place with the heuristically
    computed scaling paramaters below """
    data = readTrackIntoFloatArray(track, allIntervals)
    if len(data) > numBins:
        scaleType, scaleParam, shift = computeScale(data, numBins)
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
        logger.info("Setting track %s shift to %f" % (track.getName(),
                                                      shift))
        track.setShift(shift)
    
def readTrackIntoFloatArray(track, allIntervals):
    """ use the track API to directly read an entire data file into memory
    as an array of floats.  If the track has an associated defaultVal, it will
    be used to cover all gaps in the coverage.  If not, only annotated values
    will be kept"""
    defaultVal = track.getDefaultVal()
    hasDefault = defaultVal != None
    if not hasDefault:
        defaultVal = np.finfo(float).max
    else:
        # sanity check : we assume that no one ever actually uses this value
        defaultVal = float(defaultVal)
        assert defaultVal != np.finfo(float).max
    readBuffers = []
    totalLen = 0
    for interval in allIntervals:
        logger.debug("Allocating track array of size %d" % (
             interval[2] - interval[1]))
        buf = defaultVal + np.zeros((interval[2] - interval[1]), dtype=np.float)
        buf = readTrackData(track.getPath(), interval[0], interval[1],
                            interval[2], outputBuf=buf,
                            valCol=track.getValCol(),
                            useDelta=track.getDelta)
        readBuffers.append(buf)
        totalLen += len(buf)

    data = np.concatenate(readBuffers)
    assert len(data) == totalLen
    readBuffers = None

    if not hasDefault:
        # strip out all the float_max values we put in there since there is
        # no default value for unannotated regions, and we just ignore them
        # (ie as original implementation)
        stripData = np.ndarray((totalLen), dtype=np.float)
        basesRead = 0
        for i in xrange(totalLen):
            if buf[i] != defaultVal:
                stripDate[basesRead] = buf[i]
                basesRead += 1
        stripData.resize(basesRead)
        data = stripData

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

    # shift parameter is a constant that gets added before log scaling
    # to make sure that we always deal with positive numbers
    shift = 0.
    if minVal <= 0.:
        shift = 1.0 - minVal
        data += shift
        minVal += shift
        maxVal += shift

    ratio = float(maxVal) / float(minVal)
    logBase = np.power(ratio, 1. / float(numBins - 2.00))
    minBin = np.power(logBase, np.floor(np.log(minVal) / np.log(logBase)))
    logBins = [minBin] * numBins
    for i in xrange(1, numBins):
        logBins[i] = logBins[i-1] * logBase
    logger.debug("Log bins %s" % logBins)
    logVar = histVariance(data, sorted(logBins), fromLog = True)
    logger.debug("Log base=%f has variance=%f" % (logBase, logVar))

    ret = "scale", linearScale, 0.
    if logVar < linearVar:
        ret = "logScale", logBase, shift
    return ret
            
if __name__ == "__main__":
    sys.exit(main())
