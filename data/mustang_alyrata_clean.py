#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
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
from teHmm.common import runShellCommand, getLogLevelString, getLocalTempPath

"""

The HMM cannot use annotation tracks as output by several tools (ex repeatmasker)
without first doing some name-munging and setting some scaling parameters for binning.

This script takes as input a list of "raw" annotation tracks, and runs all the necessary scripts to produce a list of "clean" tracks that can be used by the HMM.

NOTE: This script is really hardcoded to run only on ./mustang_alyrata_tracks.xml at the moment, and needs to be updated to reflect changes to that file (maybe).  A more general workflow will need to be put in place later....
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate HMM-usable tracklist from raw tracklist. EX "
        "used to transform mustang_alyrata_tracks.xml -> "
        "mustang_alyrata_clean.xml.  Runs cleanChaux.py cleanLtrFinder.py and "
        " cleanTermini.py and setTrackScaling.py")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("cleanTrackPath", help="Directory to write cleaned BED"
                        " tracks to")
    parser.add_argument("outTracksInfo", help="Path to write modified tracks XML"
                        " to.")
    parser.add_argument("--numBins", help="Maximum number of bins after scaling",
                        default=10, type=int)
    parser.add_argument("--scaleTracks", help="Comma-separated list of tracks "
                        "to process for scaling. If not set, all"
                        " tracks listed as having a multinomial distribution"
                        " (since this is the default value, this includes "
                        "tracks with no distribution attribute) will be"
                        " processed.", default=None)
    parser.add_argument("--skipScale", help="Comma-separated list of tracks to "
                        "skip for scaling.", default=None)
    parser.add_argument("--chaux", help="Name of chaux track", default="chaux")
    parser.add_argument("--ltrfinder", help="Name of ltrfinder track",
                        default="ltr_finder")
    parser.add_argument("--termini", help="Name of termini track",
                        default="termini")
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    try:
        os.makedirs(args.cleanTrackPath)
    except:
        pass
    if not os.path.isdir(args.cleanTrackPath):
        raise RuntimeError("Unable to find or create cleanTrack dir %s" %
                           args.cleanTrackPath)

    tempTracksInfo = getLocalTempPath("mustang_alyrata_clean", "xml")
    runCleaning(args, tempTracksInfo)
    assert os.path.isfile(tempTracksInfo)
    
    runScaling(args, tempTracksInfo)

    runShellCommand("rm -f %s" % tempTracksInfo)

    cleanBedTool(tempBedToolPath)

def cleanPath(args, track):
    """ path of cleaned track """
    oldPath = track.getPath()
    oldFile = os.path.basename(oldPath)
    oldName, oldExt = os.path.splitext(oldFile)
    return os.path.join(args.cleanTrackPath, oldName + "_clean" + oldExt)
    
def runCleaning(args, tempTracksInfo):
    """ run scripts for cleaning chaux, ltr_finder, and termini"""
    trackList = TrackList(args.tracksInfo)

    # run cleanChaux.py --keepSlash
    chauxTrack = trackList.getTrackByName(args.chaux)
    if chauxTrack is not None:
        outFile = cleanPath(args, chauxTrack)
        runShellCommand("cleanChaux.py %s --keepSlash > %s" % (
            chauxTrack.getPath(), outFile))
        chauxTrack.setPath(outFile)
    else:
        logger.warning("Could not find chaux track")

    # run cleanTermini.py
    terminiTrack = trackList.getTrackByName(args.termini)
    if terminiTrack is not None:
        outFile = cleanPath(args, terminiTrack)
        runShellCommand("cleanTermini.py %s %s" % (terminiTrack.getPath(),
                                                   outFile))
        terminiTrack.setPath(outFile)
    else:
        logger.warning("Could not find termini track")

    # run cleanLtrFinder.py
    ltrfinderTrack = trackList.getTrackByName(args.ltrfinder)
    if ltrfinderTrack is not None:
        outFile = cleanPath(args, ltrfinderTrack)
        runShellCommand("cleanLtrFinderID.py %s %s" % (ltrfinderTrack.getPath(),
                                                       outFile))
        ltrfinderTrack.setPath(outFile)
    else:
        logger.warning("Could not find ltrfinder track")

    # save a temporary xml
    trackList.saveXML(tempTracksInfo)

def runScaling(args, tempTracksInfo):
    """ run setTrackScaling on temp track list"""
    tracksArg = ""
    if args.scaleTracks is not None:
        tracksArg = args.scaleTracks
    skipArg = ""
    if args.skipScale is not None:
        skipArg = args.skipScale

    cmd = "setTrackScaling.py %s %d %s --logLevel %s %s %s" % (
        tempTracksInfo, args.numBins, args.outTracksInfo,
        getLogLevelString(), tracksArg, skipArg)
    runShellCommand(cmd)


if __name__ == "__main__":
    sys.exit(main())
