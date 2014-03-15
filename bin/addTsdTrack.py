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

from teHmm.track import TrackList, Track
from teHmm.trackIO import readTrackData
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand, getLocalTempPath, getLogLevelString

""" NOTE: TSD tracks will get flattended with the simplistic removeBedOVerlaps.py
Could be worth doing more principled later..."""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Add a TSD track (or modify an existing one) based on a "
        "given track")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("tsdTrackDir", help="Directory to write cleaned BED"
                        " tracks to")
    parser.add_argument("outTracksInfo", help="Path to write modified tracks XML"
                        " to.")
    parser.add_argument("inputTrack", help="Name of track to createTSDs from")
    parser.add_argument("fastaTrack", help="Name of track for fasta sequence")
    parser.add_argument("outputTrack", help="Name of tsd track to add.  Will"
                        " overwrite if it already exists (or append with"
                        " --append option)")
    parser.add_argument("--append", help="Add onto existing TSD track if exists",
                        default=False, action="store_true")
    parser.add_argument("--inPath", help="Use given file instead of inputTrack"
                        " path to generate TSD", default=None)

    ############ TSDFINDER OPTIONS ##############
    parser.add_argument("--min", help="Minimum length of a TSD",
                        default=None, type=int)
    parser.add_argument("--max", help="Maximum length of a TSD",
                        default=None, type=int)
    parser.add_argument("--all", help="Report all matches in region (as opposed"
                        " to only the nearest to the BED element which is the "
                        "default behaviour", action="store_true", default=False)
    parser.add_argument("--left", help="Number of bases immediately left of the "
                        "BED element to search for the left TSD",
                        default=None, type=int)
    parser.add_argument("--right", help="Number of bases immediately right of "
                        "the BED element to search for the right TSD",
                        default=None, type=int)
    parser.add_argument("--overlap", help="Number of bases overlapping the "
                        "BED element to include in search (so total space "
                        "on each side will be --left + overlap, and --right + "
                        "--overlap", default=None, type=int)
    parser.add_argument("--leftName", help="Name of left TSDs in output Bed",
                        default=None)
    parser.add_argument("--rightName", help="Name of right TSDs in output Bed",
                        default=None)
    parser.add_argument("--id", help="Assign left/right pairs of TSDs a unique"
                        " matching ID", action="store_true", default=False)
    parser.add_argument("--names", help="Only apply to bed interval whose "
                        "name is in (comma-separated) list.  If not specified"
                        " then all intervals are processed", default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    # copy out all options for call to tsd finder
    args.tsdFinderOptions = "--logLevel %s" % getLogLevelString()
    if args.logFile is not None:
        args.tsdFinderOptions += " --logFile %s" % args.logFile
    for option in ["min", "max", "all", "left", "right", "overlap",
                   "leftName", "rightName", "id", "names"]:
        val = getattr(args, option)
        if val is True:
            args.tsdFinderOptions += " --%s" % option
        elif val is not None and val is not False:
            args.tsdFinderOptions += " --%s %s" % (option, val)
            
    try:
        os.makedirs(args.tsdTrackDir)
    except:
        pass
    if not os.path.isdir(args.tsdTrackDir):
        raise RuntimeError("Unable to find or create tsdTrack dir %s" %
                           args.tsdTrackDir)

    trackList = TrackList(args.tracksInfo)
    outTrackList = copy.deepcopy(trackList)
    inputTrack = trackList.getTrackByName(args.inputTrack)
    if inputTrack is None:
        raise RuntimeError("Track %s not found" % args.inputTrack)
    if args.inPath is not None:
        assert os.path.isfile(args.inPath)
        inputTrack.setPath(args.inPath)
    inTrackExt = os.path.splitext(inputTrack.getPath())[1].lower()
    if inTrackExt != ".bb" and inTrackExt != ".bed":
        raise RuntimeError("Track %s has non-bed extension %s" % (
            args.inputTrack, inTrackExt))

    fastaTrack = trackList.getTrackByName(args.fastaTrack)
    if fastaTrack is None:
        raise RuntimeError("Fasta Track %s not found" % args.fastaTrack)
    faTrackExt = os.path.splitext(fastaTrack.getPath())[1].lower()
    if faTrackExt[:3] != ".fa":
        raise RuntimeError("Fasta Track %s has non-fasta extension %s" % (
            args.fastaTrack, faTrackExt))

    tsdTrack = outTrackList.getTrackByName(args.outputTrack)
    if tsdTrack is None:
        tsdTrack = Track()
        tsdTrack.name = args.outputTrack
        tsdTrack.path = os.path.join(args.tsdTrackDir, args.inputTrack + "_" +
                                     args.outputTrack + ".bed")

    runTsdFinder(fastaTrack.getPath(), inputTrack.getPath(),
                  tsdTrack.getPath(), args)

    if outTrackList.getTrackByName(tsdTrack.getName()) is None:
        outTrackList.addTrack(tsdTrack)
    outTrackList.saveXML(args.outTracksInfo)

    cleanBedTool(tempBedToolPath)

def runTsdFinder(faPath, inBedPath, outBedPath, args):
    """ call tsdFinder and either overwrite or append output.  also call
    removeBedOverlaps on final output to make sure it is clean """

    # convert input to bed if necessary
    tempBed = None
    if os.path.splitext(inBedPath)[1].lower() == ".bb":
        tempBed = getLocalTempPath("Temp_addTsdTrack", ".bed")
        runShellCommand("bigBedToBed %s %s" % (inFile, tempBed))
        inBedPath = tempBed

    # run tsdfinder on input
    tempOut = getLocalTempPath("Temp_addTsdTrack", ".bed")
    runShellCommand("tsdFinder.py %s %s %s %s" % (faPath, inBedPath,
                                                  tempOut,
                                                  args.tsdFinderOptions))
    if tempBed is not None:
        runShellCommand("rm %s" % tempBed)

    # merge with existing track
    if os.path.isfile(outBedPath) and args.append is True:
        runShellCommand("cat %s >> %s" % (outBedPath, tempOut))

    # remove overlaps into final output
    runShellCommand("removeBedOverlaps.py %s > %s" % (tempOut, outBedPath))

    runShellCommand("rm %s" % tempOut)
    
                
if __name__ == "__main__":
    sys.exit(main())
