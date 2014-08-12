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
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate HMM-usable tracklist from raw tracklist. EX "
        "used to transform mustang_alyrata_tracks.xml -> "
        "mustang_alyrata_clean.xml.  Runs cleanRM.py cleanLtrFinder.py and "
        " cleanTermini.py and addTsdTrack.py and setTrackScaling.py (also runs "
        " removeBedOverlaps.py before each of the clean scripts)")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("allBed", help="Bed file spanning entire genome")
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
    parser.add_argument("--ltrfinder", help="comma-sep Name(s) of ltrfinder track(s)",
                        default="ltr_finder,ltr_harvest")
    parser.add_argument("--ltr_termini", help="Name of termini track (appy cleanTermini.py)",
                        default="ltr_termini")
    parser.add_argument("--overlap", help="Name of overlap track (appy cleanTermini.py)",
                        default="overlap")
    parser.add_argument("--palindrome", help="Name of palindrome track (appy cleanTermini.py)",
                        default="palindrome")
    parser.add_argument("--sequence", help="Name of fasta sequence track",
                        default="sequence")
    parser.add_argument("--tsd", help="Name of tsd track to generate (appy cleanTermini.py)",
                        default="tsd")
    parser.add_argument("--tir", help="Name of tir_termini track (appy cleanTermini.py)",
                        default="tir_termini")
    parser.add_argument("--irf", help="Name of irf track",
                        default="irf")
    parser.add_argument("--hollister", help="Name of hollister track",
                        default="hollister")
    parser.add_argument("--repbase", help="Name of repbase track",
                        default="repbase")
    parser.add_argument("--repeat_modeler", help="Name of repeat_modeler track",
                        default="repeat_modeler")
    parser.add_argument("--transposon_psi", help="Name of transposon_psi track",
                        default="transposon_psi")
    parser.add_argument("--repbase_censor", help="Name of repbase_censor track",
                        default="repbase_censor")
    parser.add_argument("--noScale", help="Dont do any scaling", default=False,
                        action="store_true")
    parser.add_argument("--noTsd", help="Dont generate TSD track.  NOTE:"
                        " TSD track is hardcoded to be generated from "
                        "termini and (non-LTR elements of ) chaux",
                        default=False, action="store_true")
    parser.add_argument("--numProc", help="Number of processes to use for tsdFinder.py",
                        default=1, type=int)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()
    args.logOpString = "--logLevel %s" % getLogLevelString()
    if args.logFile is not None:
        args.logOpString += " --logFile %s" % args.logFile

    try:
        os.makedirs(args.cleanTrackPath)
    except:
        pass
    if not os.path.isdir(args.cleanTrackPath):
        raise RuntimeError("Unable to find or create cleanTrack dir %s" %
                           args.cleanTrackPath)

    tempTracksInfo = getLocalTempPath("Temp_mustang_alyrata_clean", "xml")
    runCleaning(args, tempTracksInfo)
    assert os.path.isfile(tempTracksInfo)

    runTsd(args, tempTracksInfo)
    
    runScaling(args, tempTracksInfo)

    runShellCommand("rm -f %s" % tempTracksInfo)

    cleanBedTool(tempBedToolPath)

def cleanPath(args, track):
    """ path of cleaned track """
    oldPath = track.getPath()
    oldFile = os.path.basename(oldPath)
    oldName, oldExt = os.path.splitext(oldFile)
    return os.path.join(args.cleanTrackPath, oldName + "_clean" + ".bed")
    
def runCleaning(args, tempTracksInfo):
    """ run scripts for cleaning chaux, ltr_finder, and termini"""
    trackList = TrackList(args.tracksInfo)

    # run cleanRM.py on chaux track
    chauxTrack = trackList.getTrackByName(args.chaux)
    if chauxTrack is not None:
        inFile = chauxTrack.getPath()
        outFile = cleanPath(args, chauxTrack)
        tempBed = getLocalTempPath("Temp_chaux", ".bed")
        runShellCommand("cleanRM.py --keepUnderscore %s > %s" % (inFile,
                                                                    tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile))
        runShellCommand("rm -f %s" % tempBed)
        chauxTrack.setPath(outFile)
    else:
        logger.warning("Could not find chaux track")

    # run cleanRM.py on hollister track
    hollisterTrack = trackList.getTrackByName(args.hollister)
    if hollisterTrack is not None:
        inFile = hollisterTrack.getPath()
        outFile = cleanPath(args, hollisterTrack)
        tempBed = getLocalTempPath("Temp_hollister", ".bed")
        runShellCommand("cleanRM.py %s > %s" % (inFile, tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile)) 
        runShellCommand("rm -f %s" % tempBed)
        hollisterTrack.setPath(outFile)
    else:
        logger.warning("Could not find hollister track")

    # run cleanRM.py on repbase track
    repbaseTrack = trackList.getTrackByName(args.repbase)
    if repbaseTrack is not None:
        inFile = repbaseTrack.getPath()
        outFile = cleanPath(args, repbaseTrack)
        tempBed = getLocalTempPath("Temp_repbase", ".bed")
        runShellCommand("cleanRM.py %s > %s" % (inFile, tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile)) 
        runShellCommand("rm -f %s" % tempBed)
        repbaseTrack.setPath(outFile)
    else:
        logger.warning("Could not find repbase track")

    # run cleanRM.py on repeat_modeler track
    repeat_modelerTrack = trackList.getTrackByName(args.repeat_modeler)
    if repeat_modelerTrack is not None:
        inFile = repeat_modelerTrack.getPath()
        outFile = cleanPath(args, repeat_modelerTrack)
        tempBed1 = None
        if inFile[-3:] == ".bb":
            tempBed1 = getLocalTempPath("Temp_modeler", ".bed")
            runShellCommand("bigBedToBed %s %s" % (inFile, tempBed1))
            inFile = tempBed1
        tempBed = getLocalTempPath("Temp_repeat_modeler", ".bed")
        runShellCommand("cleanRM.py %s  > %s" % (inFile, tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile)) 
        runShellCommand("rm -f %s" % tempBed)
        repeat_modelerTrack.setPath(outFile)
    else:
        logger.warning("Could not find repeat_modeler track")

    # run cleanRM.py on transposon_psi track
    transposon_psiTrack = trackList.getTrackByName(args.transposon_psi)
    if transposon_psiTrack is not None:
        inFile = transposon_psiTrack.getPath()
        outFile = cleanPath(args, transposon_psiTrack)
        tempBed1 = None
        if inFile[-3:] == ".bb":
            tempBed1 = getLocalTempPath("Temp_modeler", ".bed")
            runShellCommand("bigBedToBed %s %s" % (inFile, tempBed1))
            inFile = tempBed1
        tempBed = getLocalTempPath("Temp_transposon_psi", ".bed")
        runShellCommand("cleanRM.py %s  --keepUnderscore > %s" % (inFile, tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile)) 
        runShellCommand("rm -f %s" % tempBed)
        transposon_psiTrack.setPath(outFile)
    else:
        logger.warning("Could not find transposon_psi track")

    # run cleanRM.py on repbase_censor track
    repbase_censorTrack = trackList.getTrackByName(args.repbase_censor)
    if repbase_censorTrack is not None:
        inFile = repbase_censorTrack.getPath()
        outFile = cleanPath(args, repbase_censorTrack)
        tempBed1 = None
        if inFile[-3:] == ".bb":
            tempBed1 = getLocalTempPath("Temp_modeler", ".bed")
            runShellCommand("bigBedToBed %s %s" % (inFile, tempBed1))
            inFile = tempBed1
        tempBed = getLocalTempPath("Temp_repbase_censor", ".bed")
        runShellCommand("cleanRM.py %s  --keepUnderscore > %s" % (inFile, tempBed))
        runShellCommand("removeBedOverlaps.py %s > %s" % (tempBed, outFile)) 
        runShellCommand("rm -f %s" % tempBed)
        repbase_censorTrack.setPath(outFile)
    else:
        logger.warning("Could not find repbase_censor track")
                
    # run cleanTermini.py
    lastzTracks = [trackList.getTrackByName(args.ltr_termini),
                  trackList.getTrackByName(args.tir)]
    for terminiTrack in lastzTracks:
        if terminiTrack is not None:
            outFile = cleanPath(args, terminiTrack)
            inFile = terminiTrack.getPath()
            tempBed = None
            if inFile[-3:] == ".bb":
                tempBed = getLocalTempPath("Temp_termini", ".bed")
                runShellCommand("bigBedToBed %s %s" % (inFile, tempBed))
                inFile = tempBed
            runShellCommand("cleanTermini.py %s %s" % (inFile, outFile))
            terminiTrack.setPath(outFile)
            if tempBed is not None:
                runShellCommand("rm -f %s" % tempBed)
        else:
            logger.warning("Could not find termini track")

    # run removeBedOverlaps on irf (too bad its not in termini-like format
    # to be included in above logic)
    irfTrack = trackList.getTrackByName(args.irf)
    if irfTrack is not None:
        outFile = cleanPath(args, irfTrack)
        inFile = irfTrack.getPath()
        tempBed = None
        if inFile[-3:] == ".bb":
            tempBed = getLocalTempPath("Temp_termini", ".bed")
            runShellCommand("bigBedToBed %s %s" % (inFile, tempBed))
            inFile = tempBed
        runShellCommand("removeBedOverlaps.py %s > %s" % (inFile, outFile))
        irfTrack.setPath(outFile)
        if tempBed is not None:
            runShellCommand("rm -f %s" % tempBed)
    else:
        logger.warning("Could not find irf track")

    # run cleanLtrFinder.py
    for lfTrackName in args.ltrfinder.split(","): 
        ltrfinderTrack = trackList.getTrackByName(lfTrackName)
        if ltrfinderTrack is not None:
            inFile = ltrfinderTrack.getPath()
            outFile = cleanPath(args, ltrfinderTrack)
            # note: overlaps now removed in cleanLtrFinderID script
            runShellCommand("cleanLtrFinderID.py %s %s" % (inFile, outFile))
            ltrfinderTrack.setPath(outFile)
        else:
            logger.warning("Could not find ltrfinder track %s" % lfTrackName)

    

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

    if args.noScale is False:
        cmd = "setTrackScaling.py %s %s %s --numBins %d --logLevel %s %s %s" % (
            tempTracksInfo, args.allBed, args.outTracksInfo, args.numBins,
            getLogLevelString(), tracksArg, skipArg)
    else:
        cmd = "cp %s %s" % (tempTracksInfo, args.outTracksInfo)
    runShellCommand(cmd)

def runTsd(args, tempTracksInfo):
    """ run addTsdTrack on termini and chaux to generate tsd track"""
    if args.noTsd is True:
        return

    origTrackList = TrackList(args.tracksInfo)
    outTrackList = TrackList(tempTracksInfo)

    tempFiles = []
    tsdInputFiles = []
    tsdInputTracks = []
        
    # preprocess termini
    lastzTracks = [origTrackList.getTrackByName(args.ltr_termini),
                  origTrackList.getTrackByName(args.tir)]
    for terminiTrack in lastzTracks:
        if terminiTrack is not None:
            inFile = terminiTrack.getPath()
            fillFile = getLocalTempPath("Temp_fill", ".bed")
            tempBed = None
            if inFile[-3:] == ".bb":
                tempBed = getLocalTempPath("Temp_termini", ".bed")
                runShellCommand("bigBedToBed %s %s" % (inFile, tempBed))
                inFile = tempBed
            runShellCommand("fillTermini.py %s %s" % (inFile, fillFile))
            tsdInputFiles.append(fillFile)
            tsdInputTracks.append(terminiTrack.getName())
            tempFiles.append(fillFile)
            if tempBed is not None:
                runShellCommand("rm -f %s" % tempBed)
        else:
            logger.warning("Could not find termini track")

    # add repeat_modeler
    repeat_modelerTrack = outTrackList.getTrackByName(args.repeat_modeler)
    if repeat_modelerTrack is not None:
        tsdInputFiles.append(repeat_modelerTrack.getPath())
        tsdInputTracks.append(repeat_modelerTrack.getName())

    # run addTsdTrack (appending except first time)
    # note we override input track paths in each case
    assert len(tsdInputFiles) == len(tsdInputTracks)
    for i in xrange(len(tsdInputFiles)):
        optString = ""
        if i > 0:
            optString += " --append"
        # really rough hardcoded params based on
        # (A unified classification system for eukaryotic transposable elements
        # Wicker et. al 2007)
        if tsdInputTracks[i] == args.repeat_modeler:
            optString += " --names LINE,SINE,Unknown"
            optString += " --maxScore 20"
            optString += " --left 20"
            optString += " --right 20"
            optString += " --min 5"
            optString += " --max 20"
            optString += " --overlap 20"
        elif tsdInputTracks[i] == args.ltr_termini:
            optString += " --maxScore 3"
            optString += " --left 8"
            optString += " --right 8"
            optString += " --min 3"
            optString += " --max 6"
        elif tsdInputTracks[i] == args.tir:
            optString += " --maxScore 3"
            optString += " --left 15"
            optString += " --right 15"
            optString += " --min 3"
            optString += " --max 12"

        tempXMLOut = getLocalTempPath("Temp_tsd_xml", ".xml")
        runShellCommand("addTsdTrack.py %s %s %s %s %s %s --inPath %s %s %s --numProc %d" % (
            tempTracksInfo,
            args.cleanTrackPath,
            tempXMLOut,
            tsdInputTracks[i],
            args.sequence,
            args.tsd,
            tsdInputFiles[i],
            optString,
            args.logOpString,
            args.numProc))
        
        runShellCommand("mv %s %s" % (tempXMLOut, tempTracksInfo))

    for i in xrange(len(tempFiles)):
        runShellCommand("rm %s" % tempFiles[i])

if __name__ == "__main__":
    sys.exit(main())
