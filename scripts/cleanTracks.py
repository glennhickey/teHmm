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
import itertools


from teHmm.track import TrackList
from teHmm.trackIO import readTrackData
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runShellCommand, getLogLevelString, getLocalTempPath

"""
fix up track names and sort alphabetically.  easier to do here on xml than at end for paper. 
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="fix up track names and sort alphabetically.  easier to do here on xml than at end for pape\
        r.")
    
    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("outTracksInfo", help="Path to write modified tracks XML")

    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()
    args.logOpString = "--logLevel %s" % getLogLevelString()
    if args.logFile is not None:
        args.logOpString += " --logFile %s" % args.logFile

    nm = dict()
    nm["hollister"] = "RM-RepBase-Hollister"
    nm["chaux"] = "RM-RepBase-deLaChaux"
    nm["repeat_modeler"] = "RM-RepeatModeler"
    nm["repbase"] = "RM-RepBase"
    nm["repet"] = "REPET"
    nm["ltr_finder"] = "LTR_FINDER"
    nm["ltr_harvest"] = "LTR_Harvest"
    nm["ltr_termini"] = "lastz-Termini"
    nm["lastz-Termini"] = "lastz-LTRTermini"
    nm["tir_termini"] = "lastz-InvTermini"
    nm["irf"] = "IRF"
    nm["palindrome"] = "lastz-Palindrome"
    nm["overlap"] = "lastz-Overlap"
    nm["mitehunter"] = "MITE-Hunter"
    nm["helitronscanner"] = "HelitronScanner"
    nm["cov_80-"] = "lastz-SelfLowId"
    nm["cov_80-90"] = "lastz-SelfMedId"
    nm["cov_90+"] = "lastz-SelfHighId"
    nm["left_peak_80-"] = "lastz-SelfPeakLeftLow"
    nm["lastz-SelfLowLeftPeak"] = nm["left_peak_80-"]
    nm["left_peak_80-90"] = "lastz-SelfPeakLeftMed"
    nm["lastz-SelfMedLeftPeak"] = nm["left_peak_80-90"]
    nm["left_peak_90+"] = "lastz-SelfPeakLeftHigh"
    nm["lastz-SelfHighLeftPeak"] = nm["left_peak_90+"]
    nm["right_peak_80-"] = "lastz-SelfPeakRightLow"
    nm["lastz-SelfLowRightPeak"] = nm["right_peak_80-"]
    nm["right_peak_80-90"] = "lastz-SelfPeakRightMed"
    nm["lastz-SelfMedRightPeak"] = nm["right_peak_80-90"]
    nm["right_peak_90+"] = "lastz-SelfPeakRightHigh"
    nm["lastz-SelfHighRightPeak"] = nm["right_peak_90+"]
    nm["cov_maxPId"] = "lastz-SelfPctMaxId"
    nm["lastz-SelfMaxPctId"] = nm["cov_maxPId"]
    nm["te_domains"] = "TE-Domains"
    nm["fgenesh"] = "Genes"
    nm["genes"] = nm["fgenesh"]
    nm["refseq"] = nm["fgenesh"]
    nm["mrna"] = "mRNA"
    nm["srna"] = "sRNA"
    nm["ortho_depth"] = "Alignment-Depth"
    nm["orthology"] = nm["ortho_depth"]
    nm["chain_depth"] = nm["ortho_depth"]
    nm["alignment_depth"] = nm["ortho_depth"]
    nm["gcpct"] = "GC"
    nm["trf"] = "TRF"
    nm["windowmasker"] = "WindowMasker"
    nm["polyN"] = "Ns"
    nm["phastcons_ce"] = "Conservation"
    nm["phastcons"] = nm["phastcons_ce"]
    nm["PhastCons"] = nm["phastcons_ce"]
    nm["phyloP"] = nm["phastcons_ce"]
    nm["phylop"] = nm["phastcons_ce"] 

    rtracks = dict()
    rtracks["tantan"] = True
    rtracks["polyA"] = True
    rtracks["transposon_psi"] = True
    rtracks["transposonpsi"] = True
    rtracks["repbase_censor"] = True
    rtracks["tsd"] = True
    rtracks["repbase_default"] = True
    rtracks["dustmasker"] = True
       
    inTracks = TrackList(args.tracksInfo)
    outTracks = TrackList()
    outList = []

    for track in itertools.chain(inTracks.trackList, inTracks.maskTrackList):
        if not os.path.exists(track.path):
            raise RuntimeError("Track DNE %s" % track.path)
        if track.name not in rtracks:
            if track.name in nm:
                track.name = nm[track.name]
            else:
                logger.warning("Did not map track %s" % track.name)
            outList.append(track)                        
        else:
            logger.warning("Deleted track %s" % track.name)


    # sort the list
    def sortComp(x):
        lname = x.name.lower()
        if x.name == "RM-RepeatModeler":
            return "aaaaa" + lname
        elif "RM" in x.name:
            return "aaaa" + lname
        elif "REPET" in x.name:
            return "aaa" + lname
        elif "softmask" in lname or "tigr" in lname or "te-domains" in lname:
            return "aa" + lname
        elif x.getDist == "mask":
            return "zzzz" + lname
        else:
            return lname
        
    outList = sorted(outList, key = lambda track : sortComp(track))

    for track in outList:
        outTracks.addTrack(track)

    outTracks.saveXML(args.outTracksInfo)
    
    cleanBedTool(tempBedToolPath)    
    
if __name__ == "__main__":
    sys.exit(main())
