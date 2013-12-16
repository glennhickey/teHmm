#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import itertools
import copy

from teHmm.common import runShellCommand
from teHmm.track import TrackList
from pybedtools import BedTool, Interval

""" This script is called from teHmmBenchmark.py after a training / eval / comp
is performed on one bed.  It looks through the various results and aggregates
them into a spreadsheet row, which is saved to yet another file.  The idea is
that teHmmBenchmark can then collect all these rows and make a nice table.
The idea of doing this in a different script as opposed to within teHmmBenchmark
is so that it can be done as soon as possible for each input (rather than at the
very end), making crashes or aborts less heartbreaking.  
"""
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Make benchmark summary row.  Called from within "
        "teHmmBenchmark.py")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file that"
                        " teHmmBenchmark.py was run on.")
    parser.add_argument("localTracksInfo", help="Path of Tracks Info file for"
                        " row (could be a subset of above)")
    parser.add_argument("evalBed", help="Bed file created by teHmmEval.  Used"
                        " for the Viterbi score in comment at top")
    parser.add_argument("compBed", help="Results of comparison script")
    parser.add_argument("outRow", help="File to write row information to")
     
    args = parser.parse_args()

    inputTrackList = TrackList(args.tracksInfo)
    trackList = TrackList(args.localTracksInfo)
    
    header, row = scrapeRow(inputTrackList, trackList, args.evalBed,
                            args.compBed)

    header = map(str, header)
    row = map(str, row)
    
    outFile = open(args.outRow, "w")
    outFile.write(",".join(header) + "\n")
    outFile.write(",".join(row) + "\n")
    outFile.close()

def scrapeRow(inputTrackList, trackList, evalPath, compPath):
    """ The text files that get written by this script cant really be
    aggregated.  Instead of redesigning everything, this function picks
    and chooses the information we want, and writes a nice row of data.
    Of course, if any program gets changed to modify its output, then
    this function will need to be updated which could be dangerous.
    """

    header = []
    row = []
    
    #Column 1: Number of tracks
    header.append("NumTracks")
    row.append(len(trackList))

    #Column 2->X: Track presence / absence
    for inTrack in inputTrackList:
        header.append(inTrack.getName())
        track = trackList.getTrackByName(inTrack.getName())
        if track is not None:
            row.append(1)
        else:
            row.append(0)

    #Column X+1: Viterbi score
    header.append("Vit")
    efile = open(evalPath, "r")
    line1 = efile.readline()
    vit = line1.split()[-1]
    efile.close()
    row.append(float(vit))

    #Column X+2...: Accuracy results from coompareBedStates
    cfile = open(compPath, "r")
    lineN2 = [line for line in cfile][-2]
    cfile = open(compPath, "r")
    lineN = [line for line in cfile][-1]
    cfile.close()
    header += lineN2.split()
    row += lineN.split()

    return header, row

if __name__ == "__main__":
    sys.exit(main())
