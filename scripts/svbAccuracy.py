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
from teHmm.bin.compareBedStates import extractCompStatsFromFile

""" Generate some accuracy results.  To be used on output of statesVsBic.py
(or some set of hmm prediction beds of the form *_trainsize.stateNum.bed
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=" Generate some accuracy results.  To be used on output of statesVsBic.py"
        "(or some set of hmm prediction beds of the form *_trainsize.stateNum.bed")

    parser.add_argument("tracksList", help="XML tracks list")
    parser.add_argument("truthBed", help="reference to benchmark against (ex repet)")
    parser.add_argument("fitBed", help="predition to fit against (ex modeler)")
    parser.add_argument("outDir", help="output directory")
    parser.add_argument("beds", help="one or more bed files to evaluate", nargs="*")
    parser.add_argument("--proc", help="number of parallel processes", type=int, default=1)
    parser.add_argument("--maskGap", help="interpolate masked gaps smaller than this", type=int, default=5000)

    args = parser.parse_args()

    # preloop to check files
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        
    if not os.path.isdir(args.outDir):
        runShellCommand("mkdir %s" % args.outDir)

    outFile = open(os.path.join(args.outDir, "accuracy.csv"), "w")

    # do two kinds of fitting vs modeer
    fitCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOut = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fit.bed"))
        fitOutFdr = fitOut.replace(".bed", "Fdr.bed")
        cmd = "fitStateNames.py %s %s %s --tl %s --tgt TE" % (args.fitBed, bed, fitOut, args.tracksList)
        fitCmds.append(cmd)
        cmdFdr = "fitStateNames.py %s %s %s --tl %s --tgt TE --fdr 0.65" % (args.fitBed, bed, fitOutFdr, args.tracksList)
        fitCmds.append(cmdFdr)

    # interpolate the gaps
    interpolateCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOut = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fit.bed"))
        fitOutFdr = fitOut.replace(".bed", "Fdr.bed")        
        fitOutMI = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fitMI.bed"))
        fitOutFdrMI = fitOutMI.replace(".bed", "Fdir.bed")
        cmd = "interpolateMaskedRegions.py %s %s %s %s --maxLen %d" % (args.tracksList, args.truthBed, fitOut, fitOutMI, args.maskGap)
        interpolateCmds.append(cmd)
        cmdFdr = "interpolateMaskedRegions.py %s %s %s %s --maxLen %d" % (args.tracksList, args.truthBed, fitOutFdr, fitOutFdrMI, args.maskGap)
        interpolateCmds.append(cmdFdr)

    # run the comparison
    compareCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOutMI = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fitMI.bed"))
        fitOutFdrMI = fitOutMI.replace(".bed", "Fdir.bed")
        comp = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_comp.txt"))
        compFdr = comp.replace(".txt", "Fdr.txt")
        cmd = "compareBedStates.py %s %s --tl %s --delMask %d > %s" % (args.truthBed, fitOutMI, args.tracksList, args.maskGap, comp)
        compareCmds.append(cmd)
        cmdFdr = "compareBedStates.py %s %s --tl %s --delMask %d > %s" % (args.truthBed, fitOutFdrMI, args.tracksList, args.maskGap, compFdr)
        compareCmds.append(cmdFdr)
    
    runParallelShellCommands(fitCmds, args.proc)
    runParallelShellCommands(interpolateCmds, args.proc)
    runParallelShellCommands(compareCmds, args.proc)

    # munging ############
    def prettyAcc((prec, rec)):
        f1 = 0.
        if prec + rec > 0:
            f1 = (2. * prec * rec) / (prec + rec)
        return "%.4f, %.4f, %.4f" % (prec, rec, f1)

    header = "states, trainSize, precision, recall, f1, fdrfit_precision, fdrfit_recall, fdrfit_f1"
    outFile.write(header + "\n")

    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        comp = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_comp.txt"))
        compFdr = comp.replace(".txt", "Fdr.txt")
        stats = extractCompStatsFromFile(comp)[0]
        statsFdr = extractCompStatsFromFile(compFdr)[0]
        if "TE" not in stats:
            stats["TE"] = (0,0)
        if "TE" not in statsFdr:
            statsFdr["TE"] = (0,0)
        line = "%d, %d" % (nStates, tSize) + "," + prettyAcc(stats["TE"]) + ", " + prettyAcc(statsFdr["TE"]) + "\n" 
        outFile.write(line)

    outFile.close()

        

    
if __name__ == "__main__":
    sys.exit(main())
