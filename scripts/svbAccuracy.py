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
from teHmm.bin.compareBedStates import extractCompStatsFromFile, extract2ClassSpecificityFromFile

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
    parser.add_argument("--exploreFdr", help="try a bunch of fdr values", action="store_true", default=False)
    parser.add_argument("--compWindow", help="intersect with this file before running comparison", default=None)

    args = parser.parse_args()

    # preloop to check files
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        
    if not os.path.isdir(args.outDir):
        runShellCommand("mkdir %s" % args.outDir)

    outFile = open(os.path.join(args.outDir, "accuracy.csv"), "w")

    truthBed = args.truthBed
    if args.compWindow is not None:
        truthBed = os.path.join(args.outDir, "clippedTruth.bed")
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (args.truthBed, args.compWindow, truthBed))

    if args.exploreFdr is True:
        fdrs = [0, .05, .1, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1]
    else:
        fdrs = [.65]

    # do two kinds of fitting vs modeer
    fitCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOut = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fit.bed"))
        fitLog = fitOut.replace(".bed", "_log.txt")
        cmd = "fitStateNames.py %s %s %s --tl %s --tgt TE --qualThresh 0.1 --logDebug --logFile %s" % (args.fitBed, bed, fitOut, args.tracksList, fitLog)
        fitCmds.append(cmd)
        for fdr in fdrs:
            fitOutFdr = fitOut.replace(".bed", "Fdr%f.bed" % fdr)
            fitLogFdr = fitOutFdr.replace(".bed", "_log.txt")
            cmdFdr = "fitStateNames.py %s %s %s --tl %s --tgt TE --fdr %f --logDebug --logFile %s" % (args.fitBed, bed, fitOutFdr, args.tracksList, fdr, fitLogFdr)
            fitCmds.append(cmdFdr)

    # interpolate the gaps
    interpolateCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOut = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fit.bed"))
        fitOutMI = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fitMI.bed"))
        cmd = "interpolateMaskedRegions.py %s %s %s %s --maxLen %d" % (args.tracksList, args.truthBed, fitOut, fitOutMI, args.maskGap)
        interpolateCmds.append(cmd)

        for fdr in fdrs:
            fitOutFdr = fitOut.replace(".bed", "Fdr%f.bed" % fdr)        
            fitOutFdrMI = fitOutMI.replace(".bed", "Fdr%f.bed" % fdr)
            cmdFdr = "interpolateMaskedRegions.py %s %s %s %s --maxLen %d" % (args.tracksList, args.truthBed, fitOutFdr, fitOutFdrMI, args.maskGap)
            interpolateCmds.append(cmdFdr)

    # run the comparison
    compareCmds = []
    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        fitOutMI = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_fitMI.bed"))
        comp = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_comp.txt"))
        cmd = ""
        fitOutMIClipped = fitOutMI
        if args.compWindow is not None:
            fitOutMIClipped = fitOutMI.replace(".bed", "_clipped.bed")
            cmd += "intersectBed -a %s -b %s | sortBed > %s && " % (fitOutMI, args.compWindow, fitOutMIClipped)
        cmd += "compareBedStates.py %s %s --tl %s --delMask %d > %s" % (args.truthBed, fitOutMIClipped, args.tracksList, args.maskGap, comp)
        compareCmds.append(cmd)
        for fdr in fdrs:
            fitOutFdrMI = fitOutMI.replace(".bed", "Fdr%f.bed" % fdr)
            compFdr = comp.replace(".txt", "Fdr%f.txt" % fdr)
            cmdFdr = ""
            fitOutFdrMIClipped = fitOutFdrMI
            if args.compWindow is not None:
                fitOutFdrMIClipped = fitOutFdrMI.replace(".bed", "_clipped.bed")
                cmdFdr += "intersectBed -a %s -b %s | sortBed > %s &&" % (fitOutFdrMI, args.compWindow, fitOutFdrMIClipped)
            cmdFdr += "compareBedStates.py %s %s --tl %s --delMask %d > %s" % (args.truthBed, fitOutFdrMIClipped, args.tracksList, args.maskGap, compFdr)
            compareCmds.append(cmdFdr)
    
    runParallelShellCommands(fitCmds, args.proc)
    runParallelShellCommands(interpolateCmds, args.proc)
    runParallelShellCommands(compareCmds, args.proc)
    # got a weird crash before where comp file wasn't found
    # maybe this will help?
    runShellCommand("sleep 10")

    # munging ############
    def prettyAcc((prec, rec), spec):
        f1 = 0.
        if prec + rec > 0:
            f1 = (2. * prec * rec) / (prec + rec)        
        return "%.4f, %.4f, %.4f, %.4f" % (prec, rec, f1, spec)

    header = "states, trainSize, precision, recall, f1, specificity"
    for fdr in fdrs:
        header += ", fdrfit%.3f_precision, fdrfit%.3f_recall, fdrfit%.3f_f1, fdrfit%.3f_specificity" % (fdr, fdr, fdr, fdr)
    if len(fdrs) > 1:
        header += "\n,,,,"
        for fdr in fdrs:
            header += ", %.3f, %.3f, %.3f, %.3f" % (fdr, fdr, fdr, fdr)
    outFile.write(header + "\n")

    for bed in args.beds:
        toks = "_".join(os.path.basename(bed).split(".")).split("_")
        tSize, nStates = int(toks[1]), int(toks[3])
        comp = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_comp.txt"))
        stats = extractCompStatsFromFile(comp)[0]
        if "TE" not in stats:
            stats["TE"] = (0,0)
        specificity = extract2ClassSpecificityFromFile(comp, "TE")
        line = "%d, %d" % (nStates, tSize) + "," + prettyAcc(stats["TE"], specificity) 
        for fdr in fdrs:
            compFdr = comp.replace(".txt", "Fdr%f.txt" % fdr)
            statsFdr = extractCompStatsFromFile(compFdr)[0]
            specFdr = extract2ClassSpecificityFromFile(compFdr, "TE")
            if "TE" not in statsFdr:
                statsFdr["TE"] = (0,0)
            line += ", " + prettyAcc(statsFdr["TE"], specFdr)
        line += "\n"

        outFile.write(line)

    # tack on some roc plots
    for bed in args.beds:
        header = "\n%s ROC\nfdr, prec, rec, f1, spec, 1-spec, sens" % bed
        outFile.write(header + "\n")
        for fdr in fdrs:
            line = "%.3f" % fdr            
            toks = "_".join(os.path.basename(bed).split(".")).split("_")
            tSize, nStates = int(toks[1]), int(toks[3])
            comp = os.path.join(args.outDir, os.path.basename(bed).replace(".bed", "_comp.txt"))
            compFdr = comp.replace(".txt", "Fdr%f.txt" % fdr)
            statsFdr = extractCompStatsFromFile(compFdr)[0]
            specificity = extract2ClassSpecificityFromFile(compFdr, "TE")
            if "TE" not in statsFdr:
                statsFdr["TE"] = (0,0)
            line += ", " + prettyAcc(statsFdr["TE"], specificity) + ", %.4f" % (1-specificity)
            line += ", %.4f" % statsFdr["TE"][1]
            line += "\n"
            outFile.write(line)


    outFile.close()

        

    
if __name__ == "__main__":
    sys.exit(main())
