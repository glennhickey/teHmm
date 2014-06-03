#!/usr/bin/env python

import ast
import sys
import os
import argparse
import logging
import itertools
import copy

from teHmm.common import runShellCommand
from teHmm.modelIO import loadModel

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Make table from bunch of bed predictions")
    parser.add_argument("workdir", help="working directory. lots of individual"
                        " comparison files output here")
    parser.add_argument("beds", nargs="*", help="inbeds")
    parser.add_argument("--states", help="comma-separated list of state"
                         " naems to use (MANDATORY)", default=None)
    parser.add_argument("--base", help="use base-level instead of "
                        "interval accuracy", action="store_true",
                        default=False)
    args = parser.parse_args()
    if args.states is None:
        return 1
    for i, bed in enumerate(args.beds):
        if i == 0:
            row = makeRow(bed, True, args)
            print "\t".join(row)
        row = makeRow(bed, False, args)
        print stringify(row)
    return 0

def stringify(row):
    """ change floats to strings"""
    outRow = []
    for i in row:
        try:
            if isinstance(i, int):
                raise 0
            outRow.append("%.3f" % float(i))
        except:
            outRow.append(str(i))
    return "\t".join(outRow)

def makeRow(bed, header, args):
    """ print the tab-separated table row to stdout"""
    trueBed = getTrueBed(bed)
    fixBed = os.path.join(args.workdir, os.path.basename(bed) + ".fix")
    compFile = os.path.join(args.workdir, os.path.basename(bed) + ".comp")
    runShellCommand("fitStateNames.py %s %s %s" % (trueBed, bed, fixBed))
    runShellCommand("compareBedStates.py %s %s > %s" % (trueBed, fixBed, compFile))
    idRow = extractIdRow(bed, header)
    compRow = extractCompRow(compFile, header, args)
    probRow = extractProbRow(bed, header, args)
    return idRow + compRow + probRow

def getTrueBed(bed):
    """ detect truth from filename in hardcoded manner"""
    if "segmentsm" in bed:
        return "/data/glenn.hickey/genomes/teHmm/data/truth/manualChaux_flat.bed"
    if "segments3" in bed:
        return "repet3.bed"
    assert False

def extractIdRow(bed, header):
    """ similar to above, but get some columns about input args"""
    region = "scaffold_3"
    if "segmentsm" in bed:
        region = "manual"
    thresh = 0
    if "_s1_" in bed:
        thresh = 1
    segLen = 0
    if "_b200" in bed:
        segLen = 200
    strat = "fix"
    if "unsup" in bed:
        strat = "unsup"
    if "semi" in bed:
        strat = "semi"
    rep = "no"
    if "mod.rep" in bed:
        rep = "yes"
    if not header:
        return [region, strat, segLen, thresh, rep]
    else:
        return ["Region", "Strat", "segLen", "segThresh", "Non-Opt Rep"]
    
def extractCompRow(compFile, header, args):
    """ dig out some results from the output of compareBedStates.py"""
    cf = open(compFile, "r")
    intStats = None
    yes = False
    accKeyword = "Interval Accuracy"
    if args.base is True:
        accKeyword = "Base-by-base Accuracy"
    for line in cf:
        if accKeyword in line:
            yes = True
        elif yes == True:
            intStats = ast.literal_eval(line)
            assert isinstance(intStats, dict)
            break
    assert intStats is not None
    cf.close()
    outList = []
    total = (0., 0., 0.)
    for state in args.states.split(","):
        assert state in intStats
        if not header:
            prec = intStats[state][0]
            rec = intStats[state][1]
            f1 = 0
            if prec + rec > 0:
                f1 = 2. * ((prec * rec) / (prec + rec))
            outList += [prec, rec, f1]
            total = (total[0] + prec, total[1] + rec, total[2] + f1)
        else:
            outList += ["%s_prec" % state, "%s_rec" % state, "%s_f1" % state]

    count = float(len(outList) / 3)
    if count > 0:
        if not header:
            outList += [x / count for x in total]
        else:
            outList += ["%s_prec" % "avg", "%s_rec" % "avg", "%s_f1" % "avg"]
    return outList
        
def extractProbRow(bed, header, args):
    """ get the total prob from the model and the viterbi prob from the bed
    """
    # note directory structure hardcoded in line below:
    modPath = bed.replace("predictions", "models")
    modPath = modPath[:modPath.rfind("_")]
    if ".mod" not in modPath:
        modPath += ".mod"
    assert os.path.exists(modPath)
    model = loadModel(modPath)
    totalProb = model.getLastLogProb()
    bedFile = open(bed, "r")
    line0 = [line for line in bedFile][0]
    vitProb = float(line0.split()[2])
    assert vitProb <= 0.
    if header:
        return ["TotLogProb", "VitLogProb"]
    else:
        return [totalProb, vitProb]
    bedFile.close()
    model = NULL
    
if __name__ == "__main__":
    sys.exit(main())
        

    
        
