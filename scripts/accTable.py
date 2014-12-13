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

""" another wrapper for compareBedStates.py that will compare many files
and make a decent table output
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="another wrapper for compareBedStates.py that will compare many files"
        " and make a decent table output")

    parser.add_argument("tracksList", help="XML tracks list")
    parser.add_argument("truthBeds", help="comma-separated references to benchmark against (ex repet)")
    parser.add_argument("testBeds", help="comma-spearated test beds")
    parser.add_argument("workDir", help="folder to write comparision outputs")
    parser.add_argument("outCSV", help="path for output")
    parser.add_argument("--state", help="state name", default="TE")
    parser.add_argument("--delMask", help="see help for compareBedStates.py", default=None, type=int)
    parser.add_argument("--proc", help="number of prcesses", default=1, type=int)
    parser.add_argument("--truthNames", help="comma-separated list of truth names", default =None)
    parser.add_argument("--testNames", help="comma-separated list of test names", default =None)
    
    args = parser.parse_args()

    truths = args.truthBeds.split(",")
    tests = args.testBeds.split(",")

    if args.truthNames is not None:
        truthNames = args.truthNames.split(",")
    else:
        truthNames = [os.path.splitext(os.path.basename(x))[0] for x in truths]
    if args.testNames is not None:
        testNames = args.testNames.split(",")
    else:
        testNames = [os.path.splitext(os.path.basename(x))[0] for x in tests]

    if not os.path.isdir(args.workDir):
        runShellCommand("mkdir %s" % args.workDir)

    assert len(tests) == len(testNames)
    assert len(truths) == len(truthNames)

    compCmds = []
    for i in xrange(len(tests)):
        for j in xrange(len(truths)):
            opath = os.path.join(args.workDir, "%s_vs_%s.txt" % (testNames[i], truthNames[j]))
            flags = "--tl %s" % args.tracksList
            if args.delMask is not None:
                flags += " --delMask %d" % args.delMask
            cmd = "compareBedStates.py %s %s %s > %s" % (truths[j], tests[i], flags, opath)
            compCmds.append(cmd)

    runParallelShellCommands(compCmds, args.proc)

    # munging ############
    def prettyAcc((prec, rec)):
        f1 = 0.
        if prec + rec > 0:
            f1 = (2. * prec * rec) / (prec + rec)
        return ("%.4f" % prec, "%.4f" % rec, "%.4f" % f1)

    #table in memory
    table = dict()
    for i in xrange(len(tests)):
        for j in xrange(len(truths)):
            opath = os.path.join(args.workDir, "%s_vs_%s.txt" % (testNames[i], truthNames[j]))
            stats = extractCompStatsFromFile(opath)[0]
            if args.state not in stats:
                stats[args.state] = (0,0)
            table[(i, j)] = prettyAcc(stats[args.state])

    csvFile = open(args.outCSV, "w")
    
    header = "test"
    for name in truthNames:
        header += ", F1 " + name
    for name in truthNames:
        header += ", Prec " + name  + ", Rec " + name
    csvFile.write(header + "\n")

    for i in xrange(len(tests)):
        line = testNames[i]
        for j in xrange(len(truths)):
            prec, rec, f1 = table[(i, j)]
            line += ", " + f1
        for j in xrange(len(truths)):
            prec, rec, f1 = table[(i, j)]
            line += ", " + prec + ", " + rec
        csvFile.write(line + "\n")

    csvFile.close()        

    
if __name__ == "__main__":
    sys.exit(main())
