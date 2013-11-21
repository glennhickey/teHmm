#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging

from teHmm.common import runShellCommand
from teHmm.common import runParallelShellCommands
from pybedtools import BedTool, Interval

def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train, evalaute, then compare hmm model on input")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("outputDir", help="directory to write output")
    parser.add_argument("inBeds", nargs="*", help="list of training beds")
    parser.add_argument("--verbose", help="Print out detailed logging messages",
                        action = "store_true", default = False)
    parser.add_argument("--numProc", help="Max number of processors to use",
                        type=int, default=1)
    
    args = parser.parse_args()
    if args.verbose is True:
        logging.basicConfig(level=logging.DEBUG)
        verbose = " --verbose"
    else:
        logging.basicConfig(level=logging.INFO)
        verbose = ""

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    #todo: try to get timing for each command
    commands = []
    for inBed in args.inBeds:
       # train
       base = os.path.basename(inBed)
       modPath = os.path.join(args.outputDir,
                              os.path.splitext(base)[0] + ".mod")
       command = "teHmmTrain.py %s %s %s %s --supervised" % (args.tracksInfo,
                                                             inBed,
                                                             modPath,
                                                             verbose)

       # view
       viewPath = os.path.join(args.outputDir,
                              os.path.splitext(base)[0] + "_view.txt")
       command += " && teHmmView.py %s > %s" % (modPath, viewPath)
       
       # evaluate
       evalBed = os.path.join(args.outputDir,
                              os.path.splitext(base)[0] + "_eval.bed")
       command += " && teHmmEval.py %s %s %s --bed %s %s" % (args.tracksInfo,
                                                             modPath,
                                                             inBed,
                                                             evalBed,
                                                             verbose)
       # compare
       compPath = os.path.join(args.outputDir,
                               os.path.splitext(base)[0] + "_comp.txt")
       command += " && compareBedStates.py %s %s > %s" % (inBed,
                                                          evalBed,
                                                          compPath)
       commands.append(command)

    runParallelShellCommands(commands, args.numProc)
    #runShellCommand(commands[0])

    
if __name__ == "__main__":
    sys.exit(main())

    
