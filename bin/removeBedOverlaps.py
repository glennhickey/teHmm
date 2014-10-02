#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import copy

from pybedtools import BedTool, Interval
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLocalTempPath, runShellCommand

"""
Filter out bed intervals that overlap other intervals.

Algorithm:  for each interval in sorted list, cut (or remove) such that it
doesn't overlap any intervals before it in the list. 
"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Filter overlapping intervals out")
    parser.add_argument("inputBed", help="Bed file to filter")
    parser.add_argument("--bed12", help="Use bed12 exons instead of start/end"
                        " if present (equivalent to running bed12ToBed6 on"
                        " input first).", action="store_true", default=False)
    parser.add_argument("--rm", help="Make sure intervals that are labeled as TE "
                        "by rm2State.sh script are never cut by ones that are not",
                        default=False, action='store_true')
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    assert os.path.isfile(args.inputBed)
    tempBedToolPath = initBedTool()

    # do the --rm filter.  by splitting into TE / non-TE
    # then removing everything in non-TE that overlaps
    # TE.  The adding the remainder back to TE. 
    inputPath = args.inputBed
    if args.rm is True:
        tempPath = getLocalTempPath("Temp_", ".bed")
        tePath = getLocalTempPath("Temp_te_", ".bed")
        runShellCommand("rm2State.sh %s |grep TE | sortBed > %s" % (
            args.inputBed, tempPath))
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" %(
            args.inputBed, tempPath, tePath))
        otherPath = getLocalTempPath("Temp_other_", ".bed")
        runShellCommand("rm2State.sh %s |grep -v TE | sortBed > %s" % (
            args.inputBed, tempPath))
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" %(
            args.inputBed, tempPath, otherPath))
        if os.path.getsize(tePath) > 0  and\
           os.path.getsize(otherPath) > 0:
            filterPath = getLocalTempPath("Temp_filter_", ".bed")
            runShellCommand("subtractBed -a %s -b %s | sortBed > %s" % (
                otherPath, tePath, filterPath))
            inputPath = getLocalTempPath("Temp_input_", ".bed")
            runShellCommand("cat %s %s | sortBed > %s" % (
                tePath, filterPath, inputPath))
            runShellCommand("rm -f %s" % filterPath)
        runShellCommand("rm -f %s %s %s" % (tePath, otherPath, tempPath))

    bedIntervals = BedTool(inputPath).sort()
    if args.bed12 is True:
        bedIntervals = bedIntervals.bed6()
        
    prevInterval = None

    # this code has been way to buggy for something so simple
    # keep extra list to check for sure even though it's a waste of
    # time and space
    sanity = []
    
    for interval in bedIntervals:
        if (prevInterval is not None and
            interval.chrom == prevInterval.chrom and
            interval.start < prevInterval.end):
            logger.debug("Replace %d bases of \n%s with\n%s" % (
                prevInterval.end - interval.start,
                str(interval), str(prevInterval)))
            interval.start = prevInterval.end
            
        if interval.end > interval.start:
            sys.stdout.write("%s" % str(interval))
            sanity.append(interval)
            prevInterval = interval

    for i in xrange(len(sanity) - 1):
        if sanity[i].chrom == sanity[i+1].chrom:
            assert sanity[i+1].start >= sanity[i].end
    cleanBedTool(tempBedToolPath)
    if args.inputBed != inputPath:
        runShellCommand("rm -f %s" % inputPath)

if __name__ == "__main__":
    sys.exit(main())
