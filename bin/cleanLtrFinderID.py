#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse

from pybedtools import BedTool, Interval
from teHmm.common import intersectSize, getLocalTempPath, runShellCommand
from teHmm.common import initBedTool, cleanBedTool

"""
Clean up bed output of LTR_FINDER (via bed extraction script), to remove unique ids and overlaps so it can be used for HMM. 

output.bed : Uniuqe id's removed.  Overlaps are also removed (giving priority to higher scores (length in case of tied score)

(if --all used:)
output_sym.bed : and right and left removed
output_tsd_as_gap.bed : tsd states removed
output_sym_tsd_as_gap.bed : tsd states removed and right and left removed
output_tsd_as_ltr.bed : tsd states changed to ltr
output_sym_tsd_as_ltr.bed : tsd states changed to ltr right and left removed
output_single.bed : all annotated elems get same id 

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Remove ltr_finder ids from 4th column")
    parser.add_argument("inBed", help="bed with ltr results to process")
    parser.add_argument("outBed", help="bed to write output to.  Will also "
                        "write outBed_sym.bed outBed_tsd_as_gap.bed etc.")
    parser.add_argument("--keepOl", help="by default, if LTR elements "
                        "overlap, the one with the highest score (length "
                        "in event of tie) is kept. This option disables"
                        " this logic.", action="store_true", default=False)
    parser.add_argument("--all", help="write _sym, _tsd_as_gap, etc. versions"
                        " of output", action="store_true", default=False)
    parser.add_argument("--weak", help="score threshold such that any elemetns"
                        " with a score lower or equal to will be assigned the"
                        " prefix WEAK_ to their names.", type=float,
                        default=-1)
    parser.add_argument("--weakIgnore", help="dont apply --weak to state names"
                        " that contain given keywords (defined as comma-separated"
                        " list", default=None)
    
    args = parser.parse_args()
    tempBedToolPath = initBedTool()
    assert os.path.exists(args.inBed)
    baseOut, ext = os.path.splitext(args.outBed)
    if args.weakIgnore is not None:
        args.weakIgnore = args.weakIgnore.split(",")
    else:
        args.weakIgnore = []

    inBed = args.inBed

    toRm = []
    if not args.keepOl:
        inBed = getLocalTempPath("Temp", ".bed")
        removeOverlaps(args.inBed, inBed, args)
        toRm.append(inBed)

    os.system("sed -e \"s/|LTR_TE|[0-9]*//g\" -e \"s/|-//g\" %s > %s" % (
        inBed, args.outBed))

    if args.all:
        symBed = baseOut + "_sym" + ext
        os.system("sed -e \"s/|left//g\" -e \"s/|right//g\" %s > %s" % (args.outBed,
                                                                        symBed))

        tsd_as_gapsBed = baseOut + "_tsd_as_gap" + ext
        os.system("grep -v TSD %s > %s" % (args.outBed, tsd_as_gapsBed))

        sym_tsd_as_gapsBed = baseOut + "_sym_tsd_as_gap" + ext
        os.system("grep -v TSD %s > %s" % (symBed, sym_tsd_as_gapsBed))

        tsd_as_ltrBed = baseOut + "_tsd_as_ltr" + ext
        os.system("sed -e \"s/TSD/LTR/g\" %s > %s" % (args.outBed, tsd_as_ltrBed))

        sym_tsd_as_ltrBed = baseOut + "_sym_tsd_as_ltr" + ext
        os.system("sed -e \"s/TSD/LTR/g\" %s > %s" % (symBed, sym_tsd_as_ltrBed))

        singleBed = baseOut + "_single" + ext
        os.system("sed -e \"s/LTR/inside/g\" %s > %s" % (sym_tsd_as_ltrBed,
                                                         singleBed))

    for path in toRm:
        runShellCommand("rm -f %s" % path)
        
    cleanBedTool(tempBedToolPath)
        

def removeOverlaps(inBed, outBed, args):
    """ Little hack to get this script workign with different settings of ltr_finder
    where annotations can overlap.  To resolve overlaps, we choose the best element
    (by score, then length), and delete anything it touches.  TODO: incorporate Dougs
    script for his lastz stuff? """

    bedIntervals = [x for x in BedTool(inBed).sort()]
    outFile = open(outBed, "w")

    def getLtrID(interval):
        return interval.chrom + interval.name[interval.name.rfind("|") + 1:]
    
    # pass 1: element sizes
    sizes = dict()
    for interval in bedIntervals:
        id = getLtrID(interval)
        length = int(interval.end) - int(interval.start)
        if id in sizes:
            sizes[id] += length
        else:
            sizes[id] = length

    # pass 2: greedy kill (not optimal for all transitive cases)
    # strategy: any pairwise overlap will be detected in either
    # the left or right scan of at least one of the overlapping
    # elements. 
    dead = set()
    for i, interval in enumerate(bedIntervals):
        id = getLtrID(interval)
        size = sizes[id]
        if id in dead:
            continue
        for j in xrange(i-1, -1, -1):
            if intersectSize((interval.chrom, interval.start, interval.end),
                             (bedIntervals[j].chrom, bedIntervals[j].start,
                             bedIntervals[j].end)) <= 0:
                break
            otherId = getLtrID(bedIntervals[j])
            if otherId in dead:
                continue
            if (bedIntervals[j].score > interval.score or
                (bedIntervals[j].score == interval.score and
                 sizes[otherId] > size)):
                dead.add(id)
                break
            else:
                dead.add(otherId)
        if id in dead:
            continue
        for j in xrange(i+1, len(bedIntervals), 1):
            if intersectSize((interval.chrom, interval.start, interval.end),
                             (bedIntervals[j].chrom, bedIntervals[j].start,
                             bedIntervals[j].end)) <= 0:
                break
            otherId = getLtrID(bedIntervals[j])
            if otherId in dead:
                continue
            if (bedIntervals[j].score > interval.score or
                (bedIntervals[j].score == interval.score and
                 sizes[otherId] > size)):
                dead.add(id)
                break
            else:
                dead.add(otherId)
        if id in dead:
            continue

    # pass 3: write non-killed
    for interval in bedIntervals:
        id = getLtrID(interval)
        if id not in dead:
            if interval.strand == "?":
                interval.strand = "."
            applyWeak(interval, args)
            outFile.write(str(interval))
            
def applyWeak(interval, args):
    # YOU ARE WEAK
    if float(interval.score) <= args.weak:
        for keyIg in args.weakIgnore:
            if keyIg in interval.name:
                return
        interval.name = "WEAK_" + interval.name
        

    
if __name__ == "__main__":
    sys.exit(main())
