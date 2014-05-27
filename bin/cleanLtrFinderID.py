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
Take the Bed output of ltr_finder, and make some different bed files that we can use for the model by changing the ids

output.bed : Uniuqe id's removed
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
    
    args = parser.parse_args()
    tempBedToolPath = initBedTool()
    assert os.path.exists(args.inBed)
    baseOut, ext = os.path.splitext(args.outBed)

    inBed = args.inBed

    toRm = []
    if not args.keepOl:
        inBed = getLocalTempPath("Temp", ".bed")
        removeOverlaps(args.inBed, inBed)
        toRm.append(inBed)

    os.system("sed -e \"s/|LTR_TE|[0-9]*//g\" -e \"s/|-//g\" %s > %s" % (
        inBed, args.outBed))

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
        

def removeOverlaps(inBed, outBed):
    """ Little hack to get this script workign with different settings of ltr_finder
    where annotations can overlap.  To resolve overlaps, we choose the best element
    (by score, then length), and delete anything it touches.  TODO: incorporate Dougs
    script for his lastz stuff? """

    bedIntervals = [x for x in BedTool(inBed).sort()]
    outFile = open(outBed, "w")

    def getLtrID(tok):
        return int(tok[tok.rfind("|") + 1:])
    
    # pass 1: element sizes
    sizes = dict()
    for interval in bedIntervals:
        id = getLtrID(interval.name)
        length = int(interval.end) - int(interval.start)
        if id in sizes:
            sizes[id] += length
        else:
            sizes[id] = length

    # pass 2: greedy kill (not optimal for all transitive cases)
    dead = set()
    for i, interval in enumerate(bedIntervals):
        id = getLtrID(interval.name)
        size = sizes[id]
        for j in xrange(i-1, -1, -1):
            if intersectSize((interval.chrom, interval.start, interval.end),
                             (bedIntervals[j].chrom, bedIntervals[i].start,
                             bedIntervals[j].end)) <= 0:
                break
            otherId = getLtrID(bedIntervals[j].name)
            if otherId not in dead and (
                    bedIntervals[j].score > interval.score or
                    (bedIntervals[j].score == interval.score and
                   sizes[otherId] > size)):
                dead.add(id)
                break
        for j in xrange(i+1, 1, len(bedIntervals)):
            if intersectSize((interval.chrom, interval.start, interval.end),
                             (bedIntervals[j].chrom, bedIntervals[i].start,
                             bedIntervals[j].end)) <= 0:
                break
            otherId = getLtrID(bedIntervals[j].name)
            if otherId not in dead and (
                    bedIntervals[j].score > interval.score or
                    (bedIntervals[j].score == interval.score and
                   sizes[otherId] > size)):
                dead.add(id)
                break

    # pass 3: write non-killed
    for interval in bedIntervals:
        id = getLtrID(interval.name)
        if id not in dead:
            outFile.write(str(interval))
            
        
if __name__ == "__main__":
    sys.exit(main())
