#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import logging
import math
import copy
import re

from pybedtools import BedTool, Interval
from teHmm.common import myLog, runShellCommand, initBedTool, cleanBedTool

"""
Transform a Bed12 file into a Bed6 file where the names are mapped to one of
Intron/Exon (Bed12 "blocks mapped to exons" / remaining region covered by
chromStart - chromEnd mapped to introns")

ex: 

chr1 0 10 BC040516 0 +  0 10 2 3,2 0,6

would be mapped to

chr1 0 3 Exon 0 +
chr1 3 6 Intron 0 +
chr1 6 8 Exon 0 +
chr1 8 10 Intron 0 +

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        description="Transform a Bed12 file into a Bed6 file where the names"
        " are mapped to one of Intron/Exon (Bed12 blocks mapped to exons"
        " / remaining region covered by chromStart - chromEnd mapped to introns)")
    parser.add_argument("inBed", help="bed with chaux results to process")
    parser.add_argument("outBed", help="output bed (will be copy of input)"
                        " if bed12 not detected")
    parser.add_argument("--keepName", help="keep gene names as prefix.  ie"
                        " output will be of form geneName_intron etc.",
                        action="store_true", default=False)
    parser.add_argument("--intron", help="intron name", default="intron")
    parser.add_argument("--exon", help="exon name", default="exon")

    args = parser.parse_args()
    assert os.path.exists(args.inBed)
    tempBedToolPath = initBedTool()
    outFile = open(args.outBed, "w")
    
    # convert bigbed if necessary
    inBed = args.inBed
    if args.inBed[-3:] == ".bb":
        inBed = getLocalTempPath("Temp_cleanGenes", ".bed")
        runShellCommand("bigBedToBed %s %s" % (args.inBed, inBed))

    for interval in BedTool(inBed).sort():
        if len(interval.fields) < 12:
            logger.warning("Input not bed12.. just copying")
            runShellCommand("cp %s %s" % (args.inBed, args.outBed))
            break
        else:
            numBlocks = int(interval.fields[9])
            blockSizes = [int(x) for x in interval.fields[10].split(",")[:numBlocks]]
            blockOffsets = [int(x) for x in interval.fields[11].split(",")[:numBlocks]]
            icopy = copy.deepcopy(interval)
            intron = args.intron
            exon = args.exon

            if args.keepName is True:
                intron = "%s_%s" % (icopy.name, intron)
                exon = "%s_%s" % (icopy.name, exon)

            # edge cases that probably violate bed format
            if numBlocks == 0:
                # no blocks --> one big intron
                icopy.name = intron
                outFile.write(bed6String(icopy))
                continue

            if blockOffsets[0] > 0:
                # gap between start and first block --> intron
                icopy.end = icopy.start + blockOffsets[0]
                icopy.name = intron
                outFile.write(bed6String(icopy))
            
            for i in xrange(numBlocks):
                # write block as exon
                icopy.name = exon
                icopy.start = interval.start + blockOffsets[i]
                icopy.end = icopy.start + blockSizes[i]
                outFile.write(bed6String(icopy))

                if i < numBlocks - 1:
                    gap = blockOffsets[i+1] - (blockOffsets[i] + blockSizes[i])
                    if gap > 0:
                        # room for intron before next block
                        icopy.name = intron
                        icopy.start = icopy.end
                        icopy.end = icopy.start + gap
                        outFile.write(bed6String(icopy))

            gap = interval.end - (interval.start + blockOffsets[-1] + blockSizes[-1])
            if gap > 0:
                # room for intron after last block
                icopy.name = intron
                icopy.start = interval.start + blockOffsets[-1] + blockSizes[-1]
                icopy.end = interval.end
                outFile.write(bed6String(icopy))
                        
    outFile.close()
    cleanBedTool(tempBedToolPath)
    if inBed != args.inBed:
        runShellCommand("rm %s" % inBed)


def bed6String(interval):
    """ blah, can't find way to directly do with bedtools right now"""
    return "%s\t%d\t%d\t%s\t%s\t%s\n" % (interval.chrom,
                                       interval.start,
                                       interval.end,
                                       interval.name,
                                       interval.score,
                                       interval.strand)
    
if __name__ == "__main__":
    sys.exit(main())
