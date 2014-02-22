#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import logging

from teHmm.trackIO import getMergedBedIntervals, fastaRead, writeBedIntervals
from teHmm.kmer import KmerTable
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger

"""
Find candidate target site duplications (TSD's).  These are short *exact* matches
on the forward strand that flank transposable elements (TEs).  This script takes
 as input a BED file identifying candidate TEs, and the genome sequence in FASTA
 format.  Candidate TSDs are searched for immediately before and after each 
 interval in the BED.  Note that contidugous BED intervals will be treated as a
 single interval.  

"""

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Find candidate TSDs (exact forward matches) flanking given"
        "BED intervals.  Score is distance between TSD and bed interval.")
    parser.add_argument("fastaSequence", help="DNA sequence in FASTA format")
    parser.add_argument("inBed", help="BED file with TEs whose flanking regions "
                        "we wish to search")
    parser.add_argument("outBed", help="BED file containing (only) output TSDs")
    parser.add_argument("--min", help="Minimum length of a TSD",
                        default=4, type=int)
    parser.add_argument("--max", help="Maximum length of a TSD",
                        default=8, type=int)
    parser.add_argument("--all", help="Report all matches in region (as opposed"
                        " to only the nearest to the BED element which is the "
                        "default behaviour", action="store_true", default=False)
    parser.add_argument("--left", help="Number of bases immediately left of the "
                        "BED element to search for the left TSD",
                        default=20, type=int)
    parser.add_argument("--right", help="Number of bases immediately right of "
                        "the BED element to search for the right TSD",
                        default=20, type=int)
    parser.add_argument("--leftName", help="Name of left TSDs in output Bed",
                        default="L_TSD")
    parser.add_argument("--rightName", help="Name of right TSDs in output Bed",
                        default="R_TSD")
    parser.add_argument("--id", help="Assign left/right pairs of TSDs a unique"
                        " matching ID", action="store_true", default=False)
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)

    assert os.path.exists(args.inBed)
    assert os.path.exists(args.fastaSequence)
    assert args.min <= args.max
    args.nextId = 0

    # read intervals from the bed file
    logger.info("loading target intervals from %s" % args.inBed)
    mergedIntervals = getMergedBedIntervals(args.inBed, ncol=4, sort=True)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.inBed)
    
    tsds = findTsds(args, mergedIntervals)

    writeBedIntervals(tsds, args.outBed)


def buildSeqTable(bedIntervals):
    """build table of sequence indexes from input bed file to quickly read 
    while sorting.  Table maps sequence name to range of indexes in 
    bedIntervals.  This only works if bedIntervals are sorted (and should 
    raise an assertion error if that's not the case. 
    """
    logger.debug("building index of %d bed intervals" % len(bedIntervals))
    bedSeqTable = dict()
    prevName = None
    prevIdx = 0
    for i, interval in enumerate(bedIntervals):
        seqName = interval[0]
        if seqName != prevName:
            assert seqName not in bedSeqTable
            if prevName is not None:
                bedSeqTable[prevName] = (prevIdx, i)
                prevIdx = i
        prevName = seqName
    seqName = bedIntervals[-1][0]
    assert seqName not in bedSeqTable
    bedSeqTable[seqName] = (prevIdx, len(bedIntervals))
    logger.debug("index has %d unique sequences" % len(bedSeqTable))
    return bedSeqTable
        
    
def findTsds(args, mergedIntervals):
    """ search through input bed intervals, loading up the FASTA sequence
    for each one """
    
    # index for quick lookups in bed file (to be used while scanning fasta file)
    seqTable = buildSeqTable(mergedIntervals)
    outTsds = []
    faFile = open(args.fastaSequence, "r")
    for seqName, sequence in fastaRead(faFile):
        if seqName in seqTable:
            logger.debug("Scanning FASTA sequence %s" % seqName)
            bedRange = seqTable[seqName]
            for bedIdx in xrange(bedRange[0], bedRange[1]):
                bedInterval = mergedIntervals[bedIdx]
                # we make sequence lower case below because we dont care
                # about soft masking
                outTsds += intervalTsds(args, sequence.lower(), bedInterval)
        else:
            logger.debug("Skipping FASTA sequence %s because no intervals "
                          "found" % seqName)

    return outTsds

def intervalTsds(args, sequence, bedInterval):
    """ given a single bed interval, do a string search to find tsd candidates
    on the left and right flank."""
    l1 = max(0, bedInterval[1] - args.left)
    r1 = bedInterval[1]

    l2 = bedInterval[2]
    r2 = min(bedInterval[2] + args.right, len(sequence))

    if r1 - l1 < args.min or r2 - l2 < args.min:
        return []

    kt = KmerTable(kmerLen = args.min)
    leftFlank = sequence[l1:r1]
    rightFlank = sequence[l2:r2]
    assert l2 > r1
    kt.loadString(rightFlank)
    matches = kt.exactMatches(leftFlank, minMatchLen = args.min,
                              maxMatchLen = args.max)

    # if we don't want every match, find the match with the lowest minimum
    # distance to the interval. will probably need to look into better 
    # heuristics for this
    if args.all is False and len(matches) > 1:
        dmin = len(sequence)
        bestMatch = None
        for match in matches:
            d = bedInterval[1] - match[1]
            assert d >= 0
            d += match[2] - bedInterval[2]
            if d < dmin:
                dmin = d
                bestMatch = match
        matches = [bestMatch]

    return matchesToBedInts(args, bedInterval, matches)

def matchesToBedInts(args, bedInterval, matches):
    """ convert substring matches as returned from the kmer table into bed 
    intervals that will be output by the tool"""

    bedIntervals = []
    for match in matches:
        assert len(match) == 4
        name = args.leftName
        if args.id is True:
            name += "_" + str(args.nextId)
        offset = bedInterval[1] - args.left
        left = (bedInterval[0], offset + match[0], offset + match[1], name,
                bedInterval[1] - offset - match[1])
        bedIntervals.append(left)

        name = args.rightName
        if args.id is True:
            name += "_" + str(args.nextId)
        offset = bedInterval[2]
        right = (bedInterval[0], offset + match[2], offset + match[3], name,
                 offset + match[2] - bedInterval[2])
        bedIntervals.append(right)

        args.nextId += 1
        
    return bedIntervals
        
        
        
        

        
    
    
    
                
                
        
    
if __name__ == "__main__":
    sys.exit(main())
