#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import sys
import os
import argparse
import logging
import numpy as np
from collections import defaultdict

from teHmm.trackIO import readBedIntervals, fastaRead, writeBedIntervals
from teHmm.kmer import KmerTable, hashDNA
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import getLocalTempPath, runParallelShellCommands
from teHmm.common import runShellCommand

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
                        default=6, type=int)
    parser.add_argument("--all", help="Report all matches in region (as opposed"
                        " to only the nearest to the BED element which is the "
                        "default behaviour", action="store_true", default=False)
    parser.add_argument("--maxScore", help="Only report matches with given "
                        "score or smaller.  The score  is definied as the "
                        "maximum distance between the (two) TSD intervals and "
                        "the query interval",
                        default=None, type=int)
    parser.add_argument("--left", help="Number of bases immediately left of the "
                        "BED element to search for the left TSD",
                        default=7, type=int)
    parser.add_argument("--right", help="Number of bases immediately right of "
                        "the BED element to search for the right TSD",
                        default=7, type=int)
    parser.add_argument("--overlap", help="Number of bases overlapping the "
                        "BED element to include in search (so total space "
                        "on each side will be --left + overlap, and --right + "
                        "--overlap", default=3, type=int)
    parser.add_argument("--leftName", help="Name of left TSDs in output Bed",
                        default="L_TSD")
    parser.add_argument("--rightName", help="Name of right TSDs in output Bed",
                        default="R_TSD")
    parser.add_argument("--id", help="Assign left/right pairs of TSDs a unique"
                        " matching ID", action="store_true", default=False)
    parser.add_argument("--names", help="Only apply to bed interval whose "
                        "name is in (comma-separated) list.  If not specified"
                        " then all intervals are processed", default=None)
    parser.add_argument("--numProc", help="Number of jobs to run in parallel."
                        " (parallization done on different sequences in FASTA"
                        "file", type=int, default=1)
    parser.add_argument("--sequences", help="Only process given sequences of input"
                        " FASTA file (comma-separated list).",  default=None)
    
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)

    assert os.path.exists(args.inBed)
    assert os.path.exists(args.fastaSequence)
    assert args.min <= args.max
    args.nextId = 0

    if args.sequences is not None:
        args.sequences = set(args.sequences.split(","))

    # read intervals from the bed file
    logger.info("loading target intervals from %s" % args.inBed)
    bedIntervals = readBedIntervals(args.inBed, ncol=4, sort=True)
    if bedIntervals is None or len(bedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.inBed)

    if args.numProc > 1:
        runParallel(args, bedIntervals)
        return 0
    
    tsds = findTsds(args, bedIntervals)

    writeBedIntervals(tsds, args.outBed)

def runParallel(args, bedIntervals):
    """ Quick hack to rerun parallel jobs on different interval subsets. """
    nameSet = None
    if args.names is not None:
        nameSet = set(args.names.split(","))
        
    # chunk up BED input
    numIntervals = 0
    for interval in bedIntervals:
        name = None
        if len(interval) > 3:
            name = interval[3]
        if nameSet is None or name in nameSet:
            numIntervals += 1
    jobSize = 1 + (numIntervals / args.numProc)
    logger.info("Dviding %d intervals into %d processes (%d intervals per)" % (
        numIntervals, args.numProc, jobSize))
    tempBeds = []
    curSize = sys.maxint
    curFile = None
    for interval in bedIntervals:
        name = None
        if len(interval) > 3:
            name = interval[3]
        if nameSet is None or name in nameSet:
            if curSize >= jobSize:
                if curFile is not None:
                    curFile.close()
                tempBed = getLocalTempPath("TempTsdFinderIn", ".bed")
                tempBeds.append(tempBed)
                curFile = open(tempBed, "w")
                curSize = 0
            curFile.write("\t".join([str(s) for s in interval]))
            curFile.write("\n")
            curSize += 1
    if curFile is not None:
        curFile.close()

    # map jobs
    assert len(tempBeds) <= args.numProc
    tempOuts = []
    jobCmds = []
    for tempBed in tempBeds:
        cmdLine = " ".join(sys.argv)
        cmdLine = cmdLine.replace("--numProc %d" % args.numProc,"--numProc 1")
        cmdLine = cmdLine.replace(args.inBed, tempBed)
        tempOut = getLocalTempPath("TempTsdFinderOut", ".bed")
        cmdLine = cmdLine.replace(args.outBed, tempOut)
        tempOuts.append(tempOut)
        jobCmds.append(cmdLine)
        
    runParallelShellCommands(jobCmds, args.numProc)

    # reduce
    for i, tempOut in enumerate(tempOuts):
        if i == 0:
            runShellCommand("mv %s %s" % (tempOut, args.outBed))
        else:
            runShellCommand("cat %s >> %s" % (tempOut, args.outBed))
            runShellCommand("rm -f %s" % (tempOut))

def buildSeqTable(args, bedIntervals):
    """build table of sequence indexes from input bed file to quickly read 
    while sorting.  Table maps sequence name to range of indexes in 
    bedIntervals.  This only works if bedIntervals are sorted (and should 
    raise an assertion error if that's not the case. 
    """
    logger.info("building sequence name index of %d bed intervals" %
                 len(bedIntervals))
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
        
    
def findTsds(args, bedIntervals):
    """ search through input bed intervals, loading up the FASTA sequence
    for each one """
    
    # index for quick lookups in bed file (to be used while scanning fasta file)
    seqTable = buildSeqTable(args, bedIntervals)
    outTsds = []
    faFile = open(args.fastaSequence, "r")
    nameSet = None
    if args.names is not None:
        nameSet = set(args.names.split(","))
    for seqNameFa, sequence in fastaRead(faFile):
        if args.sequences is not None and seqNameFa not in args.sequences and\
          seqNameFa.split()[0] not in args.sequences:
          # skip unflagged sequences when option specified
          continue
            
        # try name from Fasta as well as name with everything after first
        # whitespace skipped
        if seqNameFa in seqTable:
            seqName = seqNameFa
        else:
            seqName = seqNameFa.split()[0]
        if seqName in seqTable:
            logger.info("Scanning FASTA sequence %s" % seqName)
            bedRange = seqTable[seqName]
            for bedIdx in xrange(bedRange[0], bedRange[1]):
                bedInterval = bedIntervals[bedIdx]
                name = None
                if len(bedInterval) > 3:
                    name = bedInterval[3]
                if nameSet is None or name in nameSet:
                    # we make sequence lower case below because we dont care
                    # about soft masking
                    outTsds += intervalTsds(args, sequence.lower(), bedInterval)
        else:
            logger.info("Skipping FASTA sequence %s because no intervals "
                          "found" % seqName)

    return outTsds

def getScore(bedInterval, l1, l2, match):
    """ Given a bedInterval, the two window (left coords), and a candidate
    mathc, compute the score of the match (lower is better)"""
    d1 = np.abs(bedInterval[1] - (l1 + match[1]))
    assert d1 >= 0
    d2 = np.abs(bedInterval[2] - (l2 + match[2]))
    return max(d1, d2)

def intervalTsds(args, sequence, bedInterval):
    """ given a single bed interval, do a string search to find tsd candidates
    on the left and right flank."""
    overlap = min(args.overlap, (bedInterval[2] - bedInterval[1]) / 2 - 2)
    overlap = max(0, overlap)
    l1 = max(0, bedInterval[1] - args.left)
    r1 = bedInterval[1] - 1 + overlap

    l2 = bedInterval[2] - overlap
    r2 = min(bedInterval[2] - 1 + args.right, len(sequence) - 1)

    if r1 - l1 < args.min or r2 - l2 < args.min:
        return []

    kt = KmerTable(kmerLen = args.min, hashFn = hashDNA)
    leftFlank = sequence[l1:r1]
    rightFlank = sequence[l2:r2]
    assert l2 > r1
    kt.loadString(rightFlank)
    matches = kt.exactMatches(leftFlank, minMatchLen = args.min,
                              maxMatchLen = args.max)

    # filter by score if applicable
    if args.maxScore is not None:
        filterMatches = []
        for match in matches:
            if getScore(bedInterval, l1, l2, match) <= args.maxScore:
                filterMatches.append(match)
        matches = filterMatches        

    # if we don't want every match, find the match with the lowest maximum
    # distance to the interval. will probably need to look into better 
    # heuristics for this
    if args.all is False and len(matches) > 1:
        dmin = len(sequence)
        bestLen = 0
        bestMatch = None
        for match in matches:
            d = getScore(bedInterval, l1, l2, match)
            matchLen = match[1] - match[0]
            assert matchLen == match[3] - match[2]
            if d < dmin or (d == dmin and matchLen > bestLen):
                dmin = d
                bestMatch = match
                bestLen = matchLen
        matches = [bestMatch]

    tsds = matchesToBedInts(args, bedInterval, matches, l1, l2)

    # sanity check
    assert len(tsds) % 2 == 0
    for i in xrange(0, len(tsds), 2):
        lt = tsds[i]
        rt = tsds[i+1]
        assert sequence[lt[1]:lt[2]].lower() == sequence[rt[1]:rt[2]].lower()

    return tsds

def matchesToBedInts(args, bedInterval, matches, l1, l2):
    """ convert substring matches as returned from the kmer table into bed 
    intervals that will be output by the tool"""

    bedIntervals = []
    for match in matches:
        assert len(match) == 4
        name = args.leftName
        if args.id is True:
            name += "_" + str(args.nextId)
        left = (bedInterval[0], l1 + match[0], l1 + match[1], name,
                np.abs(bedInterval[1] - (l1 + match[1])))            
        bedIntervals.append(left)

        name = args.rightName
        if args.id is True:
            name += "_" + str(args.nextId)
        right = (bedInterval[0], l2 + match[2], l2 + match[3], name,
                 np.abs(bedInterval[2] - (l2 + match[2])))
        bedIntervals.append(right)

        args.nextId += 1
        
    return bedIntervals
        
    
        
    
if __name__ == "__main__":
    sys.exit(main())
