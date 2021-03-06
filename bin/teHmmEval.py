#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import numpy as np
import math
import copy
import itertools

from teHmm.track import TrackData, CategoryMap
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.trackIO import getMergedBedIntervals, readBedIntervals
from teHmm.modelIO import loadModel
from teHmm.common import myLog, EPSILON, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger
from teHmm.common import runParallelShellCommands, runShellCommand, getLocalTempPath

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a given data set with a trained HMM. Display"
        " the log probability of the input data given the model, and "
        "optionally output the most likely sequence of hidden states.")

    parser.add_argument("tracksInfo", help="Path of Tracks Info file "
                        "containing paths to genome annotation tracks")
    parser.add_argument("inputModel", help="Path of hmm created with"
                        "teHmmTrain.py")
    parser.add_argument("bedRegions", help="Intervals to process")
    parser.add_argument("--bed", help="path of file to write viterbi "
                        "output to (most likely sequence of hidden states)",
                        default=None)
    parser.add_argument("--numThreads", help="Number of threads to use (only"
                        " applies to CFG parser for the moment)",
                        type=int, default=1)
    parser.add_argument("--slice", help="Make sure that regions are sliced"
                        " to a maximum length of the given value.  Most "
                        "useful when model is a CFG to keep memory down. "
                        "When 0, no slicing is done",
                        type=int, default=0)
    parser.add_argument("--segment", help="Use the intervals in bedRegions"
                        " as segments which each count as a single column"
                        " for evaluattion.  Note the model should have been"
                        " trained with the --segment option pointing to this"
                        " same bed file.", action="store_true", default=False)
    parser.add_argument("--segLen", help="Effective segment length used for"
                        " normalizing input segments (specifying 0 means no"
                        " normalization applied)", type=int, default=0)    
    parser.add_argument("--maxPost", help="Use maximum posterior decoding instead"
                        " of Viterbi for evaluation", action="store_true",
                        default=False)
    parser.add_argument("--pd", help="Output BED file for posterior distribution. Must"
                        " be used in conjunction with --pdStates (View on the "
                        "browser via bedGraphToBigWig)", default=None)
    parser.add_argument("--pdStates", help="comma-separated list of state names to use"
                        " for computing posterior distribution.  For example: "
                        " --pdStates inside,LTR_left,LTR_right will compute the probability"
                        ", for each observation, that the hidden state is inside OR LTR_left"
                        " OR LTR_right.  Must be used with --pd to specify output "
                        "file.", default=None)
    parser.add_argument("--bic", help="save Bayesian Information Criterion (BIC) score"
                        " in given file", default=None)
    parser.add_argument("--ed", help="Output BED file for emission distribution. Must"
                        " be used in conjunction with --edStates (View on the "
                        "browser via bedGraphToBigWig)", default=None)
    parser.add_argument("--edStates", help="comma-separated list of state names to use"
                        " for computing emission distribution.  For example: "
                        " --edStates inside,LTR_left for each obsercation the probability "
                        " that inside emitted that observaiton plus the probabillity that"
                        " LTR_left emitted it. If more than one state is selected, this "
                        " is not a distribution, but a sum of distributions (and values"
                        " can exceed 1).  Mostly for debugging purposes. Note output in LOG",
                         default=None)
    parser.add_argument("--chroms", help="list of chromosomes, or regions, to run in parallel"
                        " (in BED format).  input regions will be intersected with each line"
                        " in this file, and the result will correspsond to an individual job",
                        default=None)
    parser.add_argument("--proc", help="number of processes (use in conjunction with --chroms)",
                        type=int, default=1)
    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()
    if args.slice <= 0:
        args.slice = sys.maxint
    elif args.segment is True:
        raise RuntimeError("--slice and --segment options are not compatible at "
                           "this time")
    if (args.pd is not None) ^ (args.pdStates is not None):
        raise RuntimeError("--pd requires --pdStates and vice versa")
    if (args.ed is not None) ^ (args.edStates is not None):
        raise RuntimeError("--ed requires --edStates and vice versa")
    if args.bed is None and (args.pd is not None or args.ed is not None):
        raise RuntimeError("Both --ed and --pd only usable in conjunction with"
                           " --bed")

    if args.chroms is not None:
        # hack to allow chroms argument to chunk and rerun 
        parallelDispatch(argv, args)
        cleanBedTool(tempBedToolPath)
        return 0
    
    # load model created with teHmmTrain.py
    logger.info("loading model %s" % args.inputModel)
    model = loadModel(args.inputModel)

    if isinstance(model, MultitrackCfg):
        if args.maxPost is True:
           raise RuntimeErorr("--post not supported on CFG models")

    # apply the effective segment length
    if args.segLen > 0:
        assert args.segment is True
        model.getEmissionModel().effectiveSegmentLength = args.segLen
        
    # read intervals from the bed file
    logger.info("loading target intervals from %s" % args.bedRegions)
    mergedIntervals = getMergedBedIntervals(args.bedRegions, ncol=4)
    if mergedIntervals is None or len(mergedIntervals) < 1:
        raise RuntimeError("Could not read any intervals from %s" %
                           args.bedRegions)

    # slice if desired
    choppedIntervals = [x for x in slicedIntervals(mergedIntervals, args.slice)]

    # read segment intervals
    segIntervals = None
    if args.segment is True:
        logger.info("loading segment intervals from %s" % args.bedRegions)
        segIntervals = readBedIntervals(args.bedRegions, sort=True)

    # load the input
    # read the tracks, while intersecting them with the given interval
    trackData = TrackData()
    # note we pass in the trackList that was saved as part of the model
    # because we do not want to generate a new one.
    logger.info("loading tracks %s" % args.tracksInfo)
    trackData.loadTrackData(args.tracksInfo, choppedIntervals, 
                            model.getTrackList(),
                            segmentIntervals=segIntervals)

    # do the viterbi algorithm
    if isinstance(model, MultitrackHmm):
        algname = "viterbi"
        if args.maxPost is True:
            algname = "posterior decoding"
        logger.info("running %s algorithm" % algname)
    elif isinstance(model, MultitrackCfg):
        logger.info("running CYK algorithm")

    vitOutFile = None
    if args.bed is not None:
        vitOutFile = open(args.bed, "w")
    totalScore = 0
    tableIndex = 0
    totalDatapoints = 0

    # Note: in general there's room to save on memory by only computing single
    # track table at once (just need to add table by table interface to hmm...)
    
    posteriors = [None] * trackData.getNumTrackTables()
    posteriorsFile = None
    posteriorsMask = None
    if args.pd is not None:
        posteriors = model.posteriorDistribution(trackData)
        posteriorsFile = open(args.pd, "w")
        posteriorsMask = getPosteriorsMask(args.pdStates, model)
        assert len(posteriors[0][0]) == len(posteriorsMask)
    emProbs = [None] * trackData.getNumTrackTables()
    emissionsFile = None
    emissionsMask = None
    if args.ed is not None:
        emProbs = model.emissionDistribution(trackData)
        emissionsFile = open(args.ed, "w")
        emissionsMask = getPosteriorsMask(args.edStates, model)
        assert len(emProbs[0][0]) == len(emissionsMask)

    
    decodeFunction = model.viterbi
    if args.maxPost is True:
        decodeFunction = model.posteriorDecode

    for i, (vitLogProb, vitStates) in enumerate(decodeFunction(trackData,
                                                numThreads=args.numThreads)):
        totalScore += vitLogProb
        if args.bed is not None or args.pd is not None:
            if args.bed is not None:
                vitOutFile.write("#Viterbi Score: %f\n" % (vitLogProb))
            trackTable = trackData.getTrackTableList()[tableIndex]
            tableIndex += 1
            statesToBed(trackTable,
                        vitStates, vitOutFile, posteriors[i], posteriorsMask,
                        posteriorsFile, emProbs[i], emissionsMask, emissionsFile)
            totalDatapoints += len(vitStates) * trackTable.getNumTracks()

    print "Viterbi (log) score: %f" % totalScore
    if isinstance(model, MultitrackHmm) and model.current_iteration is not None:
        print "Number of EM iterations: %d" % model.current_iteration
    if args.bed is not None:
        vitOutFile.close()
    if posteriorsFile is not None:
        posteriorsFile.close()
    if emissionsFile is not None:
        emissionsFile.close()

    if args.bic is not None:
        bicFile = open(args.bic, "w")
        # http://en.wikipedia.org/wiki/Bayesian_information_criterion
        lnL = float(totalScore)
        try:
            k = float(model.getNumFreeParameters())
        except:
            # numFreeParameters still not done for semi-supervised
            # just pass through a 0 instead of crashing for now
            k = 0.0 
        n = float(totalDatapoints)
        bic = -2.0 * lnL + k * (np.log(n) + np.log(2 * np.pi))
        bicFile.write("%f\n" % bic)
        bicFile.write("# = -2.0 * lnL + k * (lnN + ln(2 * np.pi))\n"
                      "# where lnL=%f  k=%d (%d states)  N=%d (%d obs * %d tracks)  lnN=%f\n" % (
            lnL, int(k), model.getEmissionModel().getNumStates(), int(totalDatapoints),
            totalDatapoints / model.getEmissionModel().getNumTracks(),
            model.getEmissionModel().getNumTracks(), np.log(n)))
        bicFile.close()

    cleanBedTool(tempBedToolPath)

def statesToBed(trackTable, states, bedFile,
                posteriors, posteriorsMask, posteriorsFile,
                emProbs, emissionsMask, emissionsFile):
    """write a sequence of states out in bed format. Note: continguous
    intervals with same state no longer merged (since mask support added)
    """
    chrom = trackTable.getChrom()
    start = trackTable.getStart()
    end = trackTable.getEnd()
    segOffsets = trackTable.getSegmentOffsets()
    maskOffsets = trackTable.getMaskRunningOffsets()
    if segOffsets is None:
        assert len(states) == end - start

    segDist = 0
    for i in xrange(len(states)):

        curStart = start + segDist
        
        intLen = 1
        if segOffsets is not None:
            intLen = trackTable.getSegmentLength(i)
        segDist += intLen

        if maskOffsets is not None:
            curStart += maskOffsets[curStart - trackTable.getStart()]

        curEnd = curStart + intLen

        if bedFile is not None:
            bedFile.write("%s\t%d\t%d\t%s\n" % (chrom, curStart, curEnd, states[i]))
            
        if posteriors is not None:
            posteriorsFile.write("%s\t%d\t%d\t%s\n" % (chrom, curStart, curEnd, 
                                 np.sum(posteriors[i-1] * posteriorsMask)))
        if emProbs is not None:
            emissionsFile.write("%s\t%d\t%d\t%s\n" % (chrom, curStart, curEnd, 
                                 np.log(np.sum(np.exp(emProbs[i-1]) * emissionsMask))))

def slicedIntervals(bedIntervals, chunkSize):
    """slice bed intervals by a given length.  used as a quick way to get
    cfg working via cutting up the input beds (after they get merged)."""
    for interval in bedIntervals:
        iLen = interval[2] - interval[1]
        if iLen <= chunkSize:
            yield interval
        else:
            nCuts = int(math.ceil(float(iLen) / float(chunkSize)))
            for sliceNo in xrange(nCuts):
                sInt = list(copy.deepcopy(interval))
                sInt[1] = sliceNo * chunkSize
                if sliceNo < nCuts - 1:
                    sInt[2] = sInt[1] + chunkSize
                assert sInt[2] > sInt[1]
                yield tuple(sInt)

def getPosteriorsMask(pdStates, hmm):
    """ returns array mask where mask[i] == 1 iff state i is part of our desired
    posterior distribution"""
    stateMap = hmm.getStateNameMap()
    if stateMap is None:
        stateMap = CategoryMap(reserved = 0)
        for i in xrange(hmm.getEmissionModel().getNumStates()):
            stateMap.update(str(i))
    mask = np.zeros((len(stateMap)), dtype=np.int8)
    for state in pdStates.split(","):
        if not stateMap.has(state):
            logger.warning("Posterior (or Emission) Distribution state %s"
                           " not found in model" % state)
        else:
            stateNumber = stateMap.getMap(state)
            mask[stateNumber] = 1
    return mask

def parallelDispatch(argv, args):
    """ chunk up input with chrom option.  recursivlely launch eval. merge
    results """
    jobList = []
    chromIntervals = readBedIntervals(args.chroms, sort=True)
    chromFiles = []
    regionFiles = []
    bedFiles = []
    pdFiles = []
    bicFiles = []
    edFiles = []
    for chrom in chromIntervals:
        cmdToks = copy.deepcopy(argv)
        cmdToks[cmdToks.index("--chrom") + 1] = ""
        cmdToks[cmdToks.index("--chrom")] = ""
        
        chromPath = getLocalTempPath("Temp", ".bed")
        cpFile = open(chromPath, "w")
        cpFile.write("%s\t%d\t%d\t0\t0\t.\n" % (chrom[0], chrom[1], chrom[2]))
        cpFile.close()
        
        regionPath = getLocalTempPath("Temp", ".bed")
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (args.bedRegions,
                                                                     chromPath,
                                                                     regionPath))

        if os.path.getsize(regionPath) < 2:
            continue
        
        regionFiles.append(regionPath)
        chromFiles.append(chromPath)

        cmdToks[3] = regionPath

        if args.bed is not None:
            bedPath =  getLocalTempPath("Temp", ".bed")
            cmdToks[cmdToks.index("--bed")+1] = bedPath
            bedFiles.append(bedPath)
        if args.pd is not None:
            pdPath = getLocalTempPath("Temp", ".bed")
            cmdToks[cmdToks.index("--pd")+1] = pdPath
            pdFiles.append(pdPath)
        if args.ed is not None:
            edPath = getLocalTempPath("Temp", ".bed")
            cmdToks[cmdToks.index("--ed")+1] = edPath
            edFiles.append(edPath)
        if args.bic is not None:
            bicPath = getLocalTempPath("Temp", ".bic")
            cmdToks[cmdToks.index("--bic")+1] = bicPath
            bicFiles.append(bicPath)
        cmd = " ".join(cmdToks)
        jobList.append(cmd)

    runParallelShellCommands(jobList, args.proc)

    for i in xrange(len(jobList)):
        if i == 0:
            ct = ">"
        else:
            ct = ">>"
        if len(bedFiles) > 0:
            runShellCommand("cat %s %s %s" % (bedFiles[i], ct, args.bed))
        if len(pdFiles) > 0:
            runShellCommand("cat %s %s %s" % (pdFiles[i], ct, args.pd))
        if len(edFiles) > 0:
            runShellCommand("cat %s %s %s" % (edFiles[i], ct, args.ed))
        if len(bicFiles) > 0:
            runShellCommand("cat %s %s %s" % (bicFiles[i], ct, args.bic))

    for i in itertools.chain(chromFiles, regionFiles, bedFiles, pdFiles, edFiles,
                             bicFiles):
        runShellCommand("rm %s" % i)            
    
    
if __name__ == "__main__":
    sys.exit(main())
