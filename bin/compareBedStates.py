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
import copy
import ast
import itertools
from collections import defaultdict

from teHmm.trackIO import readBedIntervals
from teHmm.common import intersectSize, initBedTool, cleanBedTool
from teHmm.common import logger, getLocalTempPath, runShellCommand
from teHmm.track import TrackList

try:
    from teHmm.parameterAnalysis import pcaFlatten, plotPoints2d
    canPlot = True
except:
    canPlot = False

""" Compare bed files (EX Truth vs. Viterbi output).  They must cover same
genomic region in the same order """

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare two bed files where Model states are represented"
        " in a column.  Used to determine sensitivity and specificity.  NOTE"
        " that both bed files must be sorted and cover the exact same regions"
        " of the same genome.")

    parser.add_argument("bed1", help="Bed file (TRUTH)")
    parser.add_argument("bed2", help="Bed file covering same regions in same"
                        " order as bed1")
    parser.add_argument("--col", help="Column of bed files to use for state"
                        " (currently only support 4(name) or 5(score))",
                        default = 4, type = int)
    parser.add_argument("--thresh", help="Threshold to consider interval from"
                        " bed1 covered by bed2.",
                        type=float, default=0.8)
    parser.add_argument("--plot", help="Path of file to write Precision/Recall"
                        " graphs to in PDF format", default=None)
    parser.add_argument("--ignore", help="Comma-separated list of stateNames to"
                        " ignore", default=None)
    parser.add_argument("--strictPrec", help="By default, precision is computed"
                        " in a manner strictly symmetric to recall.  So calling"
                        " compareBedStates.py A.bed B.bed would give the exact"
                        " same output as compareBedStates.py B.bed A.bed except"
                        " precision and recall values would be swapped.  With "
                        " this option, a predicted element only counts toward"
                        " precision if it overlaps with 80pct of the true"
                        " element, as opposed to only needing 80pct of itself"
                        " overlapping with the true element. ",
                        action="store_true", default = False)
    parser.add_argument("--noBase", help="Skip base-level stats (and only show"
                        " interval stats).  Runs faster", action="store_true",
                        default=False)
    parser.add_argument("--noFrag", help="Do not allow fragmented matches in"
                        "interval predictions.  ie if a single truth interval"
                        " is covered by a series of predicted intervals, only "
                        "the best match will be counted if this flag is used", 
                        action="store_true", default=False)
    parser.add_argument("--tl", help="Path to tracks XML file.  Used to cut "
                        "out mask tracks so they are removed from comparison."
                        " (convenience option to not have to manually run "
                        "subtractBed everytime...)", default=None)
    parser.add_argument("--delMask", help="Entirely remove intervals from "
                        "mask tracks that are > given length.  Probably "
                        "only want to set to non-zero value K if using"
                        " with a prediction that was processed with "
                        "interpolateMaskedRegions.py --max K",
                        type=int, default=0)
    parser.add_argument("--window", help="A comma-delimited 5-tuple of "
                        "windowSize,stateName,compType,score,outBed.  "
                        "Where windowSize  is the sliding window size "
                        "(overlap .5), stateName is target stateName,"
                        " compType is in {base,interval,weighted}, sore is"
                        " in {f1,precision,recall} and "
                        "outBed is the path of a bedFile to write positional"
                        " accuracy to.  For example, --window 1000000,TE,base,f1"
                        ",acc.bed will write base-level f1 for 1MB sliding windows"
                        " to acc.bed.  These can be viewed on the browser by first"
                        " converting to BigWig.", default=None)

    args = parser.parse_args()
    tempBedToolPath = initBedTool()

    if args.ignore is not None:
        args.ignore = set(args.ignore.split(","))
    else:
        args.ignore = set()

    assert args.col == 4 or args.col == 5
    print "Commandline %s" % " ".join(sys.argv)
    origArgs = copy.deepcopy(args)
    
    tempFiles = []
    if args.tl is not None:
        cutBed1 = cutOutMaskIntervals(args.bed1, args.delMask,
                                      sys.maxint, args.tl)
        cutBed2 = cutOutMaskIntervals(args.bed2, args.delMask,
                                      sys.maxint, args.tl)
        if cutBed1 is not None:
            assert cutBed2 is not None
            tempFiles += [cutBed1, cutBed2]
            args.bed1 = cutBed1
            args.bed2 = cutBed2

    checkExactOverlap(args.bed1, args.bed2)

    if args.window is not None:
        runPositionalComparison(argv, origArgs)

    intervals1 = readBedIntervals(args.bed1, ncol = args.col)
    intervals2 = readBedIntervals(args.bed2, ncol = args.col)

    if args.noBase is False:
        stats = compareBaseLevel(intervals1, intervals2, args.col - 1)[0]

        totalRight, totalWrong, accMap = summarizeBaseComparision(stats, args.ignore)
        print "Base counts [False Negatives, False Positives, True Positives]:"
        print stats
        totalBoth = totalRight + totalWrong
        accuracy = float(totalRight) / float(totalBoth)
        print "Accuaracy: %d / %d = %f" % (totalRight, totalBoth, accuracy)
        print "State-by-state (Precision, Recall):"
        print "Base-by-base Accuracy"    
        print accMap

    trueStats = compareIntervalsOneSided(intervals1, intervals2, args.col -1,
                                         args.thresh, False, not args.noFrag)[0]
    predStats = compareIntervalsOneSided(intervals2, intervals1, args.col -1,
                                         args.thresh, args.strictPrec,
                                         not args.noFrag)[0]
    intAccMap = summarizeIntervalComparison(trueStats, predStats, False,
                                            args.ignore)
    intAccMapWeighted = summarizeIntervalComparison(trueStats, predStats, True,
                                                     args.ignore)
    print "\nInterval Accuracy"
    print intAccMap
    print ""

    print "\nWeighted Interval Accuracy"
    print intAccMapWeighted
    print ""


    # print some row data to be picked up by scrapeBenchmarkRow.py
    if args.noBase is False:
        header, row = summaryRow(accuracy, stats, accMap)
        print " ".join(header)
        print " ".join(row)

    # make graph
    if args.plot is not None:
        if canPlot is False:
            raise RuntimeError("Unable to write plots.  Maybe matplotlib is "
                               "not installed?")
        writeAccPlots(accuracy, accMap, intAccMap, intAccMapWeighted,
                      args.thresh, args.plot)

    if len(tempFiles) > 0:
        runShellCommand("rm -f %s" % " ".join(tempFiles))
    cleanBedTool(tempBedToolPath)

def compareBaseLevel(intervals1, intervals2, col):
    """ return dictionary that maps each state to (i1 but not i2, i2 but not i1,
    both) for base level stats.  Update: now returns a tuple of accuracy stats
    and confusionf matrix. """

    assert intervals1[0][0] == intervals2[0][0]
    assert intervals1[0][1] == intervals2[0][1]
    assert intervals1[-1][2] == intervals2[-1][2]

    # base level dictionary
    stats = dict()
    # confusion matrix
    confMat = dict()
    # yuck:
    p2 = 0
    for p1 in xrange(len(intervals1)):
        i1 = intervals1[p1]
        assert len(i1) > col
        for pos in xrange(i1[2] - i1[1]):
            i2 = intervals2[p2]
            assert len(i2) > col
            chrom = i1[0]
            coord = i1[1] + pos
            if i2[0] != chrom or not (coord >= i2[1] and coord < i2[2]):
                p2 += 1
                i2back = i2
                i2 = intervals2[p2]
                if not (i2[0] == chrom and coord >= i2[1] and coord < i2[2]):
                    raise RuntimeError("Error comparing %s and %s: Perhaps input"
                                       " files are not sorted or do not exactly"
                                       " overlap?" % (
                        str(intervals1[p1]), str(intervals2[p2])))
            state1 = i1[col]
            state2 = i2[col]
            if state1 not in stats:
                stats[state1] = [0, 0, 0]
            if state2 not in stats:
                stats[state2] = [0, 0, 0]
            if state1 == state2:
                stats[state1][2] += 1
            else:
                stats[state1][0] += 1
                stats[state2][1] += 1
            updateConfMatrix(confMat, state2, state1)

    return stats, confMat

def compareIntervalsOneSided(trueIntervals, predIntervals, col, threshold,
                             usePredLenForThreshold, allowMultipleMatches):
    """ Same idea as baselevel comparison above, but treats bed intervals
    as single unit, and does not perform symmetric test.  In particular, we
    return the following stats here: for each true interval, is it covered
    by a predicted interval (with the same name) by at least threshold pct?
    The stats returned is therefore a pair for each state:
    (num intervals in truth correctly predicted , num intervals in truth
    incorrectly predicted)
    This is effectively a recall measure.  Of course, calling a second time
    with truth and pred swapped, will yield the precision.

    We also include the total lengths of the true predicted and false predicted
    elements.  So each states maps to a tuplie like
    (numTrue, totTrueLen, numFalse, totFalseLen)

    the usePredLenForThreshold option is activated by the args.strictPrec
    flag when computing prediction vs truth (see description of this flag
    for what it does)

    NOTE: this test will return a positive hit if a giant predicted interval
    overlaps a tiny true interval.  this can be changed, but since this form
    of innacuracy will be caught when called with true/pred swapped (precision)
    I'm not sure if it's necessary
    """

    # as in base level comp, both interval sets must cover exactly same regions
    # in same order.  the asserts below only partially check this:
    assert trueIntervals[0][0] == predIntervals[0][0]
    assert trueIntervals[0][1] == predIntervals[0][1]
    assert trueIntervals[-1][2] == predIntervals[-1][2]

    LP = len(predIntervals)
    LT = len(trueIntervals)

    stats = dict()
    confMat = dict()
    
    pi = 0
    for ti in xrange(LT):

        trueInterval = trueIntervals[ti]
        trueState = trueInterval[col]
        trueLen = float(trueInterval[2] - trueInterval[1])
        
        # advance pi to first pred interval that intersects ti
        while True:
            if pi < LP and intersectSize(trueInterval,
                                         predIntervals[pi]) == 0:
                pi += 1
            else:
                break

        # scan all intersecting predIntervals with ti
        bestFrac = 0.0
        totalFrac = 0.0
        for i in xrange(pi, LP):
            overlapSize = intersectSize(trueInterval, predIntervals[i])
            if overlapSize > 0:
                denom = trueLen
                if usePredLenForThreshold is True:
                    denom = float(predIntervals[i][2] - predIntervals[i][1])
                frac = float(overlapSize) / denom
                # look for biggest true overlap when computing accuracy
                if predIntervals[i][col] == trueState:
                    bestFrac = max(bestFrac, frac)
                    # compute total overlap for allowMultipleMatches option
                    totalFrac += frac
                # count all overlaps >= thresh when computing confusion matrix
                if frac >= threshold:
                    updateConfMatrix(confMat, predIntervals[i][col], trueState)
            else:
                break

        if allowMultipleMatches is True:
            bestFrac = totalFrac

        # update stats
        if trueState not in stats:
            stats[trueState] = [0, 0, 0, 0]

        if bestFrac >= threshold:
            stats[trueState][0] += 1
            stats[trueState][1] += trueLen
        else:
            # dont really need this (can be inferred from total number of
            # true intervals but whatever)
            stats[trueState][2] += 1
            stats[trueState][3] += trueLen

    return stats, confMat
    
def summarizeBaseComparision(stats, ignore):
    totalRight = 0
    totalWrong = 0
    accMap = dict()
    for state, stat in stats.items():
        if state in ignore:
            continue
        totalRight += stat[2]
        totalWrong += stat[0] + stat[1]
        tp = float(stat[2])
        fn = float(stat[0])
        fp = float(stat[1])
        accMap[state] = (tp / (np.finfo(float).eps + tp + fp),
                         tp / (np.finfo(float).eps + tp + fn))
    return (totalRight, totalWrong, accMap)

def summarizeIntervalComparison(trueStats, predStats, weighted, ignore):
    """ like above but done on two 1-sided interval comparisions.  only
    retunrs a map (ie no totalright total wrong) """
    accMap = dict()
    stateSet = set(predStats.keys()).union(set(trueStats.keys())) - ignore

    totalTrueTp = 0
    totalTrueFp = 0
    totalPredTp = 0
    totalPredFp = 0
    
    for state in stateSet:
        recall = 0.0
        if state in trueStats:
            tp = trueStats[state][0]
            fp = trueStats[state][2]
            if weighted is True:
                tp *= trueStats[state][1]
                fp *= trueStats[state][3]
            # Warning: can skip total fail in overall with this step
            # (but we need to do it to filter out stuff were not trying to
            # predict
            if state in trueStats and state in predStats:
                totalTrueTp += tp
                totalTrueFp += fp
            if tp + fp > 0:
                recall = float(tp) / float(tp + fp)

        precision = 0.0
        if state in predStats:
            tp = predStats[state][0]
            fp = predStats[state][2]
            if weighted is True:
                tp *= predStats[state][1]
                fp *= predStats[state][3]
            # Warning: can skip total fail in overall with this step
            # (but we need to do it to filter out stuff were not trying to
            # predict
            if state in trueStats and state in predStats:
                totalPredTp += tp
                totalPredFp += fp
            if tp + fp > 0:
                precision = float(tp) / float(tp + fp)

        accMap[state] = (precision, recall)

    totalRecall = 0.
    if totalTrueTp + totalTrueFp > 0:
        totalRecall = float(totalTrueTp) / float(totalTrueTp + totalTrueFp)
    totalPrecision = 0.
    if totalPredTp + totalPredFp > 0:
        totalPrecision = float(totalPredTp) / float(totalPredTp + totalPredFp)

    assert "Overall" not in accMap
    accMap["Overall"] = (totalPrecision, totalRecall)
    
    return accMap
        

def summaryRow(accuracy, stats, accMap):
    header = []
    row = []
    header.append("totAcc")
    row.append(accuracy)
    for state in sorted(accMap.keys()):
        acc = accMap[state]
        # precision
        header.append("%s_Prec" % state)
        row.append(acc[0])
        # recall
        header.append("%s_Rec" % state)
        row.append(acc[1])
        # fscore
        header.append("%s_F1" % state) 
        fscore = 0
        if (acc[0] > 0 and acc[1] > 0):
            fscore = 2 * ((acc[0] * acc[1]) / (acc[1] + acc[0]))
        row.append(fscore)
    row = map(str, row)
    assert len(header) == len(row)
    return header, row

def writeAccPlots(accuracy, baseAccMap, intAccMap, intAccMapWeighted,
                  threshold, outFile):
    """ plot accuracies as scatter plots"""

    accMaps = [baseAccMap, intAccMap, intAccMapWeighted]
    names = ["Base Acc.", "Int. Acc. thr=%.2f" % threshold,
             "Wgt Int. Acc. thr=%.2f" % threshold]

    stateNames = set()
    for am in accMaps:
        stateNames = stateNames.union(set(am.keys()))
    emptyStates = set()
    for state in stateNames:
        total = 0.
        for am in accMaps:
            if state in am:
                total += am[state][0]
                total += am[state][1]
        if total == 0.:
            emptyStates.add(state)
        
    stateNames = list(stateNames - emptyStates)

    distList = []
    for i in xrange(len(accMaps)):
        distList.append([(0,0)] * len(stateNames))
    titles = []

    for i, accMap in enumerate(accMaps):
        totalF = 0.0
        numF = 0.0
        for state in sorted(accMap.keys()):
            acc = accMap[state]
            prec = acc[0]
            rec = acc[1]

            if prec > 0.0 or rec > 0.0:
                stateIdx = stateNames.index(state)
                fs = 2. * ((prec * rec) / (prec + rec))
                totalF += fs
                numF += 1.
                distList[i][stateIdx] = (prec, rec)
            
        avgF = 0.
        if totalF > 0:
            avgF = totalF / numF
        titles.append("%s (avg f1=%.3f)" % (names[i], avgF))

    plotPoints2d(distList, titles, stateNames, outFile, xRange=(-0.1,1.1),
                 yRange=(-0.1, 1.4), ptSize=75, xLabel="Precision",
                 yLabel="Recall", cols=2, width=10, rowHeight=5)

def updateConfMatrix(matrix, predState, trueState):
    """ update a confusion matrix which is just represented as a 2-d dictionary
    of state names as strings.  matrix[predState][trueState] stores a count of
    how many times a prediction overlaps a true state.  Taking the maximum
    value across all trueStates here can be used to assign state names to
    unsupervised predictions
     """
    if predState not in matrix:
        matrix[predState] = dict()
    pred = matrix[predState]
    if trueState not in pred:
        pred[trueState] = 0
    pred[trueState] += 1
    return matrix

def getStateMapFromConfMatrix_simple(forwardMatrix):
    """ return a dictionary mapping predicted state names to true state names
    using the confusion matrix that was generated using the above function...
    in addition to the name, the number of overlaps and total overlaps are
    returned to give an indication of the fit...
    """
    stateMap = dict()
    for predName, predDict in forwardMatrix.items():
        maxName, maxCount = None, -1
        total = 0
        for trueName, count in predDict.items():
            total += count
            if count > maxCount:
                maxName, maxCount = trueName, count
        stateMap[predName] = maxName, maxCount, total
    return stateMap

def getStateMapFromConfMatrix(reverseMatrix, truthTgt, truthIgnore, predIgnore, thresh,
                              fdr):
    """ Use greedy algorithm to construct state map in order to maximize F1 score of
    each non-ignored state (in order of size in truth).

    The greedy heuristic here (mapping to truth states in order of their genome
    coverage) is worrisome.  Since once a predicted state is mapped to the truth
    state it is left out of consideration for all other truth states.  One hopes
    the F1 metric compensates for this somewhat, and observes that the "truth"
    annotations we currently consider (TE/non-TE) should *still be optimal* despite
    the heuristic. 

    NOTE: Unlike old version above, the input matrix is the reverse confusion matrix,
    ie mapping truth states back to predictions (tho matrix data is symmetrical,
    representation used in this module is not, and more convenient to work in one
    direction or another)

    UPDATE: FDR option allows to skip F1 optimization and just use fdr directly
    as a cutoff for candidates
    """

    # build maps of state names to # bases in resepctive annotations
    truthStateSizes = defaultdict(int)
    predStateSizes = defaultdict(int)
    for truthState in reverseMatrix.keys():
        for predState, overlap in reverseMatrix[truthState].items():
            truthStateSizes[truthState] += overlap
            predStateSizes[predState] += overlap
            
    # sort truth states decreasing order
    truthStateList = truthStateSizes.items()
    truthStateList.sort(key = lambda x : x[1], reverse = True)
    logger.debug("State ranking in f1Fit:" + str(truthStateList))
    
    # main loop
    stateNameMap = dict()
    for truthState, truthSize in truthStateList:
        if truthState in truthIgnore or \
               (len(truthTgt) > 0 and truthState not in truthTgt):
            continue
        # assemble list of candidate pred states that meet threshold
        predCandidates = []
        # assemble list of andidate pred states that exceed 1-threshold
        # these will be sure bets that we assume are good
        sureBets = []
        # tack in extra list for FDR option that overrides other two
        # if FDR activated
        fdrSureBets = []
        for predState, overlap in reverseMatrix[truthState].items():
            if predState not in stateNameMap and\
              predState not in predIgnore:
                predFrac = float(overlap) / float(min(truthSize,
                                                      predStateSizes[predState]))
                if predFrac >= thresh:
                    if predFrac >= 1. - thresh:
                        sureBets.append(predState)
                    else:
                        predCandidates.append(predState)
                if fdr is not None:
                    # above calculation of predFrac is effective heuristic but
                    # runs against definition of fdr
                    predFrac = float(overlap) / float(predStateSizes[predState])
                    if predFrac >= 1. - fdr:
                        fdrSureBets.append(predState)
            else:
                logger.debug("state mapper skipping %s with othresh %f" % (
                    predState, float(overlap) / float(min(truthSize,
                                                          predStateSizes[predState]))))
        if fdr is not None:
            sureBets = fdrSureBets
            predCandidates = []
        logger.debug("candidates for %s: %s" % (truthState, str(predCandidates)))
        logger.debug("sure bets for %s: %s" % (truthState, str(sureBets)))

        # iterate over all combinaations of candidate mappings
        def allSubsets(s):
            if len(sureBets) > 0:
                yield []
            for i in xrange(1, len(s) + 1):
                for j in itertools.combinations(s, i):
                    yield j
        bestF1, bestMapSet = -1., []
        for candidateSetIter in allSubsets(predCandidates):
            candidateSet = list(candidateSetIter) + sureBets
            # compute the f1 score of this mapping
            p, r, f1, tp, fp, fn = 0.,0.,0.,0.,0., float(truthStateSizes[truthState])
            bsSortMap = dict()
            for predState in candidateSet:
                overlap = reverseMatrix[truthState][predState]
                tp += overlap
                fp += predStateSizes[predState] - overlap
                bsSortMap[predState] = tp + fp
                fn -= overlap
            if tp > 0.:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = (2. * p * r) / (p + r)
            #print f1, p, r, tp, fp, fn, str(candidateSet)
            if f1 > bestF1:
                # sort by total number of bases 
                bestF1, bestMapSet = f1, sorted(candidateSet, reverse=True,
                                                key = lambda x : bsSortMap[x])
                
        # add best candidate set to prediction state name map
        for predState in bestMapSet:
            assert predState not in stateNameMap
            stateNameMap[predState] = [truthState,
                                       reverseMatrix[truthState][predState],
                                       predStateSizes[predState]]
        logger.debug("map %s <---- %s" % (truthState, str(bestMapSet)))
        logger.debug("best F1 = %s" % bestF1)
    
    # predStateName - > (truthStateName, tp, tp+fp)        
    return stateNameMap
    
            
def extractCompStatsFromFile(dumpPath):
    """ I've developed a habit of dumping the output of this program to
    variuous text files (ie in teHmmBenchmark).  This function can recover
    the three maps (base, interval, weighted) from such a file,
    for use in another script (as regenerating comparison can be slow)"""
    dumpFile = open(dumpPath, "r")
    baseStats, intervalStats, weightedStats = None, None, None
    mode = None
    for line in dumpFile:
        if line.find("Base-by-base Accuracy") == 0:
            mode = "base"
        elif line.find("Interval Accuracy") == 0:
            mode = "interval"
        elif line.find("Weighted Interval Accuracy") == 0:
            mode = "weighted"
        elif mode is not None:
            stats = ast.literal_eval(line)
            if mode == "base":
                baseStats = stats
            elif mode == "interval":
                intervalStats = stats
            elif mode == "weighted":
                weightedStats = stats
            mode = None
        if baseStats is not None and intervalStats is not None and\
          weightedStats is not None:
          break
    dumpFile.close()
    return baseStats, intervalStats, weightedStats

def extractCompCountsFromFile(dumpPath):
    """ like above, but return the first dictionary with counts instead of
    precision recall. this is a little hack to play with computation of
    sensitivity / specificity for two-class cases. """
    dumpFile = open(dumpPath, "r")
    counts = None
    mode = None
    for line in dumpFile:
        if line.find("Base counts") == 0:
            mode = "counts"
        elif mode == "counts":
            counts = ast.literal_eval(line)
            break
    dumpFile.close()
    return counts

def extract2ClassSpecificityFromFile(dumpPath, state):
    """ Compute specificity for just one state name vs everything else """
    countTable = extractCompCountsFromFile(dumpPath)
    totalBases = 0
    # count the total number of bases
    for name, value in countTable.items():
        fn, fp, tp = value[0], value[1], value[2]
        totalBases += tp + fn
    if state not in countTable:
        fn, fp, tp = 0, 0, 0
    else:
        fn, fp, tp = countTable[state]
    # are true negatives are everything but false pos/neg and true pos for the
    # state in question
    tn = totalBases - (fp + tp + fn)
    if fp + tn == 0:
        spec = 0.
    else:
        spec = float(tn) / (float(fp) + float(tn))
    return spec

def cutOutMaskIntervals(inBed, minLength, maxLength, tracksInfoPath):
    """ Filter out intervals of mask tracks from inBed with lengths
    outside given range. Idea is that it makes less sense to simply ignore,
    say, giant stretches of N's (like centromeres), as we would by masking
    them normally, than it does to remove them entirely, splitting the
    genome into multiple chunks.  Can also be used during comparision to
    get rid of all masked intervals """
    outPath = getLocalTempPath("Tempcut", ".bed")
    trackList = TrackList(tracksInfoPath)
    maskPaths = [t.getPath() for t in trackList.getMaskTracks()]
    if len(maskPaths) == 0:
        return None
    tempPath1 = getLocalTempPath("Tempcut1", ".bed")
    tempPath2 = getLocalTempPath("Tempcut2", ".bed")
    runShellCommand("cp %s %s" % (inBed, outPath))
    for maskPath in maskPaths:
        runShellCommand("cat %s | awk \'{print $1\"\t\"$2\"\t\"$3}\' >> %s" % (
            maskPath, tempPath1))
    if os.path.getsize(tempPath1) > 0:
        runShellCommand("sortBed -i %s > %s ; mergeBed -i %s > %s" % (
            tempPath1, tempPath2, tempPath2, tempPath1))
        runShellCommand("filterBedLengths.py %s %d %d > %s" % (
            tempPath1, minLength+1, maxLength-1, tempPath2))
        runShellCommand("subtractBed -a %s -b %s | sortBed > %s" % (
            outPath, tempPath2, tempPath1))
        runShellCommand("mv %s %s" % (tempPath1, outPath))
    runShellCommand("rm -f %s %s" % (tempPath1, tempPath2))
    if os.path.getsize(outPath) == 0:
        raise RuntimeError("cutOutMaskIntervals removed everything.  Can't continue."
                           " probably best to rerun calling script on bigger region?")
    return outPath

def checkExactOverlap(bed1, bed2):
    """ make sure two bed files cover same region exactly: a requirement for all
    code based on the comparisons in this module."""

    errorMessage = ("Bed files %s and %s cannot be compared. xxx. "
    " Input files must be both sorted, cover the exact same region,"
    " and contain no self-overlaps.") % (bed1, bed2)

    # empty file may break downstream comparisons
    size1 = os.path.getsize(bed1)
    size2 = os.path.getsize(bed2)
    if size1 == 0 or size2 == 0:
        raise RuntimeError(errorMessage.replace("xxx", "one or both inputs empty"))
                            

    # test self-overlap and sorting
    intervals1 = readBedIntervals(bed1, sort=False)
    for i in xrange(1, len(intervals1)):
        if intersectSize(intervals1[i-1], intervals1[i]) != 0:
            raise RuntimeError(errorMessage.replace(
                "xxx", "Overlapping intervals %s and %s found in input1" % (
                    intervals1[i-1], intervals1[i])))
        if intervals1[i-1] > intervals1[i]:
            raise RuntimeError(errorMessage.replace(
                "xxx", "Out of order intervals %s and %s found in input1" % (
                    intervals1[i-1], intervals1[i])))

    # test self-overlap and sorting
    intervals2 = readBedIntervals(bed1, sort=False)
    for i in xrange(1, len(intervals2)):
        if intersectSize(intervals2[i-1], intervals2[i]) != 0:
            raise RuntimeError(errorMessage.replace(
                "xxx", "Overlapping intervals %s and %s found in input2" % (
                    intervals2[i-1], intervals2[i])))
        if intervals2[i-1] > intervals2[i]:
            raise RuntimeError(errorMessage.replace(
                "xxx", "Out of order intervals %s and %s found in input2" % (
                    intervals2[i-1], intervals2[i])))
        

    # test intersection size
    tempFile = getLocalTempPath("Temp_test", ".bed")
    runShellCommand("subtractBed -a %s -b %s > %s" % (bed1, bed2, tempFile))
    if os.path.getsize(tempFile) != 0:
        runShellCommand("rm -f %s" % tempFile)
        raise RuntimeError(errorMessage.replace(
            "xxx", "Input1 covers regions outside input2"))
    runShellCommand("subtractBed -a %s -b %s > %s" % (bed2, bed1, tempFile))
    if os.path.getsize(tempFile) != 0:
        runShellCommand("rm -f %s" % tempFile)
        raise RuntimeError(errorMessage.replace(
            "xxx", "Input2 covers regions outside input1"))
    runShellCommand("rm -f %s" % tempFile)

    
def runPositionalComparison(argv, args):
    """ hack to recursively exectute compareBedStates.py on a sliding window of the two
    inputs and report accuracy in a BED file """
    try:
        windowToks = args.window.split(",")
        assert len(windowToks) == 5
        windowSize = int(windowToks[0])
        stateName = windowToks[1]
        compType = windowToks[2]
        score = windowToks[3]
        outBed = windowToks[4]
    except:
        raise RuntimeError("value passed to --window is not in valid format")
    if compType == "base":
        compIdx = 0
    elif compType == "interval":
        compIdx = 1
    elif compType == "weighted":
        compIdx = 2
    else:
        raise RuntimeError("invalid compType, %s, passed to --window" % compType)
    if score != "f1" and score != "precision" and score != "recall":
        raise RuntimeError("invalid score, %s, passed to --window" % score)
    try:
        outFile = open(outBed, "w")
    except:
        raise RuntimeError("invalid outBed, %s, passed to --window" % outBed)

    tempBed = getLocalTempPath("Temp_region", ".bed")
    runShellCommand("mergeBed -i %s > %s" % (args.bed1, tempBed))
    chunkBed = getLocalTempPath("Temp_chunkBed", ".bed")
    runShellCommand("chunkBedRegions.py %s %d --overlap .5 > %s" % (
        tempBed, windowSize, chunkBed))
    window = getLocalTempPath("Temp_window", ".bed")
    slice1 = getLocalTempPath("Temp_slice1", ".bed")
    slice2 = getLocalTempPath("Temp_slice2", ".bed")
    compFile = getLocalTempPath("Temp_compFile", ".bed")
    compOpts = ""
    winIdx = argv.index("--window")
    assert winIdx > 0 and winIdx < len(argv) -1 and argv[winIdx + 1] == args.window
    for i in xrange(3, len(argv)):
        if i != winIdx and i != winIdx + 1:
            compOpts += " " + argv[i]
    
    for chunk in readBedIntervals(chunkBed):
        runShellCommand("echo \"%s\t%d\t%d\" > %s" % (chunk[0], chunk[1], chunk[2],
                                                   window))
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (
            args.bed1, window, slice1))
        runShellCommand("intersectBed -a %s -b %s | sortBed > %s" % (
            args.bed2, window, slice2))
        runShellCommand("compareBedStates.py %s %s %s > %s" % (
            slice1, slice2, compOpts, compFile))
        stats = extractCompStatsFromFile(compFile)[compIdx]
        if stateName not in stats:
            stats[stateName] = (0,0)
        f1 = 0.
        prec, rec = stats[stateName]
        if prec + rec > 0:
            f1 = (2. * prec * rec) / (prec + rec)
        val = f1
        if score == "precision":
            val = prec
        elif score == "recall":
            val = rec
        outFile.write("%s\t%d\t%d\t%f\n" % (chunk[0], chunk[1], chunk[2], val))

    runShellCommand("rm -f %s %s %s %s %s %s" % (tempBed, chunkBed, window,
                                                 slice1, slice2, compFile))
    outFile.close()
                                                
                    
if __name__ == "__main__":
    sys.exit(main())
