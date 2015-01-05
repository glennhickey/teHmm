#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import argparse
import logging
import numpy as np
import copy
from scipy.stats import mode
from collections import defaultdict

from numpy.testing import assert_array_equal

from teHmm.trackIO import readBedIntervals
from teHmm.modelIO import loadModel
from teHmm.common import intersectSize, initBedTool, cleanBedTool
from teHmm.common import addLoggingOptions, setLoggingFromOptions, logger


totalTok = "#\To\tal#"

def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Make some tables of statistics from a BED file.  All"
        " output will be written in one big CSV table to be viewed in a "
        "spreadsheet.")

    parser.add_argument("inBed", help="Input bed file")
    parser.add_argument("outCsv", help="Path to write output in CSV format")
    parser.add_argument("--ignore", help="Comma-separated list of names"
                        " to ignore", default="")
    parser.add_argument("--numBins", help="Number of (linear) bins for "
                        "histograms", type=int, default=10)
    parser.add_argument("--logHist", help="Apply log-transform to data for "
                        "histogram", action="store_true", default=False)
    parser.add_argument("--histRange", help="Histogram range as comma-"
                        "separated pair of numbers", default=None)
    parser.add_argument("--noHist", help="skip hisograms", action="store_true",
                        default=False)
    parser.add_argument("--noScore", help="Just do length stats",
                        action="store_true", default=False)
    parser.add_argument("--noLen", help="Just do score stats",
                        action="store_true", default=False)

    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    if args.histRange is not None:
        args.histRange = args.histRange.split(",")
        assert len(args.histRange) == 2
        args.histRange = int(args.histRange[0]), int(args.histRange[1])

    outFile = open(args.outCsv, "w")
    args.ignoreSet = set(args.ignore.split(","))

    intervals = readBedIntervals(args.inBed, ncol = 5)

    csvStats = ""
    # length stats
    if args.noLength is False:
        csvStats = makeCSV(intervals, args, lambda x : int(x[2])-int(x[1]),
                           "Length")
    # score stats
    try:
        if args.noScore is False:
            csvStats += "\n" + makeCSV(intervals, args, lambda x : float(x[4]),
                                       "Score")
            csvStats += "\n" + makeCSV(intervals, args, lambda x : float(x[4]) * (
                float(x[2]) - float(x[1])), "Score*Length")
    except Exception as e:
        logger.warning("Couldn't make score stats because %s" % str(e))
    outFile.write(csvStats)
    outFile.write("\n")
    outFile.close()
    cleanBedTool(tempBedToolPath)

def makeCSV(intervals, args, dataFn, sectionName):
    """ Make string in CSV format with summary and histogram stats for
    intervals """

    dataDict = bedIntervalsToDataDict(intervals, dataFn, args)
    csv = ""
    if len(dataDict) == 0:
        return csv

    # summary
    csv += "%s SummaryStats,\n" % sectionName
    summaryData = summaryStats(dataDict)
    assert len(summaryData) == len(dataDict)
    csv += "ID,Min,Max,Mean,Mode,Median,Count,Sum\n"
    for name, data in summaryData.items():
        assert len(data) == 7
        if name != totalTok:
            csv += name + "," + ",".join([str(x) for x in data]) + "\n"
    data = summaryData[totalTok]
    csv += "Total" + "," + ",".join([str(x) for x in data]) + "\n"
        
    # histogram
    start = min(0, summaryData[totalTok][0])
    end = max(start, summaryData[totalTok][1]) + 1
    if args.histRange is not None:
        start, end = args.histRange[0], args.histRange[1]
    histData = histogramStats(dataDict, summaryData, args,
                              start=start, end=end, density=True)
    headerBins = histData[totalTok][1]
    histTable = [["ID"] + [str(x) for x in headerBins]]
    for name, data in histData.items():
        freq, bins = data[0], data[1]
        if name != totalTok:
            if not args.logHist:
                assert_array_equal(bins, headerBins)
            histTable.append([name] + [str(x) for x in freq])
    data = histData[totalTok]
    freq, bins = data[0], data[1]
    histTable.append(["Total"] + [str(x) for x in freq])

    # transpose histTable into csv (mostly because Numbers is so increadibly
    # bad that you can't make a chart out of row-data in any way)
    rows = len(histTable)
    cols = len(histTable[0])
    if args.noHist is False:
        csv += "\n%s HistogramStats,\n" % sectionName
        for col in xrange(cols):
            csv += ",".join([histTable[x][col] for x in xrange(rows)]) + "\n"
    return csv
    
    
def bedIntervalsToDataDict(intervals, dataFn, args, nameCol=3, 
                           dtype=np.float, totalTok = totalTok):
    """ return dictionary mapping name to list of data points in numpy array """
    # pass 1: count
    counts = defaultdict(int)
    for interval in intervals:
        if interval[nameCol] in args.ignoreSet:
            continue
        counts[interval[nameCol]] += 1
        counts[totalTok] = counts[totalTok] + 1

    # create arrays
    data = dict()
    for name, count in counts.items():
        data[name] = np.zeros((count), dtype)

    # slowly write arrays
    cur = defaultdict(int)
    for interval in intervals:
        name = interval[nameCol]
        if name in args.ignoreSet:
            continue
        assert cur[name] < counts[name]
        data[name][cur[name]] = dataFn(interval)
        cur[name] += 1
        data[totalTok][cur[totalTok]] = dataFn(interval)
        cur[totalTok] += 1
    assert cur == counts
    
    return data

def summaryStats(dataDict):
    """ read some quick summary stats from data dict returned by
    bedIntervalsToDataDict() """
    summaryStats = dict()
    for name, data in dataDict.items():
        summaryStats[name] = (np.min(data), np.max(data), np.mean(data),
                              mode(data)[0][0], np.median(data), len(data),
                              np.sum(data))
    return summaryStats

def histogramStats(dataDict, summaryDict, args, start=None, end=None,
                   density=False):
    """ create a histogram for each category """
    histStats = dict()
    assert len(dataDict) == len(summaryDict)
    for name, data in dataDict.items():
        summary = dataDict[name]
        hStart = start
        if hStart is None:
            hStart = summary[0]
        hEnd = end
        if hEnd is None:
            hEnd = summary[1]
        hEnd = max(hEnd, hStart + 1)
        hData = data
        if args.logHist is True:
            hFn = lambda x : np.log(x + summary[0] + 1)
            hData, hStart, hEnd = hFn(data), hFn(hStart), hFn(hEnd)
        freq, bins = np.histogram(hData, bins=args.numBins, range=(hStart, hEnd),
                                  density=density)
        histStats[name] = (freq, bins[:-1])
    
    return histStats

if __name__ == "__main__":
    sys.exit(main())
