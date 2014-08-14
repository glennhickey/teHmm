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
                        " to ignore", default=None)
    parser.add_argument("--numBins", help="Number of (linear) bins for "
                        "histograms", type=int, default=10)

    addLoggingOptions(parser)
    args = parser.parse_args()
    setLoggingFromOptions(args)
    tempBedToolPath = initBedTool()

    outFile = open(args.outCsv, "w")

    intervals = readBedIntervals(args.inBed, ncol = 4)

    # length stats
    csvStats = makeCSV(intervals, args, lambda x : int(x[2])-int(x[1]))
    # score stats
    try: 
        csvStats += "\n" + makeCSV(intervals, args, lambda x : float(x[4]))
    except Exception as e:
        logger.warning("Couldn't make score stats because %s" % str(e))
    outFile.write(csvStats)
    outFile.write("\n")
    outFile.close()
    cleanBedTool(tempBedToolPath)

def makeCSV(intervals, args, dataFn):
    """ Make string in CSV format with summary and histogram stats for
    intervals """

    dataDict = bedIntervalsToDataDict(intervals, dataFn)
    csv = ""
    if len(dataDict) == 0:
        return csv

    # summary
    csv += "SummaryStats,\n"
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
    csv += "\nHistogramStats,\n"
    start = min(0, summaryData[totalTok][0])
    end = max(start, summaryData[totalTok][1]) + 1
    histData = histogramStats(dataDict, summaryData, numBins=args.numBins,
                              start=start, end=end)
    headerBins = histData[totalTok][1]
    csv += "ID," + ",".join([str(x) for x in headerBins]) + "\n"
    for name, data in histData.items():
        freq, bins = data[0], data[1]
        if name != totalTok:
            assert_array_equal(bins, headerBins)
            csv += name + "," + ",".join([str(x) for x in freq]) + "\n"
    data = histData[totalTok]
    freq, bins = data[0], data[1]
    csv += "Total" + "," + ",".join([str(x) for x in freq]) + "\n"
    
    return csv
    
    
def bedIntervalsToDataDict(intervals, dataFn, nameCol=3, 
                           dtype=np.float, totalTok = totalTok):
    """ return dictionary mapping name to list of data points in numpy array """
    # pass 1: count
    counts = defaultdict(int)
    for interval in intervals:
        counts[interval[nameCol]] += 1
    counts[totalTok] = len(intervals)

    # create arrays
    data = dict()
    for name, count in counts.items():
        data[name] = np.zeros((count), dtype)

    # slowly write arrays
    cur = defaultdict(int)
    for interval in intervals:
        name = interval[nameCol]
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

def histogramStats(dataDict, summaryDict, numBins=10, start=None, end=None):
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
        freq, bins = np.histogram(data, bins=numBins, range=(hStart, hEnd))
        histStats[name] = (freq, bins)
    
    return histStats

if __name__ == "__main__":
    sys.exit(main())
