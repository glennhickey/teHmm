#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import logging
from pybedtools import BedTool, Interval

""" all track-data specific io code goes here.  Just BED implemented for now,
will eventually add WIG and maybe eventually bigbed / bigwig """

###########################################################################

def readTrackData(trackPath, chrom, start, end, **kwargs):
    """ read genome annotation track into python list of values.  a value
    is returned for every element in range (default value is None).  The
    type of file is detected from the extension"""
    data = None
    if not os.path.isfile(trackPath):
        sys.stderr.write("Warning: track file not found %s\n" %
                         trackPath)
        return None

    trackExt = os.path.splitext(trackPath)[1]
    if trackExt == ".bed":
        return readBedData(trackPath, chrom, start, end, **kwargs)
    else:
        sys.stderr.write("Warning: non-BED file skipped %s\n" %
                         trackPath)
    return None

###########################################################################

def readBedData(bedPath, chrom, start, end, **kwargs):

    valCol = None
    sort = False
    if kwargs is not None and "valCol" in kwargs:
        valCol = int(kwargs["valCol"])
    valMap = None
    if kwargs is not None and "valMap" in kwargs:
        valMap = kwargs["valMap"]
    defVal = None
    if valMap is not None:
        defVal = valMap.getMissingVal()
    updateMap = False
    if kwargs is not None and "updateValMap" in kwargs:
        updateMap = kwargs["updateValMap"]
    if kwargs is not None and "sort" in kwargs:
        sort = kwargs["sort"] == True

    data = [defVal] * (end - start)
    logging.debug("readBedData(%s)" % bedPath)
    bedTool = BedTool(bedPath)
    if sort is True:
        logging.debug("sortBed(%s)" % bedPath)
        bedTool = bedTool.sort()
        
    interval = Interval(chrom, start, end)

    # todo: check how efficient this is
    logging.debug("intersecting (%s,%d,%d) and %s" % (
        chrom, start, end, bedPath))
    intersections = bedTool.all_hits(interval)
    logging.debug("loading data from intersections")
    for overlap in intersections:
        oStart = max(start, overlap.start)
        oEnd = min(end, overlap.end)
        val = overlap.name
        if valCol is not None:
            if valCol == 0:
                val = 1
            elif valCol == 4:
                val = overlap.score
            else:
                assert valCol == 3
        if valMap is not None:
            val = valMap.getMap(val, update=updateMap)
            
        for i in xrange(oEnd - oStart):
            data[i + oStart - start] = val

    logging.debug("done readBedData(%s)" % bedPath)
    return data

###########################################################################

def readBedIntervals(bedPath, ncol = 3, 
                     chrom = None, start = None, end = None,
                     sort = False):
    """ Read bed intervals from a bed file (or a specifeid range therein).
    NOTE: intervals are sorted by their coordinates"""
    
    if not os.path.isfile(bedPath):
        raise RuntimeError("Bed interval file %s not found" % bedPath)
    assert ncol == 3 or ncol == 4
    outIntervals = []
    logging.debug("readBedIntervals(%s)" % bedPath)
    bedTool = BedTool(bedPath)
    if sort is True:
        bedTool = bedTool.sort()
        logging.debug("sortBed(%s)" % bedPath)
    if chrom is None:
        bedIntervals = bedTool
    else:
        assert start is not None and end is not None
        interval = Interval(chrom, start, end)
        logging.debug("intersecting (%s,%d,%d) and %s" % (chrom, start, end,
                                                          bedPath))
        bedIntervals = bedTool.all_hits(interval)

    logging.debug("appending bed intervals")
    for feat in bedIntervals:
        outInterval = (feat.chrom, feat.start, feat.end)
        if ncol >= 4:
            outInterval += (feat.name,)
        if ncol >= 5:
            outInterval += (feat.score,)
        outIntervals.append(outInterval)
    logging.debug("finished readBedIntervals(%s)" % bedPath)
        
    return outIntervals

###########################################################################

def getMergedBedIntervals(bedPath, ncol=3):
    """ Merge all contiguous and overlapping intervals""" 

    if not os.path.isfile(bedPath):
        raise RuntimeError("Bed interval file %s not found" % bedPath)
    logging.debug("mergeBedIntervals(%s)" % bedPath)
    outIntervals = []
    bedTool = BedTool(bedPath)
    for feat in bedTool.merge():
        outInterval = (feat.chrom, feat.start, feat.end)
        if ncol >= 4:
            outInterval += (feat.name,)
        if ncol >= 5:
            outInterval += (feat.score,)
        outIntervals.append(outInterval)
    logging.debug("finished mergeBedIntervals(%s)" % bedPath)

    return outIntervals

