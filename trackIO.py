#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
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

    data = [defVal] * (end - start)
    bedTool = BedTool(bedPath).sort()
    interval = Interval(chrom, start, end)

    # todo: check how efficient this is
    for overlap in bedTool.all_hits(interval):
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

    return data

###########################################################################

def readBedIntervals(bedPath, ncol = 3, 
                     chrom = None, start = None, end = None):
    """ Read bed intervals from a bed file (or a specifeid range therein).
    NOTE: intervals are sorted by their coordinates"""
    
    if not os.path.isfile(bedPath):
        raise RuntimeError("Bed interval file %s not found" % bedPath)
    assert ncol == 3 or ncol == 4
    outIntervals = []
    bedTool = BedTool(bedPath).sort()
    if chrom is None:
        bedIntervals = bedTool
    else:
        assert start is not None and end is not None
        interval = Interval(chrom, start, end)            
        bedIntervals = bedTool.all_hits(interval)

    for feat in bedIntervals:
        outInterval = (feat.chrom, feat.start, feat.end)
        if ncol == 4:
            outInterval += (feat.name,)
        outIntervals.append(outInterval)
        
    return outIntervals

