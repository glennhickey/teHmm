#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import string
import random
import logging
import array
from pybedtools import BedTool, Interval
from .common import runShellCommand

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
    tempPath = None
    if trackExt == ".bw" or trackExt == ".bigwig" or trackExt == ".wg":
        #just writing in current directory.  something more principaled might
        #be safer / nicer eventually
        # make a little id tag:
        S = string.ascii_uppercase + string.digits
        tag = ''.join(random.choice(S) for x in range(5))
  
        tempPath = os.path.splitext(os.path.basename(trackPath))[0] \
                   + "_temp%s.bed" % tag
        logging.info("Extracting wig to temp bed %s. Make sure to erase"
                     " in event of crash" % os.path.abspath(tempPath)) 
        runShellCommand("bigWigToBedGraph %s %s -chrom=%s -start=%d -end=%d" %
                        (trackPath, tempPath, chrom, start, end))
        trackExt = ".bed"
        trackPath = tempPath
        if (kwargs is None):
            kwargs = dict()
        kwargs["needIntersect"] = False
    if trackExt == ".bed":
        data = readBedData(trackPath, chrom, start, end, **kwargs)
    else:     
        sys.stderr.write("Warning: non-BED file skipped %s\n" %
                         trackPath)
        
    if tempPath is not None:
        runShellCommand("rm -f %s" % tempPath)
    return data

###########################################################################

def readBedData(bedPath, chrom, start, end, **kwargs):

    valCol = None
    sort = False
    needIntersect = True
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
    if kwargs is not None and "needIntersect" in kwargs:
        needIntersect = kwargs["needIntersect"]

    data = [defVal] * (end - start)
    logging.debug("readBedData(%s, update=%s)" % (bedPath, updateMap))
    bedTool = BedTool(bedPath)
    if sort is True:
        logging.debug("sortBed(%s)" % bedPath)
        bedTool = bedTool.sort()
        
    interval = Interval(chrom, start, end)

    # todo: check how efficient this is
    if needIntersect is True:
        logging.debug("intersecting (%s,%d,%d) and %s" % (
            chrom, start, end, bedPath))
        intersections = bedTool.all_hits(interval)
    else:
        intersections = bedTool
    logging.debug("loading data from intersections")
    basesRead = 0
    for overlap in intersections:
        oStart = max(start, overlap.start)
        oEnd = min(end, overlap.end)
        val = overlap.name
        if valCol is not None:
            if valCol == 0:
                val = 1
            elif valCol == 4:
                assert overlap.score is not None and overlap.score != ""
                val = overlap.score
            else:
                assert valCol == 3
                assert overlap.name is not None and overlap.name != ""
        if valMap is not None:
            ov  = val
            val = valMap.getMap(val, update=updateMap)
            
        for i in xrange(oEnd - oStart):
            data[i + oStart - start] = val
        basesRead += oEnd - oStart

    logging.debug("done readBedData(%s). %d bases read" % (bedPath, basesRead))
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

def getMergedBedIntervals(bedPath, ncol=3, sort = False):
    """ Merge all contiguous and overlapping intervals""" 

    if not os.path.isfile(bedPath):
        raise RuntimeError("Bed interval file %s not found" % bedPath)
    logging.debug("mergeBedIntervals(%s)" % bedPath)
    outIntervals = []
    bedTool = BedTool(bedPath)
    if sort is True:
        bedTool = bedTool.sort()
        logging.debug("sortBed(%s)" % bedPath)
    for feat in bedTool.merge():
        outInterval = (feat.chrom, feat.start, feat.end)
        if ncol >= 4:
            outInterval += (feat.name,)
        if ncol >= 5:
            outInterval += (feat.score,)
        outIntervals.append(outInterval)
    logging.debug("finished mergeBedIntervals(%s)" % bedPath)

    return outIntervals

###########################################################################

def writeBedIntervals(intervals, outPath):
    """ write bed intervals to disk """
    outFile = open(outPath, "w")
    for interval in intervals:
        bi = Interval(interval[0], interval[1], interval[2], 
                      interval[3])
        outFile.write(str(bi))    

###########################################################################

# Copied from bioio.py from sonLib (https://github.com/benedictpaten/sonLib):
# Copyright (C) 2006-2012 by Benedict Paten (benedictpaten@gmail.com)
# Released under the MIT license, see LICENSE.txt
def fastaRead(fileHandle):
    """iteratively a sequence for each '>' it encounters, ignores '#' lines
    """
    line = fileHandle.readline()
    while line != '':
        if line[0] == '>':
            name = line[1:-1]
            line = fileHandle.readline()
            seq = array.array('c')
            while line != '' and line[0] != '>':
                if line[0] != '#':
                    seq.extend([ i for i in line[:-1] if i != '\t' and i != ' ' ])
                line = fileHandle.readline()
            for i in seq:
                #For safety and sanity I only allows roman alphabet characters in fasta sequences. 
                if not ((i >= 'A' and i <= 'Z') or (i >= 'a' and i <= 'z') or i == '-'):
                    raise RuntimeError("Invalid FASTA character, ASCII code = \'%d\', found in input sequence %s" % (ord(i), name))
            yield name, seq.tostring()
        else:
            line = fileHandle.readline()
