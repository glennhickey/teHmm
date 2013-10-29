#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
from pybedtools import BedTool, Interval

"""meta data for a track that may get saved as part of a trained model"""
class Track(object):
    def __init__(self, name = None, number = None, valMap = None):
        #: Name of track
        self.name = name
        #: Unique integer id, also will be track's row in data array
        self.number = number
        #: Optional mapping class (see below) to convert data values into
        #: numeric format
        self.valMap = valMap
        
""" map a value to an integer category """
class TrackCategoryMap(object):
    def __init__(self):
        self.catMap = dict()
        
    def update(self, val):
        if val not in self.catMap:
            newVal = len(self.catMap) + 1
            assert newVal not in self.catMap
            self.catMap[val] = newVal
        
    def has(self, val):
        return val in self.catMap
        
    def getMap(self, val):
        return self.catMap[val]

""" map a range of values to an integer category """
class TrackRangeMap(object):
    def __init__self(self, intervalSize):
        assert intervalSize > 0
        self.intervalSize = intervalSize
        self.rangeSet = sortedset()

    def update(self, val):
        pass

    def has(self, val):
        return True

    def getMap(self, val):
        return math.floor(val / self.intervalSize) + 1
        

""" Vector of annotations for a genomic region.  Array and row index
are passed as input.
"""
class TrackData(object):
    def __init__(self, seqName, start, end, data, track):
        #: Name of Sequence (or Chromosome)
        self.seqName = seqName
        #: Start Position in sequence
        self.start = start
        assert end > start
        self.end = end

        self.rows = data.shape[1]
        self.cols = data.shape[0]
        assert track.number < self.rows
        assert self.end - self.start == self.cols 
        self.data = data
        self.track = track

    def getLength(self):
        return self.cols
    
    def setRange(self, start, end, val, updateMap=False):
        """ set an array entry, mapping to a numeric value using
        the valMap dictionary if present """
        assert start >= self.start
        assert end <= self.start + self.getLength()
        mappedVal = val
        if self.track.valMap is not None:
            if updateMap is True and self.track.valMap.has(val) is False:
                self.track.valMap.update(val)
            if self.track.valMap.has(val) is False:
                mappedVal = 0
            else:
                mappedVal = self.track.valMap.getMap(val)
        #todo: change to numpy iterator nditer
        for i in xrange(start, end):
            self.data[i - self.start][self.track.number] = mappedVal

    def getVal(self, pos):
        assert pos >= self.start
        assert pos < self.start + self.getLength()
        return self.data[pos - self.start][self.track.number]

""" Implementation of reading track from a Bed File.  Currently based
on pybedtools """
class BedTrackData(TrackData):
    def __init__(self, seqName, start, end, data, track):
        super(BedTrackData, self).__init__(seqName, start, end, data, track)

    def loadBedInterval(self, bedPath, useScore=False, updateMap=False):
        if not os.path.isfile(bedPath):
            raise RuntimeError("BED file not found %s" % bedPath)
        bedTool = BedTool(bedPath)
        interval = Interval(self.seqName, self.start, self.end)
        # todo: check how efficient this is
        for overlap in bedTool.all_hits(interval):
            start = max(self.start, overlap.start)
            end = min(self.end, overlap.end)
            assert start < end
            if useScore is True:
                val = overlap.score
            else:
                val = overlap.name                
            self.setRange(start, end, val, updateMap)
            
