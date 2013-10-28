#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np

""" map a value to an integer category """
def TrackCategoryMap(object):
    def __init__(self):
        self.catMap = dict()
        
    def update(val):
        if val not in self.catMap:
            newVal = len(self.catMap) + 1
            assert newVal not in self.catMap
            self.catMap[val] = newVal
        
    def has(val):
        return val in self.catMap
        
    def getMap(val):
        return self.catMap[val]

""" map a range of values to an integer category """
def TrackRangeMap(object):
    def __init__self(self, intervalSize):
        assert intervalSize > 0
        self.intervalSize = intervalSize
        self.rangeSet = sortedset()

    def update(val):
        pass

    def has(val):
        return True

    def getMap(val):
        return math.floor(val / self.intervalSize) + 1
        

""" Vector of annotations for a genomic region
"""
def Track(object):
    def __init__(self, seqName, start, end, dataType=int, valMap=None):
        #: Name of Sequence (or Chromosome)
        self.seqName = seqName
        #: Start Position in sequence
        self.start = start
        assert end > start
        #: Init empty array.  Note that 0 is same as no value here
        self.data = np.zeroes((end - start,), dtype=dataType)
        #: Map input values to numeric values if necessary
        self.valMap = valMap

    def getLength(self):
        return self.data.shape[0]

    def getRange(self):
        return (self.start, self.start + self.getLength())
    
    def setRange(self, start, end, val, updateMap=False):
        """ set an array entry, mapping to a numeric value using
        the valMap dictionary if present """
        assert start >= self.start
        assert end <= self.start + self.getLength()
        mappedVal = val
        if self.valMap is not None:
            if updateMap is True and self.valMap.has(val) is False:
                self.valMap.update(val)
            if self.valMap.has(val) is False:
                mappedVal = 0
            else:
                mappedVal = self.valMap.getMap(val)
        for i in xrange(start, end):
            self.data[i - self.start] = mappedVal

    def getVal(self, pos):
        assert pos >= self.start
        assert pos < self.start + self.getLength()
        return self.data[pos - self.start]
