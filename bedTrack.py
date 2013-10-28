#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
from pybedtools import BedTool, Interval

from track import Track

""" Implementation of reading track from a Bed File.  Currently based
on pybedtools """
def BedTrack(Track):
    def __init__(self, seqName, startPos, length, dataType=int, valMap=None):
        super.__init__(self, seqName, startPos, length, dataType, valMap)

    def loadBedInterval(self, bedPath, useScore=False, updateMap=False):
        if not os.path.isfile(bedPath):
            raise RuntimeError("BED file not found %s" % bedPath)
        bedTool = BedTool(bedPath)
        interval = Interval(self.seqName, self.getRange()[0],
                            self.getRange()[1])
        # todo: check how efficient this is
        for overlap in bedTool.allHits(interval):
            start = max(self.start, overlap.start)
            end = min(self.end, overlap.end)
            assert start < end
            if useScore is True:
                val = overlap.score
            else:
                val = overlap.name                
            self.setRange(start, end, updateMap)
            
            
            
        

