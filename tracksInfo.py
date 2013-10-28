#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys

""" Read some tracks info.  For now just a map between track name
and track path.  
"""
def TracksInfo(object):
    def __init__(self, inputPath = None):
        #: Store list of paths as mapped to track names
        #: (spaces not supported for now)
        self.pathMap = dict()
        if inputPath is not None:
            self.loadFile(inputPath)

    def loadFile(self, inputPath):
        """Load in a text file with lines of formant TrackName TrackPath"""
        self.pathMap = dict()
        inputFile open(inputPath, "r")
        lineNo = 0
        for line in inputFile:
            lineNo += 1
            strippedLine = line.strip()
            if len(strippedLine) > 0 and strippedLine[0] != '#':
                toks = strippedLine.split()
                if len(toks) != 2:
                    raise RuntimeError("Line %d of %s is invalid" % (lineNo,
                                                                     inputPath))
                elif toks[0] in self.pathMap:
                    raise RuntimeError("Duplicate ine %d of %s" % (lineNo,
                                                                   inputPath))
                elif not os.path.isfile(tosk[1]):
                    raise RuntimeError("Track file %s not found" % toks[1])
                
                self.pathMap[toks[0]] = toks[1]

    def getNumTracks(self):
        """Number of tracks for model"""
        return len(self.pathMap)

    def getPath(self, trackName):
        """Retrieve path of annoation track.  Return None if not found"""
        if trackName not in self.pathMap:
            return None
        else:
            return self.pathMap[trackName]
    
        
                 
