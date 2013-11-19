#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import xml.etree.ElementTree as ET

from .trackIO import readTrackData

INTEGER_ARRAY_TYPE = np.int16

###########################################################################

"""meta data for a track that may get saved as part of a trained model,
and can also be specified in the xml file"""
class Track(object):
    def __init__(self, xmlElement=None, number=-1):
        #: Name of track
        self.name = None
        #: Unique integer id, also will be track's row in data array
        self.number = number
        #: Optional mapping class (see below) to convert data values into
        #: numeric format
        self.valMap = None
        #: Path of the bedfile.
        self.path = None
        #: Distribution type (only multinomial for now)
        self.dist = "multinomial"
        #: Number of bins for mapping numeric values to discrete states
        self.binSize = 1
        #: Bed column to take value from (default 3==name)
        self.valCol = 3

        if xmlElement is not None:
            self._fromXMLElement(xmlElement)
        self._init()

    def _init(self):
        if self.dist == "multinomial":
            self.valMap = CategoryMap()
        elif self.dist == "binary":
            self.valMap = BinaryMap()
            self.binSize = None
            self.valCol = 0
        assert self.dist == "multinomial" or self.dist == "binary"

    def _fromXMLElement(self, elem, number=-1):
        self.name = elem.attrib["name"]
        self.path =  elem.attrib["path"]
        if "distribution" in elem.attrib:
            self.dist = elem.attrib["distribution"]
        if "binSize" in elem.attrib:
            self.binSize = int(elem.attrib["binSize"])
        if "valCol" in elem.attrib:
            self.valCol = int(elem.attrib["valCol"])       

    def getValueMap(self):
        return self.valMap

    def getNumber(self):
        return self.number

    def getName(self):
        return self.name

    def getDist(self):
        return self.dist

    def getPath(self):
        return self.path

    def getValCol(self):
        return self.valCol

    def getBinSize(self):
        return self.binSize


###########################################################################
"""list of tracks (see above) that we can index by name or number as well as
load from or save to a file. this strucuture needs to accompany a trained
model. """
class TrackList(object):
   def __init__(self, xmlPath = None):
       #: list of tracks.  track.number = its position in this list
       self.trackList = []
       #: map a track name to its position in the list
       self.trackMap = dict()
       if xmlPath is not None:
           self.loadXML(xmlPath)

   def getTrackByName(self, name):
       if name in self.trackMap:
           trackIdx = self.trackMap[name]
           assert self.trackList[trackIdx].number == trackIdx
           return self.trackList[trackIdx]
       return None

   def getTrackByNumber(self, idx):
       if idx < len(self.trackList):
           assert self.trackList[idx].number == idx
           return self.trackList[idx]
       return None

   def addTrack(self, track):
       track.number = len(self.trackList)
       self.trackList.append(track)
       assert track.name not in self.trackMap
       self.trackMap[track.name] = track.number
       self.__check()

   def load(self, path):
       f = open(path, "rb")
       tmp_dict = pickle.load(f)
       f.close()
       self.__dict__.update(tmp_dict)
       self.__check()

   def save(self, path):
       f = open(path, "wb")
       pickle.dump(self.__dict__, f, 2)
       f.close()

   def loadXML(self, path):
       """Load in an xml file that contains a list of track elements right
       below its root node.  Will extend to contain more options..."""
       root = ET.parse(path).getroot()
       for child in root.findall("track"):
           track = Track(child)
           self.addTrack(track)

   def __check(self):
       for i,track in enumerate(self.trackList):
           assert track.number == i
           assert track.name in self.trackMap
       assert len(self.trackMap) == len(self.trackList)

   def __len__(self):
       return len(self.trackList)

   def __iter__(self):
       for track in self.trackList:
           yield track

###########################################################################

"""array of data for interval on several tracks.  We use this interface
rather than just numpy arrays so that we can eventually (hopefully) add
mixed datatypes without having to change any of the calling code"""
class TrackTable(object):
    def __init__(self, numTracks, chrom, start, end):
        #: Number of rows
        self.numTracks = numTracks
        #: Chromosome name
        self.chrom = chrom
        #: Start coordinate
        self.start = start
        #: End coordinate (last coordinate plus 1)
        self.end = end
        assert end > start
        #: mimic numpy array
        self.shape = (len(self), self.getNumTracks())

    def __len__(self):
        """ Number of columns in the table """
        return self.end - self.start

    def getNumTracks(self):
        """ Number of rows in the table """
        return self.numTracks
    
    def __getitem__(self, index):
        """ Get a vector corresponding to a single column (ie vector of
        annotation values for the same genome coordinate """
        raise RuntimeError("Not implemented")

    def writeRow(self, row, rawArray):
        """ Write a row of data """
        raise RuntimeError("Not implemented")

    def getOverlap(self, bedInterval):
        """ Compute overlap with a bed coordinate. return None if do not
        intersect"""
        assert len(bedInterval) > 2
        overlap = None
        chrom, start, end = bedInterval[0], bedInterval[1], bedInterval[2]
        if self.chrom == chrom and self.start < end and self.end > start:
            overlap = self.chrom, max(self.start, start), min(self.end, end)
            for i in xrange(3, len(bedInterval)):
                overlap += (bedInterval[i],)
        return overlap

    def getChrom(self):
        return self.chrom
    
    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

###########################################################################

"""Track Table where every value is an integer"""
class IntegerTrackTable(TrackTable):
    def __init__(self, numTracks, chrom, start, end):
        super(IntegerTrackTable, self).__init__(numTracks, chrom, start, end)
        
        #: (end-start) X (numTracks) integer data array
        self.data = np.empty((end-start, numTracks), dtype=INTEGER_ARRAY_TYPE)

    def __getitem__(self, index):
        return self.data[index]

    def writeRow(self, row, rowArray):
        """ write exactly one full row of data values to the table, mapping
        each value using valueMap if it's specified """
        assert row < self.getNumTracks()
        assert len(rowArray) == len(self)
        for i in xrange(len(self)):
            self.data[i][row] = rowArray[i]
            
###########################################################################
            
""" map a value to an integer category """
class CategoryMap(object):
    def __init__(self):
        self.catMap = dict()
        self.catMapBack = dict()
        
    def update(self, val):
        if val not in self.catMap:
            newVal = len(self.catMap)
            assert newVal not in self.catMap
            self.catMap[val] = newVal
            self.catMapBack[newVal] = val
        
    def has(self, val):
        return val in self.catMap
        
    def getMap(self, val, update = False):
        if val is not None and update is True and self.has(val) is False:
            self.update(val)
        if self.has(val) is True:
            return self.catMap[val]
        return self.getMissingVal()

    def getMapBack(self, val):
        if val == self.unknown:
            return self.unknown
        return self.catMapBack[val]

    def getMissingVal(self):
        return len(self)

    def __len__(self):
        return len(self.catMap)

###########################################################################
    
""" Act like a cateogry map but dont do any mapping.  still useful for
keeping track of the number of distinct values """
class NoMap(CategoryMap):
    def __init__(self):
        super(NoMap, self).__init__()

    def getMap(self, val, update=False):
        super(NoMap, self).update(val, update)
        return val
    
    def getMapBack(self, val):
        return val    

###########################################################################
    
""" Act like a cateogry map but dont do any mapping.  By default everything
is 0, unless it's present then it's a 1"""
class BinaryMap(CategoryMap):
    def __init__(self):
        super(BinaryMap, self).__init__()

    def getMap(self, val, update=False):
        if val is not None:
            return 1
        return 0

    def getMapBack(self, val):
        return val

    def getMissingVal(self):
        return 0

    def __len__(self):
        return 2

###########################################################################
    
""" Data Array formed by a series of tracks over the same coordinates of the
same genomes.  Multiple intervals are supported. """
class TrackData(object):
    def __init__(self):
        #: list of tracks (of type TrackList)
        self.trackList = None
        #: list of track tables (of type TrackTable)
        self.trackTableList = None

    def getNumTracks(self):
        return len(self.trackList)

    def getTrackList(self):
        return self.trackList

    def getTrackTableList(self):
        return self.trackTableList

    def getNumTrackTables(self):
        return len(self.trackTableList)

    def getNumSymbolsPerTrack(self):
        nspt = [0] * self.getNumTracks()        
        for i in xrange(self.getNumTracks()):
            track = self.trackList.getTrackByNumber(i)
            nspt[i] = len(track.getValueMap())
        return nspt
    
    def loadTrackData(self, trackListPath, intervals, trackList = None):
        """ load track data for list of given intervals.  tracks is either
        a TrackList object loaded from a saved pickle, or None in
        which case they will be generated from the data.  each interval
        is a 3-tuple of chrom,start,end"""
        assert len(intervals) > 0
        inputTrackList = TrackList(trackListPath)
        if trackList is None:
            initTracks = True
            self.trackList = inputTrackList
            # note, need to make sure category maps get properly set
        else:
            initTracks = False
            self.trackList = trackList
        self.trackIdx = dict()

        self.trackTableList = []
        for interval in intervals:
            assert len(interval) >= 3 and interval[2] > interval[1]
            self.__loadTrackDataInterval(inputTrackList, interval[0],
                                         interval[1], interval[2], initTracks)

    def __loadTrackDataInterval(self, inputTrackList, chrom, start, end, init):
        trackTable = IntegerTrackTable(self.getNumTracks(), chrom, start, end)
        for inputTrack in inputTrackList:
            trackName = inputTrack.getName()
            trackPath = inputTrack.getPath()
            if self.trackList.getTrackByName(trackName) is None:
                sys.stderr.write("Warning: track %s not learned\n" %
                                 trackName)
                continue
            rawArray = readTrackData(trackPath, chrom, start, end,
                                     valCol=inputTrack.getValCol(),
                                     valMap=inputTrack.getValueMap(),
                                     updateValMap=init)
            if rawArray is not None:
                track = self.getTrackList().getTrackByName(trackName)
                trackTable.writeRow(track.getNumber(), rawArray)

        self.trackTableList.append(trackTable)

            
            
        
            
        
        

        
