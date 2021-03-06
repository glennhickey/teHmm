#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
from scipy.stats import mode
import xml.etree.ElementTree as ET
import xml.dom.minidom
from numpy.testing import assert_array_equal, assert_array_almost_equal
from _track import runSum
import itertools
import math

from .trackIO import readTrackData
from .common import EPSILON, logger, binSearch

INTEGER_ARRAY_TYPE = np.uint8

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
        #: Scale numeric values (use fraction to bin)
        self.scale = None
        #: Use specified value as logarithm base for scaling
        self.logScale = None
        #: Value to add (before scaling)
        self.shift = None
        #: Bed column to take value from (default 3==name)
        self.valCol = 3
        #: For fasta only
        self.caseSensitive = False
        #: Flag specifying that track values are represented as the difference
        # between the previous and current value (or 0 for start).  Best
        # used on numeric tracks.  Note that we only store the flag here and
        # it is up to the data reader to actually do the computations.  The
        # delta operation occurs *before* any binning or scaling. 
        self.delta = False
        #: Specify value given to unannotated bases
        self.defaultVal = None
        #: Specify preprocessor token
        self.preprocess = None

        if xmlElement is not None:
            self._fromXMLElement(xmlElement)
        self._init()

    def _init(self):
        if self.dist == "multinomial" or self.dist == "gaussian":
            reserved = 2
            if self.defaultVal is not None:
                reserved = 1
            elif self.dist == "gaussian":
                raise RuntimeError("\"default\" attribute must be specified for"
                                   " gaussian distribution (track %s)" %\
                                   self.name)
            self.valMap = CategoryMap(reserved=reserved,
                                      defaultVal=self.defaultVal,
                                      scale=self.scale, logScale=self.logScale,
                                      shift=self.shift)
        if self.dist == "sparse_multinomial":
            self.valMap = CategoryMap(reserved=1,
                                      scale=self.scale, logScale=self.logScale,
                                      shift=self.shift)
        elif self.dist == "binary" or self.dist == "mask":
            self.valMap = BinaryMap()
            self.valCol = 0
        elif self.dist == "alignment":
            self.valMap = CategoryMap(reserved=1)
        assert self.dist == "multinomial" or self.dist == "binary" \
               or self.dist == "alignment" or self.dist == "gaussian" \
               or self.dist == "mask"
        if self.logScale is not None:
            if self.scale is not None:
                logger.warning("logScale overriding scale for track %s" %(
                    self.name))
        if self.delta is True:
            if self.logScale is not None:
                raise RuntimeError("track %s: delta attribute not compatible"
                                   " with logScale" % self.getName())

    def _fromXMLElement(self, elem, number=-1):
        self.name = elem.attrib["name"]
        self.path =  elem.attrib["path"]
        if "distribution" in elem.attrib:
            self.dist = elem.attrib["distribution"]
            assert self.dist in ["binary", "multinomial", "sparse_multinomial",
                                 "alignment", "gaussian", "mask"]
        if "valCol" in elem.attrib:
            self.valCol = int(elem.attrib["valCol"])
        if "scale" in elem.attrib:
            self.scale = float(elem.attrib["scale"])
        if "logScale" in elem.attrib:
            self.logScale = float(elem.attrib["logScale"])
        if "shift" in elem.attrib:
            self.shift = float(elem.attrib["shift"])
        if "caseSensitive" in elem.attrib:
            cs = elem.attrib["caseSensitive"].lower()
            if cs == "1" or cs == "true":
                self.caseSensitive = True
            else:
                self.caseSensitive = False
        if "delta" in elem.attrib:
            d = elem.attrib["delta"].lower()
            if d == "1" or d == "true":
                self.delta = True
            else:
                self.delta = False
            if self.logScale is not None and self.delta is True:
                raise RuntimeError("track %s: delta attribute not compatible"
                                   " with logScale" % self.getName())
        if "default" in elem.attrib:
            self.defaultVal = elem.attrib["default"]
            df = float(self.defaultVal)
            if df <= 0. and self.logScale is not None:
                if self.shift is None or float(self.shift) + df <= 0:
                    raise RuntimeError("track %s: default set to %s in "
                                       "conjunction with logScale requires "
                                       "shift attribute set to at least %f" % (
                                           self.getName(), self.defaultVal,
                                           1. - df))
        if "preprocess" in elem.attrib:
            self.preprocess = elem.attrib["preprocess"]
            if self.preprocess not in ["rm", "rmu", "ltr_finder", "termini", "overlap"]:
                raise RuntimeError("track %s: preprocess set to invalid value %s.  must"
                                   " be rm, rmu, ltr_finder, termini or overlap" % (
                    self.getName(), self.preprocess))
            
    def toXMLElement(self):
        elem = ET.Element("track")
        if self.name is not None:
            elem.attrib["name"] = str(self.name)
        if self.path is not None:
            elem.attrib["path"] = str(self.path)
        if self.dist is not None:
            elem.attrib["distribution"] = str(self.dist)
        if self.valCol is not None:
            elem.attrib["valCol"] = str(self.valCol)
        if self.logScale is not None:
            elem.attrib["logScale"] = str(self.logScale)
        elif self.scale is not None:
            elem.attrib["scale"] = str(self.scale)
        if self.shift is not None:
            elem.attrib["shift"] = str(self.shift)
        if self.caseSensitive is not None and\
          self.caseSensitive is not False:
            elem.attrib["caseSensitive"] = str(self.caseSensitive)
        if self.delta is True:
            elem.attrib["delta"] = "True"
        if self.defaultVal is not None:
            elem.attrib["default"] = str(self.defaultVal)
        if self.preprocess is not None:
            elem.attrib["preprocess"] = str(self.preprocess)
        return elem

    def getValueMap(self):
        return self.valMap

    def getNumber(self):
        return self.number

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getDist(self):
        return self.dist

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path

    def getValCol(self):
        return self.valCol

    def getScale(self):
        return self.scale

    def setScale(self, scale):
        self.scale = scale
        self.logScale = None

    def getLogScale(self):
        return self.logScale

    def setLogScale(self, logScale):
        self.logScale = logScale
        self.scale = None

    def getShift(self):
        return self.shift

    def setShift(self, shift):
        self.shift = shift

    def getCaseSensitive(self):
        return self.caseSensitive

    def getDelta(self):
        return self.delta

    def getDefaultVal(self):
        return self.defaultVal

    def getPreprocess(self):
        return self.preprocess
    
###########################################################################
"""list of tracks (see above) that we can index by name or number as well as
load from or save to a file. this strucuture needs to accompany a trained
model. """
class TrackList(object):
   def __init__(self, xmlPath = None, treatMaskAsBinary = False):
       #: list of tracks.  track.number = its position in this list
       self.trackList = []
       #: keep alignment track separate because it's special
       self.alignmentTrack = None
       #: mask tracks also kept seperately
       self.maskTrackList = []
       #: map a track name to its position in the list
       self.trackMap = dict()
       #: hack to have option to not give mask tracks special treatment
       self.treatMaskAsBinary = treatMaskAsBinary
       if xmlPath is not None:
           self.loadXML(xmlPath)

   def getTrackByName(self, name, isMask = False):
       if name in self.trackMap:
           trackIdx = self.trackMap[name]
           trackList = self.trackList
           if isMask is True:
               trackList = self.maskTrackList
           track = trackList[trackIdx]
           assert track.name == name
           assert track.number == trackIdx
           return track
       return None

   def getTrackByNumber(self, idx):
       if idx < len(self.trackList):
           assert self.trackList[idx].number == idx
           return self.trackList[idx]
       return None

   def getMaskTrackByNumber(self, idx):
       if idx < len(self.maskTrackList):
           assert self.maskTrackList[idx].number == idx
           return self.maskTrackList[idx]
       return None

   def getMaskTracks(self):
       return self.maskTrackList

   def getAlignmentTrack(self):
       return self.alignmentTrack

   def addTrack(self, track):
       if track.dist == "alignment":
           track.number = 0
           self.alignmentTrack = track
       elif track.dist == "mask" and not self.treatMaskAsBinary:
           track.number = len(self.maskTrackList)
           self.maskTrackList.append(track)
           assert track.name not in self.trackMap
           self.trackMap[track.name] = track.number
       else:
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
       alignmentCount = 0
       for child in root.findall("track"):
           track = Track(child)
           if track.dist == "alignment":
               alignmentCount += 1
           if alignmentCount > 1:
               raise RuntimeError("Only one track with alignment "
                                  "distribution permitted")
           self.addTrack(track)

   def saveXML(self, path):
       root = ET.Element("teModelConfig")
       for track in itertools.chain(self.trackList, self.maskTrackList):
           root.append(track.toXMLElement())
       if self.alignmentTrack is not None:
           root.append(self.alignmentTrack.toXMLElement())
       x = xml.dom.minidom.parseString(ET.tostring(root))
       pretty_xml_as_string = x.toprettyxml()
       f = open(path, "w")
       f.write(pretty_xml_as_string)
       f.close()
       
   def __check(self):
       for i,track in enumerate(self.trackList):
           assert track.number == i
           assert track.name in self.trackMap
       assert len(self.trackMap) == len(self.trackList) + \
         len(self.maskTrackList)

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
        self.origEnd = end
        assert end > start
        #: offsets used for segmentation (optional)
        self.segOffsets = None
        #: mimic numpy array
        self.shape = (len(self), self.getNumTracks())

    def __len__(self):
        """ Number of columns in the table """
        if self.segOffsets is None:
            return self.end - self.start
        else:
            return len(self.segOffsets)

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

    def getOverlapInTableCoords(self, bedInterval, startHint = None):
        """ Compute overlap with a bed coordinate. return None if do not
        intersect. Note that output coordinates are relative to the table,
        accounting for segmentation if present, and not genome coordinates.
        The input is a regular bed region in genome coordinates....
        """
        assert len(bedInterval) > 2
        overlap = None
        chrom, start, end = bedInterval[0], bedInterval[1], bedInterval[2]
        if self.chrom == chrom and self.start < end and self.end > start:
            overlap = [self.chrom, max(self.start, start), min(self.end, end)]
            for i in xrange(3, len(bedInterval)):
                overlap.append(bedInterval[i])

            if self.segOffsets is not None:
                genStart = overlap[1]
                genEnd = overlap[2]
                overlap[1] = None
                overlap[2] = None
                firstLook = 0
                if startHint is not None and\
                       startHint < len(self.segOffsets) and\
                       genStart >= self.start + self.segOffsets[startHint]:
                    firstLook = startHint
                for j, so in enumerate(self.segOffsets[firstLook:]):
                    i = j + firstLook
                    if overlap[1] is None and\
                       genStart >= self.start + so and\
                       genStart < self.start + so + self.getSegmentLength(i):
                        overlap[1] = i
                    if overlap[1] is not None and\
                       genEnd > self.start + so and\
                       genEnd <= self.start + so + self.getSegmentLength(i):
                        overlap[2] = i + 1
                    if overlap[1] is not None and overlap[2] is not None:
                        break
                assert overlap[1] is not None and overlap[2] is not None
            else:
                # map to table coordintes
                overlap[1] -= self.start
                overlap[2] -= self.start
            
        return overlap

    def getChrom(self):
        return self.chrom
    
    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    def getNumPyArray(self):
        raise RuntimeError("Not implemented")

    def getSegmentOffsets(self):
        return self.segOffsets

    def segment(self, segIntervals, trackList, interpolate=True):
        """ completely transform table to contain only one coordinate per
        segment interval (compression).  """
                
        firstIdx = binSearch(segIntervals, (self.chrom, self.start), [0,1])
        lastIdx = binSearch(segIntervals, (self.chrom, self.origEnd), [0,2])

        assert firstIdx is not None
        assert lastIdx is not None

        maskTransform = self.getMaskRunningOffsets(reverseTransform = True)
        self.segOffsets = np.zeros((1 + lastIdx - firstIdx), np.int)
        j = 0
        logger.info("Computing mask and segment offsets in table with shape %s" %
                    str(self.shape))
        for i in xrange(firstIdx, lastIdx + 1):
            offset = int(segIntervals[i][1]) - self.start
            if maskTransform is not None:
                # segment should be either all masksed or not all masked
                # assertion failure here means mask segment were not cut
                segLen = segIntervals[i][2] - segIntervals[i][1]
                if self.maskArray[offset] == True:
                    assert maskTransform[offset] == \
                      maskTransform[offset + segLen - 1]
                    offset -= maskTransform[offset]
                else:
                    # segment is masked
                    continue
            self.segOffsets[j] = offset
            j += 1
        if j != len(self.segOffsets):
            assert maskTransform is not None
            self.segOffsets = self.segOffsets[:j]

        if len(self.segOffsets) == 0:
            # we've masked out the whole table!
            assert maskTransform is not None
            self.shape = (0, self.getNumTracks())
            return
        
        if interpolate is True:
            logger.info("Interpolating segments (shape=%s)" % str(self.shape))
            self.interpolateSegments(trackList)
        logger.info("Compressing the track table...")
        self.compressSegments()
        self.shape = (len(self), self.getNumTracks())
        assert len(self.segOffsets) > 0

    def getSegmentLength(self, i):
        """ get the length of a segment corresponding to a given index """
        if i == len(self.segOffsets) - 1:
            return self.end - (self.start + self.segOffsets[-1])
        elif i < len(self.segOffsets) - 1:
            return self.segOffsets[i+1] - self.segOffsets[i]

    def getSegmentLengthsAsRatio(self, effectiveSegmentLength):
        """ return the segment length / effective segment length as an array"""
        if self.segOffsets is None:
            return None
        seglens = np.ndarray((len(self)), dtype=np.float)
        effectiveSegmentLength = float(effectiveSegmentLength)
        assert effectiveSegmentLength >= 1
        for i in xrange(len(self)):
            seglens[i] = float(self.getSegmentLength(i)) / effectiveSegmentLength
        return seglens

    def interpolateSegments(self, trackList):
        """ A segment interval (i, j) in the original data, A,  is mapped to just
        A[i] in the segmented data.  Here we scan over the entire segment and put
        an average value into the first entriy (A[i]).  We use mode as it can
        apply to both numerical and categorical data"""
        # make a faster mapback table for each gaussian track
        mbTables = dict()
        for track in trackList:
            if track.getDist() == "gaussian":
                mbTables[track.getNumber()] = \
                  track.getValueMap().getMapBackTable(INTEGER_ARRAY_TYPE)
            else:
                mbTables[track.getNumber()] = None

        for so in xrange(len(self.segOffsets)):
            segLen = self.getSegmentLength(so)
            start = self.segOffsets[so]
            end = start + segLen
            self.setAverages(start, start, end, trackList, mbTables)

    def setMaskTable(self, maskTable):
        """ Pair a table with a mask table (of binary-style mask tracks) of
        same type. """
        assert False

    def hasMask(self):
        assert False
        
###########################################################################

"""Track Table where every value is an integer"""
class IntegerTrackTable(TrackTable):
    """ Note: we consider each row as an array corresponding to a single track
    for the purposes of this interface (ie writeRow, getRow etc).  Internally,
    the rows are stored in array columns, because we want quicker access to
    data columns for the HMM interface.  Ie, value for each track at a given base
    """
    def __init__(self, numTracks, chrom, start, end, dtype=INTEGER_ARRAY_TYPE):
        super(IntegerTrackTable, self).__init__(numTracks, chrom, start, end)
        #: (end-start) X (numTracks) integer data array
        self.data = np.zeros((end-start, numTracks), dtype=dtype)
        self.iinfo = np.iinfo(dtype)
        self.segOffsets
        self.maskArray = None

    def __getitem__(self, index):
        return self.data[index]

    def writeRow(self, row, rowArray):
        """ write exactly one full row of data values to the table, mapping
        each value using valueMap if it's specified """
        assert row < self.getNumTracks()
        assert len(rowArray) == len(self)
        for i in xrange(len(self)):
            if rowArray[i] > self.iinfo.max:
                logger.warning("Clamping input value %d of track# %d"
                               " from %d to %d\n" % (i, row, rowArray[i],
                                                       self.iinfo.max))
                self.data[i][row] = self.iinfo.max
            elif rowArray[i] < self.iinfo.min:
                logger.warning("Clamping input value %d of track# %d"
                               " from %d to %d\n" % (i, row, rowArray[i],
                                                       self.iinfo.min))
                self.data[i][row] = self.iinfo.min
            else:
                self.data[i][row] = rowArray[i]

    def getNumPyArray(self):
        return self.data

    def getRow(self, row):
        assert row < self.data.shape[1]
        rowArray = self.data[:,row]
        assert rowArray is not None
        return rowArray

    def initRow(self, row, val):
        self.data[:,row] = val

    def compressSegments(self):
        """ cut up data so that only one value per segment """
        assert self.segOffsets is not None and len(self.segOffsets) > 0
        oldShape = self.data.shape
        self.data = self.data[self.segOffsets]
        newShape = self.data.shape
        assert newShape[0] == len(self.segOffsets)
        assert_array_equal(oldShape[1:], newShape[1:])

    def setAverages(self, pos, start, end, trackList, mbTables):
        """ set position pos to mode of range from [start, end) for all
        tracks except gaussian distributions, where we use mean isntead of
        mode """
        for track in trackList:
            trackNo = track.getNumber()
            if track.getDist() == "gaussian":
                mbTable = mbTables[trackNo]
                assert mbTable is not None
                valMap = track.getValueMap()
                total = 0.
                for x in xrange(start, end):
                    total += mbTable[self.data[x, trackNo]]
                meanVal = total / (end-start)
                self.data[pos, trackNo] = valMap.getMap(meanVal, update=True)
            else:
                self.data[pos, trackNo] = mode(self.data[start:end,
                                                         trackNo])[0][0]

    def setMaskTable(self, maskTable):
        """ take in another integertable, but expect it to contain only
        binary tracks to be used as a mask (ie everything not covered cut
        out).  The data is then masked out """
        self.maskArray = None
        if maskTable is not None:
            # logic below:  in a binaryCategoryMap (which mask tracks use)
            # False is 1 and True is 2. So here we collapse all mask tracks
            # into one with an OR operation, so if at least one track is
            # true (2), then the sum will be more then the length
            self.maskArray = sum(maskTable.data.T) <= maskTable.shape[1]

            # do the masking.  we artificially shrink the table's end
            # coordinate.  mask array should have False for regions we
            # want to mask out. 
            oldShape = self.shape
            self.data = self.data[self.maskArray]
            self.shape = self.data.shape
            assert self.shape[0] <= oldShape[0]
            assert self.shape[1] == oldShape[1]
            logger.debug("masked table %s -> %s" % (str(oldShape),
                                                    str(self.shape)))
            self.origEnd = self.end            
            self.end = self.start + self.shape[0]

    def hasMask(self):
        return self.maskArray is not None

    def getMaskRunningOffsets(self, reverseTransform = False):
        """ get a 1-dimensional array (that spans masked table) where the ith
        value is the number of masked bases before position i in the table.
        this can be used to reverse the masking compression. """
        if self.hasMask() is True:
            # compute running offsets on uncompressed dimensions
            runningOffsets = np.zeros(self.maskArray.shape, np.int32)
            runSum(self.maskArray.astype(np.uint8), runningOffsets)
            if not reverseTransform:
                # mask the offsets so they are relative to masked coordates
                runningOffsets = runningOffsets[self.maskArray]
            return runningOffsets
        return None
        
            
###########################################################################
            
""" map a value to an integer category """
class CategoryMap(object):
    def __init__(self, reserved = 1, defaultVal = None, scale=None,
                 logScale=None, shift=None):
        self.catMap = dict()
        self.catMapBack = dict()
        self.reserved = reserved
        self.scaleFac = None
        self.logScaleBase = None
        self.logScaleDiv = None
        self.shift = None
        self.defaultVal = defaultVal
        self.missingVal = max(0, self.reserved - 1)
        if logScale is not None:
            self.__setLogScale(logScale)
        elif scale is not None:
            self.__setScale(scale)
        if shift is not None:
            self.__setShift(shift)
        # Note: scale needs to be set before missingVal (because getMap used)
        # which is way setScale methods made private and scaling now only
        # passed in constructor
        if self.defaultVal is not None:
            self.missingVal = int(self.getMap(self.defaultVal, update = True))
        else:
            self.missingVal = max(0, self.reserved - 1)
        
    def update(self, inVal):
        val = self.__scale(inVal)
        if val not in self.catMap:
            newVal = len(self.catMap) + self.reserved
            assert val not in self.catMap
            self.catMap[val] = newVal
            self.catMapBack[newVal] = val
        
    def has(self, inVal):
        val = self.__scale(inVal)
        return val in self.catMap
        
    def getMap(self, inVal, update = False):
        val = self.__scale(inVal)
        if val is not None and update is True and val not in self.catMap:
            self.update(inVal)
        if val in self.catMap:
            return self.catMap[val]
        return self.getMissingVal()

    def getMapBack(self, val):
        if val in self.catMapBack:
            return self.__scaleInv(self.catMapBack[val])
        elif self.defaultVal is not None:
            return self.__scaleInv(self.catMapBack[
                self.getMap(self.defaultVal)])
        else:
            return None

    def getMapBackTable(self, dtype):
        """ doing getMapBack millions of times is super slow (ie
        when interpolating gaussian segments. speed up with a table"""
        tsize = np.iinfo(dtype).max + 1
        table = np.zeros((tsize,), dtype=np.float) + sys.maxint
        for i in xrange(len(table)):
            val = self.getMapBack(i)
            if val is not None:
                table[i] = val
        return table

    def getMissingVal(self):
        return self.missingVal

    def getReserved(self):
        return self.reserved

    def __len__(self):
        return len(self.catMap) + max(0, self.reserved - 1)

    def __setScale(self, scale):
        self.scaleFac = scale
        self.logScaleBase = None
        
    def __setLogScale(self, logScale):
        self.logScaleBase = logScale
        assert self.logScaleBase != 0.0
        self.logScaleDiv = math.log(self.logScaleBase)
        self.scaleFac = None

    def __setShift(self, shift):
        self.shift = float(shift)

    def sort(self):
        """ sort dictionary so that value1 < value2 iff key1 < key2 """
        oldMap = self.catMap
        self.catMap = dict()
        self.catMapBack = dict()
        keys = oldMap.keys()
        # sort by numeric value whenever possible, otherwise resort to lex
        try:
            numericKeys = [float(key) for key in oldMap.keys()]
        except:
            numericKeys = keys
        for numericKey, key in sorted(zip(numericKeys, keys)):
            newVal = len(self.catMap) + self.reserved
            self.catMap[key] = newVal
            self.catMapBack[newVal] = key
        assert len(oldMap) == len(self.catMap)
        assert len(self.catMap) == len(self.catMapBack)


    def __scale(self, x):
        y = x
        if self.shift is not None:
            y = float(y) + self.shift
        if self.scaleFac is not None:
            return str(int(self.scaleFac * float(y)))
        elif self.logScaleBase is not None:
            assert y >= 0.0
            return str(int(math.log(float(y)) / self.logScaleDiv))
        return y

    def __scaleInv(self, x):
        y = x
        if self.scaleFac is not None:
            y = float(x) / float(self.scaleFac)
        elif self.logScaleBase is not None:
            y = math.pow(self.logScaleBase, float(x))
        if self.shift is not None:
            y = float(y) - self.shift
        return y

    
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
            return 2
        return 1

    def getMapBack(self, val):
        return val - 1

    def getMissingVal(self):
        return 1

    def __len__(self):
        return 2

###########################################################################
    
""" Data Array formed by a series of tracks over the same coordinates of the
same genomes.  Multiple intervals are supported. """
class TrackData(object):
    def __init__(self, dtype=INTEGER_ARRAY_TYPE):
        #: list of tracks (of type TrackList)
        self.trackList = None
        #: list of track tables (of type TrackTable)
        self.trackTableList = None
        #: separate list for alignment track because it's so specual
        self.alignmentTrackTableList = None
        #: datatype for array values in observation matrix
        # (lower the better since memory adds up quickly)
        self.dtype = dtype
        #: special datatype for alignment arrays
        self.adtype = np.uint16

    def getNumTracks(self):
        return len(self.trackList)

    def getTrackList(self):
        return self.trackList

    def getTrackTableList(self):
        return self.trackTableList

    def getAlignmentTrackTableList(self):
        return self.alignmentTrackTableList

    def getNumTrackTables(self):
        return len(self.trackTableList)

    def getNumSymbolsPerTrack(self):
        nspt = [0] * self.getNumTracks()        
        for i in xrange(self.getNumTracks()):
            track = self.trackList.getTrackByNumber(i)
            nspt[i] = len(track.getValueMap())
        return nspt
    
    def loadTrackData(self, trackListPath, intervals, trackList = None,
                      segmentIntervals = None, interpolateSegments=True,
                      applyMasking = True, treatMaskAsBinary=False):
        """ load track data for list of given intervals.  tracks is either
        a TrackList object loaded from a saved pickle, or None in
        which case they will be generated from the data.  each interval
        is a 3-tuple of chrom,start,end"""
        assert len(intervals) > 0
        inputTrackList = TrackList(trackListPath, treatMaskAsBinary)
        if trackList is None:
            initTracks = True
            self.trackList = inputTrackList
            # note, need to make sure category maps get properly set
        else:
            initTracks = False
            self.trackList = trackList
        self.trackIdx = dict()

        self.trackTableList = []
        logger.debug("Loading track data for %d intervals" % len(intervals))
        self.alignmentTrackTableList = []
        for interval in intervals:
            assert len(interval) >= 3 and interval[2] > interval[1]
            self.__loadTrackDataInterval(inputTrackList, interval[0],
                                         interval[1], interval[2], initTracks,
                                         applyMasking)
            if segmentIntervals is not None:
                oldShape = self.trackTableList[-1].getNumPyArray().shape
                # remove masked out table
                if oldShape[0] == 0:
                    self.trackTableList.pop()
                else:
                    logger.info("Applying segmentation on track table with shape %s" %
                                str(self.trackTableList[-1].getNumPyArray().shape))
                    self.trackTableList[-1].segment(segmentIntervals,
                                                    self.trackList,
                                                    interpolate=interpolateSegments)
                    newShape = self.trackTableList[-1].getNumPyArray().shape
                    logger.info("Compressed track table from %s to %s" % (
                        str(oldShape), str(newShape)))

    def __loadTrackDataInterval(self, inputTrackList, chrom, start, end, init,
                                applyMasking):
        trackTable = IntegerTrackTable(self.getNumTracks(), chrom, start, end,
                                       dtype=self.dtype)
        maskTable = None
        numMaskTracks = len(inputTrackList.getMaskTracks())
        if numMaskTracks > 0:
            maskTable = IntegerTrackTable(numMaskTracks, chrom, start, end,
                                          dtype=self.dtype)
        
        for idx, inputTrack in enumerate(itertools.chain(inputTrackList,
                                          inputTrackList.getMaskTracks())):
            isMask = idx >= len(inputTrackList)
            trackName = inputTrack.getName()
            trackPath = inputTrack.getPath()
            selfTrack = self.trackList.getTrackByName(trackName, isMask)
            if selfTrack is None and not isMask:
                logger.warning("track %s not learned\n" % trackName)
                continue
            if selfTrack is None and isMask:
                # hack to allow unlearned mask track (since training independent)
                track = inputTrack
                selfTrack = inputTrack
            else:
                track = self.getTrackList().getTrackByName(trackName, isMask)
            trackNo = track.getNumber()
            table = trackTable
            if isMask is True:
                table = maskTable
            table.initRow(trackNo, selfTrack.getValueMap().getMissingVal())
            readTrackData(trackPath, chrom, start, end,
                          valCol=inputTrack.getValCol(),
                          valMap=selfTrack.getValueMap(),
                          updateValMap=init,
                          caseSensitive=inputTrack.getCaseSensitive(),
                          outputBuf=table.getRow(trackNo),
                          useDelta=inputTrack.getDelta())
            
        if applyMasking is True:
            trackTable.setMaskTable(maskTable)
        self.trackTableList.append(trackTable)

        if self.trackList.getAlignmentTrack() is not None:
            inputTrack = self.trackList.getAlignmentTrack()
            alignmentTrackTable = IntegerTrackTable(1, chrom, start, end,
                                                    dtype = self.adtype)
            trackName = inputTrack.getName()
            trackPath = inputTrack.getPath()
            rowArray = readTrackData(trackPath, chrom, start, end,
                                     valCol=inputTrack.getValCol(),
                                     valMap=inputTrack.getValueMap(),
                                     updateValMap=True)
            alignmentTrackTable.writeRow(0, rowArray)
            self.alignmentTrackTableList.append(alignmentTrackTable)

            
            
        
            
        
        

        
