#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from tracksInfo import TracksInfo
from track import Track, BedTrackData, TrackCategoryMap

"""
Model for a transposable element.  Gets trained by a set of tracks.  
"""
class TEModel(object):
    def __init__(self):
        #: list of tracks
        self.tracks = None
        #: annotation array
        self.data = None
        
    def initTracks(self, tracksInfoPath):
        """either the TEModel is already trained and loaded from a pickle,
        otherwise we need to initialize it with this function"""
        tinfo = TracksInfo(tracksInfoPath)
        self.tracks = dict()
        for trackName, trackPath in tinfo.pathMap.items():
            # todo: specify valueMap as input option or field in TracksInfo
            # instead of hardcoding here
            track = Track(trackName, len(self.tracks) - 1, TrackCategoryMap())
            self.tracks[trackName] = track

    def checkTrackNumbers(self):
        assert self.tracks is not None
        numbers = dict()
        for trackName, track in self.tracks.items():
            assert track.number not in numbers
            numbers[track.number] = True

    def load(self, modelPath):
        modelFile = open(modelPath,'rb')
        tmp_dict = pickle.load(modelFile)
        modelFile.close()                  
        self.__dict__.update(tmp_dict) 

    def save(self, modelPath):
        dataTemp = self.data
        self.data = None
        modelFile = open(modelPath,'wb')
        pickle.dump(self.__dict__, modelFile, 2)
        modelFile.close()
        self.data = dataTemp
        
    def loadTrackData(self, tracksInfoPath, seqName, start, end,
                      forTraining = False):
        """load up a slice of track data"""
        assert self.tracks is not None and len(self.tracks) > 0
        self.checkTrackNumbers()
        numTracks = len(self.tracks)
        trackLen = end - start
        if self.data is None or self.data.shape[1] != numTracks \
           or self.data.shape[0] != trackLen:
            # note will need to review hardcoding of np.int here
            self.data = np.empty((trackLen, numTracks), np.int)
        for x in np.nditer(self.data, op_flags=['readwrite']):
            x[...] = 0

        tinfo = TracksInfo(tracksInfoPath)
        for trackName, trackPath in tinfo.pathMap.items():
            trackExt = os.path.splitext(trackPath)[1]
            if trackExt != ".bed":
                sys.stderr.write("Warning: non-BED file skipped %s\n" %
                                 trackPath)
                continue
            if trackName not in self.tracks:
                sys.stderr.write("Warning: track %s not learned\n" %
                                 trackName)
                continue
            track = self.tracks[trackName]
            bedData = BedTrackData(seqName, start, end, self.data, track)
            bedData.loadBedInterval(trackPath, False, forTraining)

    def tracks(self):
        for trackName, track in self.tracks:
            yield track
        
