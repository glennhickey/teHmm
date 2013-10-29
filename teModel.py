#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
from tracksInfo import TrackInfo
from track.py import Track, BedTrack, TrackCategoryMap, TrackData

"""
Model for a transposable element.  Gets trained by a set of tracks.  
"""
def TEModel(object):
    def __init__(self):
        #: list of tracks
        self.tracks = None
        #: annotation array
        self.trackData = None
        
    def initTracks(self, tracksInfoPath):
        """either the TEModel is already trained and loaded from a pickle,
        otherwise we need to initialize it with this function"""
        tinfo = TracksInfo(tracksInfoPath)
        self.tracks = dict()
        for trackName, trackPath in tinfo.pathMap:
            # todo: specify valueMap as input option or field in TracksInfo
            # instead of hardcoding here
            track = Track(trackName, len(self.tracks) - 1, TrackCategoryMap())
            self.tracks[trackNAme] = track

    def checkTrackNumbers(self):
        assert self.tracks is not None
        numbers = dict()
        for track in self.tracks:
            assert track.number not in numbers
            numbers[track.number] = True

    def loadTrackData(self, tracksInfoPath, seqName, start, end,
                      forTraining = False):
        """load up a slice of track data"""
        assert self.tracks is not None and len(self.tracks) > 0
        self.checkTrackNumbers()
        numTracks = len(self.tracks)
        trackLen = end - start
        if self.trackData is None or self.trackData.shape[0] != numTracks \
           or self.trackData.shape[1] != trackLen:
            # note will need to review hardcoding of np.int here
            self.trackData = np.empty((numTracks, trackLen), np.int)
        for x in np.nditer(self.trackData, op_flags='write'):
            x[...] = 0

        tinfo = TracksInfo(tracksInfoPath)
        for trackName, trackPath in tinfo.pathMap:
            if trackExt != ".bed":
                sys.stderr.print("Warning: non-BED file skipped %s" %
                                 trackPath)
                continue
            if trackName not in self.tracks:
                sys.stderr.print("Warning: track %s not learned" %
                                 trackName)
                continue
            track = self.tracks[trackName]
            trackData = BedTrackData(seqName, start, end, self.data, track)
            trackData.loadBedInterval(trackPath, False, forTraining)
            
        
