#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os

from teHmm.tracksInfo import TracksInfo
from teHmm.track import *
from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfo
from teHmm.teHmmModel import TEHMMModel

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
        self.tiPath = self.getTempFilePath()
        ti = getTracksInfo()
        ti.save(self.tiPath)
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testInit(self):
        hmmModel = TEHMMModel()
        hmmModel.initTracks(self.tiPath)
        assert len(hmmModel.tracks) == getTracksInfo().getNumTracks()
        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

