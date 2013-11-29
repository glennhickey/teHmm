#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
# Contains some fragments of code from sklearn/tests/test_hmm.py
# (2010 - 2013, scikit-learn developers (BSD License))
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal

from teHmm.tracksInfo import TracksInfo
from teHmm.track import *
from teHmm.hmm import MultitrackHmm
from teHmm.cfg import MultitrackCfg
from teHmm.emission import IndependentMultinomialEmissionModel

from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfoPath
from teHmm.tests.emissionTest import getBedStates

class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testInit(self):
        emissionModel = IndependentMultinomialEmissionModel(
            10, [3], zeroAsMissingData=False)
        cfg = MultitrackCfg(emissionModel)
        cfg = MultitrackCfg(emissionModel, [3,8])
        

        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

