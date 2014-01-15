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
import random
import string

from teHmm.kmer import KmerTable

from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase
from teHmm.tests.bedTrackTest import getTracksInfoPath
from teHmm.tests.emissionTest import getBedStates


class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testLoad(self):
        s = "hey hey we're the monkees"
        kt = KmerTable()
        kt.loadString(s)
        assert kt.getKmer("hey") == [0, 4]
        assert kt.getKmer("y h") == [2]
        assert kt.getKmer("mon") == [18]
        assert kt.getKmer("key") == []        

    def testOverlap(self):
        kt = KmerTable()
        assert kt.getMerge([0, 4, 10, 14], [1, 5, 11, 15]) == [0, 5, 10, 15]
        assert kt.getMerge([0, 4, 10, 14], [1, 5, 12, 16]) == None
        assert kt.getMerge([9, 12, 3, 6], [10, 13, 4, 7]) == [9, 13, 3, 7]

    def testMatch(self):
        s = "hey hey we're the monkees"
        kt = KmerTable()
        kt.loadString(s)
        em = kt.exactMatches("monkees")
        assert em == [[0, 6, 18, 24]]
        em = kt.exactMatches("something hey")
        assert len(em) == 2
        assert [9,13,3,7] in em
        assert [10,13,0,3] in em

    def testRandom(self):
        S = ["a", "c", "g", "t"]
        for repeat in xrange(5):
            sa = ''.join(random.choice(S) for x in xrange(50))
            sb = ''.join(random.choice(S) for x in xrange(50))
            kt = KmerTable()
            kt.loadString(sa)
            em = kt.exactMatches(sb)
            # doesnt check for any missed matches but will at least test
            # robustness of given matches
            for match in em:
                assert sb[match[0]:match[1]] == sa[match[2]:match[3]]
        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

