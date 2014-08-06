#!/usr/bin/env python

#Copyright (C) 2014 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal
import random
import string

from teHmm.kmer import KmerTable, hashDNA

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
        assert em == [[0, 7, 18, 25]]
        em = kt.exactMatches("something hey")
        assert len(em) == 2
        assert [9,13,3,7] in em
        assert [10,13,0,3] in em

    def testMatch2(self):
        kt = KmerTable(kmerLen=3, hashFn = hashDNA)
        kt.loadString("CCAGAT")
        em = kt.exactMatches("ATGTTCACTCTTATCCAGAT")
        assert em == [[14, 20, 0, 6]]
        kt.loadString("ATGTTCACTCTTATCCAGAT")
        em = kt.exactMatches("CCAGAT")
        assert em == [[0, 6, 14, 20]]


    def testRandom(self):
        S = ["a", "c", "g", "t"]
        for repeat in xrange(5):
            sa = ''.join(random.choice(S) for x in xrange(50))
            sb = ''.join(random.choice(S) for x in xrange(50))
            kt = KmerTable(hashFn = hashDNA)
            kt.loadString(sa)
            em = kt.exactMatches(sb)
            # doesnt check for any missed matches but will at least test
            # robustness of given matches
            for match in em:
                assert sb[match[0]:match[1]] == sa[match[2]:match[3]]

            # try again without "optimization"
            kts = KmerTable()
            kts.useClosed = False
            kts.loadString(sa)
            assert kts.exactMatches(sb) == em

    def testHashFn(self):
        numTestsPerLen = 50000
        lengths = [3, 6, 10, 20]
        S = ['A', 'a', 'C', 'c', 'G', 'g', 'T', 't', 'N', 'n']
        for length in lengths:
            stringSet = set()
            intSet = set()
            for trial in xrange(numTestsPerLen):
                s = ''.join(random.choice(S) for x in xrange(length))
                v = hashDNA(s)
                stringSet.add(s.upper())
                intSet.add(v)
                assert len(intSet) == len(stringSet)
        

def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

