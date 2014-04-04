#!/usr/bin/env python

#Copyright (C) 2013 by Glenn Hickey
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import numpy as np
import math
from numpy.testing import assert_array_equal, assert_array_almost_equal
import itertools
import ast

from teHmm.common import intersectSize
from teHmm.bin.compareBedStates import compareIntervalsOneSided
from teHmm.tests.common import getTestDirPath
from teHmm.tests.common import TestBase


class TestCase(TestBase):

    def setUp(self):
        super(TestCase, self).setUp()
    
    def tearDown(self):
        super(TestCase, self).tearDown()

    def testCompare(self):
        bed1Path = self.getTempFilePath()
        bed2Path = self.getTempFilePath()
        outputPath = self.getTempFilePath()
        bed1 = open(bed1Path, "w")
        bed1.write("chr1\t0\t10\tA\n")
        bed1.write("chr1\t10\t20\tB\n")
        bed1.write("chr1\t30\t40\tA\n")
        bed1.write("chr2\t10\t30\tC\n")

        bed2 = open(bed2Path, "w")
        bed2.write("chr1\t0\t15\tA\n")
        bed2.write("chr1\t15\t20\tB\n")
        bed2.write("chr1\t30\t40\tA\n")
        bed2.write("chr2\t10\t20\tD\n")
        bed2.write("chr2\t20\t30\tC\n")

        bed1.close()
        bed2.close()
        
        ret = os.system("compareBedStates.py %s %s > %s" % (bed1Path, bed2Path,
                                                            outputPath))
        assert ret == 0
        output = open(outputPath, "r")
        outputString = output.readline()
        outputStats = ast.literal_eval(outputString)

        assert outputStats['A'] == [0, 5, 20]
        assert outputStats['B'] == [5, 0, 5]
        assert outputStats['C'] == [10, 0, 10]
        assert outputStats['D'] == [0, 10, 0]

    def testIntersect(self):
        a = [0] * 6
        a[0] = ("b", 10, 100)
        a[1] = ("a", 10, 100)
        a[2] = ("a", 5, 15)
        a[3] = ("a", 11, 12)
        a[4] = ("a", 95, 105)
        a[5] = ("a", 0, 1000)

        for i in xrange(1, 6):
            assert intersectSize(a[0], a[i]) == 0
            assert intersectSize(a[i], a[0]) == 0

        assert intersectSize(a[1], a[2]) == 5
        assert intersectSize(a[2], a[1]) == 5

        assert intersectSize(a[1], a[3]) == 1
        assert intersectSize(a[3], a[1]) == 1
        
        assert intersectSize(a[1], a[4]) == 5
        assert intersectSize(a[4], a[1]) == 5

        assert intersectSize(a[1], a[5]) == 90
        assert intersectSize(a[5], a[1]) == 90

    def testCompareIntervals(self):
        bed1 = []
        bed1.append(("chr1", 0, 10, 'A'))
        bed1.append(("chr1", 10, 20, 'B'))
        bed1.append(("chr1", 30, 40, 'A'))
        bed1.append(("chr2", 10, 30, 'C'))

        bed2 = []
        bed2.append(("chr1", 0, 15, 'A'))
        bed2.append(("chr1", 15, 20, 'B'))
        bed2.append(("chr1", 30, 40, 'A'))
        bed2.append(("chr2", 10, 20, 'D'))
        bed2.append(("chr2", 20, 30, 'C'))

        stats = compareIntervalsOneSided(bed1, bed2, 3, 0.8)

        trueA, falseA = stats['A'][0], stats['A'][2]
        assert trueA == 2
        assert falseA == 0

        trueB, falseB = stats['B'][0], stats['B'][2]
        assert trueB == 0
        assert falseB == 1

        trueC, falseC = stats['C'][0], stats['C'][2]
        assert trueC == 0
        assert falseC == 1

        assert 'D' not in stats

        stats = compareIntervalsOneSided(bed1, bed2, 3, 0.51)

        trueA, falseA = stats['A'][0], stats['A'][2]
        assert trueA == 2
        assert falseA == 0

        trueB, falseB = stats['B'][0], stats['B'][2]
        assert trueB == 0
        assert falseB == 1

        trueC, falseC = stats['C'][0], stats['C'][2]
        assert trueC == 0
        assert falseC == 1

        assert 'D' not in stats

    
        stats = compareIntervalsOneSided(bed1, bed2, 3, 0.5)

        trueA, falseA = stats['A'][0], stats['A'][2]
        assert trueA == 2
        assert falseA == 0

        trueB, falseB = stats['B'][0], stats['B'][2]
        assert trueB == 1
        assert falseB == 0

        trueC, falseC = stats['C'][0], stats['C'][2]
        assert trueC == 1
        assert falseC == 0

        assert 'D' not in stats

        stats = compareIntervalsOneSided(bed2, bed1, 3, 0.8)

        trueA, falseA = stats['A'][0], stats['A'][2]
        assert trueA == 1
        assert falseA == 1

        trueB, falseB = stats['B'][0], stats['B'][2]
        assert trueB == 1
        assert falseB == 0

        trueC, falseC = stats['C'][0], stats['C'][2]
        assert trueC == 1
        assert falseC == 0

        trueD, falseD = stats['D'][0], stats['D'][2]
        assert trueD == 0
        assert falseD == 1
        
        
        
        
def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()

