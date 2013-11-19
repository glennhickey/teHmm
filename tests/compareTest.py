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
        
def main():
    sys.argv = sys.argv[:1]
    unittest.main()
        
if __name__ == '__main__':
    main()
