#!/usr/bin/env python

#Copyright (C) 2012 by Glenn Hickey (hickey@soe.ucsc.edu)
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os

def getTestDirPath(path = None):
    testPath = os.path.abspath('./tests/data')
    if path is not None:
        assert path[0] != '/'
        testPath = os.path.join(testPath, path)
    return path
