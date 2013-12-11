#!/usr/bin/env python

#Copyright (C) 2012 by Glenn Hickey (hickey@soe.ucsc.edu)
#
#Released under the MIT license, see LICENSE.txt
import unittest
import sys
import os
import tempfile

def getTestDirPath(path = None):
    testPath = os.path.abspath('./tests/data')
    if path is not None:
        assert path[0] != '/'
        testPath = os.path.join(testPath, path)
    return testPath

class TestBase(unittest.TestCase):

    def setUp(self):
        self.tempFiles = []
        super(TestBase, self).setUp()
    
    def tearDown(self):
        for tempFile in self.tempFiles:
            if os.path.isfile(tempFile):
                os.remove(tempFile)
        super(TestBase, self).tearDown()
        
    def getTempFile(self):
        if not os.path.exists('./tests/temp'):
            os.makedirs('./tests/temp')
        tempFile, tempPath = tempfile.mkstemp(dir='./tests/temp')
        self.tempFiles.append(tempPath)
        return tempFile, tempPath

    def getTempFilePath(self):
        tempFile, tempPath = self.getTempFile()
        os.close(tempFile)
        return tempPath
 
